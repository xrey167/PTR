//! The arms of R010: the deterministic revocation scenario of expert `(j, i)`, the
//! nondeterministic-routing negative control, the quarantine (fence) arm through the real
//! authority path, and the 44-cell invalidation matrix on the MoE binding.

use super::params;
use crate::experiments::layered_kv::corpus::{self, CHUNK_TOKENS};
use crate::experiments::layered_kv::emit::{Json, Tally};
use crate::experiments::layered_kv::experts;
use crate::experiments::layered_kv::lab::{
    self, compare, fail, refused_not_live, Authority, Ids, Res,
};
use crate::experiments::layered_kv::matrix::{
    self, CellInfo, CellRecord, Invalidation, Kind, Observed, Path, Rig, SEGMENT_KINDS,
};
use crate::experiments::layered_kv::pins;
use crate::experiments::layered_kv::{Scale, SEQUENCE_SEED};
use ptr_ledger::ExpertFenceCause;
use ptr_model_api::{KvReuseHint, KvReuseMode, KvReusePolicy};
use ptr_pods::{ExpertState, Qwen2Moe, Qwen2MoeSpec, Qwen2RoutingMode};
use ptr_runtime::{
    BindingRecord, BindingSuccessors, BoundExpert, ExpertSuccessor, KvDependency, KvSession,
    MoeBinding, ReferenceKvError, ReferenceKvReuseBackend, StackBinding,
};
use ptr_types::{Generation, PodId, ProjectId};
use std::collections::BTreeSet;

const PROMPT_STREAM: u64 = 1_000;
const SECOND_PROMPT_STREAM: u64 = 1_001;
const EXTRA_STREAM: u64 = 2_000;
const PROMPT: usize = params::PROMPT_TOKENS as usize;
const LAYERS: usize = params::TEST_MODEL_LAYERS as usize;
const EXPERTS: usize = params::MOE_TOPOLOGY_NUM_EXPERTS as usize;
const J: usize = params::J;
/// Full pages of a run request.
/// The state version of a freshly reactivated generation.
const SUCCESSOR_FENCE_EPOCH: u64 = 1;
const PAGES: usize = PROMPT / CHUNK_TOKENS;

fn prompt() -> Vec<u32> {
    corpus::sequence(SEQUENCE_SEED, PROMPT_STREAM, PROMPT)
}

fn second_prompt() -> Vec<u32> {
    corpus::sequence(SEQUENCE_SEED, SECOND_PROMPT_STREAM, PROMPT)
}

/// The prompt extended by eight tokens (no new page boundary): the reuse probes' request.
fn extended(variant: u64) -> Vec<u32> {
    let mut tokens = prompt();
    tokens.extend(corpus::sequence(SEQUENCE_SEED, EXTRA_STREAM + variant, 8));
    tokens
}

/// Every sequence the run draws, in a fixed order: what the selection digest covers.
pub fn selection_sequences() -> Vec<Vec<u32>> {
    let mut sequences = vec![prompt(), second_prompt(), extended(1), extended(2)];
    for spec in SEGMENT_KINDS.iter().chain([&matrix::NEUTRAL]) {
        for offset in [0u64, 1, 7, 9, 11] {
            sequences.push(corpus::sequence(
                SEQUENCE_SEED,
                10_000 + spec.label + offset,
                16,
            ));
        }
    }
    sequences
}

pub fn selection_digest() -> String {
    corpus::selection_digest(params::SELECTION_DOMAIN, &selection_sequences())
}

fn raw(id: &str, tokens: &[u32]) -> ptr_model_api::SegmentedModelRequest {
    corpus::raw_request(id, tokens)
}

/// The set of `(layer, page)` pages, as a digest over their sorted names.
fn page_set_digest(pages: &BTreeSet<(usize, usize)>) -> String {
    let names: Vec<String> = pages
        .iter()
        .map(|(layer, page)| format!("l{layer}p{page}"))
        .collect();
    lab::id_set_digest(names.iter().map(String::as_str))
}

/// The MoE of the run and the expert it revokes.
pub struct World {
    pub moe: Qwen2Moe,
    /// The index `i` of the revoked expert of layer `j`: the most-routed one on the probe
    /// prompt.
    pub index: usize,
    /// Tokens the probe prompt routed to it at layer `j`.
    pub routed_tokens: u64,
    /// Whether the index is the one the preregistration pins.
    pub matches_pinned: bool,
}

fn moe_of(backend: &ReferenceKvReuseBackend) -> Res<Qwen2Moe> {
    Qwen2Moe::seeded(
        backend.stack_config(),
        &Qwen2MoeSpec::pinned(params::MOE_SEED, LAYERS),
    )
    .map_err(fail("seeding the MoE"))
}

impl World {
    /// Seed the MoE and pick `i`; a revoked expert that is never routed to would make the run
    /// vacuous, so that is an error (exit status 2).
    pub fn new() -> Res<Self> {
        let backend = lab::backend()?;
        let moe = moe_of(&backend)?;
        let stack = pins::reference_stack()?
            .with_moe(moe.clone())
            .map_err(fail("the MoE stack"))?;
        let counts = pins::routing_trace(&stack, &prompt())?;
        let index = counts
            .most_routed(J)
            .ok_or_else(|| "no expert of layer j was routed to on the probe prompt".to_string())?;
        let routed_tokens = counts.selected(J, index);
        if routed_tokens == 0 {
            return Err("the revoked expert is never routed to: the run would be vacuous".into());
        }
        Ok(Self {
            moe,
            index,
            routed_tokens,
            matches_pinned: index as u64 == params::REVOKED_EXPERT_INDEX,
        })
    }

    /// Every live expert of every MoE layer, each recording the fence epoch (the state version)
    /// the registry holds for it now, so a later transition of any kind ends the dependency.
    fn tracked(&self, authority: &Authority) -> Vec<BoundExpert> {
        (0..LAYERS)
            .flat_map(|layer| {
                (0..EXPERTS).map(move |index| BoundExpert {
                    layer,
                    index,
                    project: ProjectId::from(lab::PROJECT),
                    pod: experts::pod_of(layer, index),
                    fence_epoch: experts::state_of(&authority.runtime, layer, index)
                        .map_or(0, |(_, _, version)| version),
                })
            })
            .collect()
    }

    /// The live binding. With `successors`, a supersession of `(j, i)` moves to generation 2.
    /// A reactivated generation starts at state version 1 (the registry's transition epoch at
    /// the commit that admitted it), which is the successor's own fence epoch.
    pub fn binding(&self, authority: &Authority, successors: bool) -> StackBinding {
        let binding = StackBinding::with_moe(MoeBinding {
            moe: self.moe.clone(),
            experts: self.tracked(authority),
        });
        if successors {
            binding.with_successors(BindingSuccessors {
                adapters: Vec::new(),
                experts: vec![ExpertSuccessor {
                    layer: J,
                    index: self.index,
                    generation: 2,
                    fence_epoch: SUCCESSOR_FENCE_EPOCH,
                }],
            })
        } else {
            binding
        }
    }

    /// The router with expert `(j, i)` masked: what a restart re-resolves a revoked or fenced
    /// expert to.
    pub fn masked(&self, authority: &Authority) -> StackBinding {
        let moe = self
            .moe
            .clone()
            .set_expert_live(J, self.index, false)
            .expect("the expert exists");
        StackBinding::with_moe(MoeBinding {
            moe,
            experts: self.tracked(authority),
        })
    }

    /// The binding of generation 2 of `(j, i)`, read after the supersession committed.
    pub fn successor(
        &self,
        backend: &ReferenceKvReuseBackend,
        authority: &Authority,
    ) -> StackBinding {
        let moe = self
            .moe
            .clone()
            .with_expert_generation(backend.stack_config(), J, self.index, 2)
            .expect("the expert exists");
        StackBinding::with_moe(MoeBinding {
            moe,
            experts: self.tracked(authority),
        })
    }

    /// The negative control: capacity-limited, batch-variant routing.
    pub fn variant(&self, authority: &Authority) -> StackBinding {
        let moe = self
            .moe
            .clone()
            .with_mode(Qwen2RoutingMode::BatchVariantCapacity {
                capacity: params::BATCH_VARIANT_CAPACITY as usize,
            })
            .expect("a non-zero capacity");
        StackBinding::with_moe(MoeBinding {
            moe,
            experts: self.tracked(authority),
        })
    }
}

/// Birth every expert of every layer through the real admission path.
pub fn birth_all(authority: &mut Authority) -> Res<()> {
    for layer in 0..LAYERS {
        for index in 0..EXPERTS {
            experts::birth(&mut authority.runtime, layer, index)?;
        }
    }
    Ok(())
}

fn begin<'a>(
    backend: &'a ReferenceKvReuseBackend,
    request: &ptr_model_api::SegmentedModelRequest,
    binding: &StackBinding,
    authority: &Authority,
) -> Result<KvSession<'a>, ReferenceKvError> {
    backend.begin_session_bound(&lab::scope(), request, None, binding, authority)
}

fn complete(
    session: &mut KvSession<'_>,
    authority: &Authority,
) -> Res<ptr_runtime::ReferenceKvOutcome> {
    match session.run_to_end(authority).map_err(fail("a run"))? {
        ptr_runtime::KvSessionFinish::Completed(outcome) => Ok(*outcome),
        ptr_runtime::KvSessionFinish::Cancelled(_) => Err("a session was cancelled".to_string()),
    }
}

/// `(retained, newly_shared)`: page layers `l` in `layers` whose first `PAGES` pages share a
/// whole page of rows with `other`.
fn whole_pages_shared(
    session: &KvSession<'_>,
    other: &KvSession<'_>,
    layers: std::ops::Range<usize>,
) -> usize {
    layers
        .map(|layer| {
            (0..PAGES)
                .filter(|page| session.shared_leading_rows(other, *page, layer) == CHUNK_TOKENS)
                .count()
        })
        .sum()
}

/// Page layers of `layers` that share any row with `other`.
fn any_pages_shared(
    session: &KvSession<'_>,
    other: &KvSession<'_>,
    layers: std::ops::Range<usize>,
) -> usize {
    layers
        .map(|layer| {
            (0..PAGES)
                .filter(|page| session.shared_leading_rows(other, *page, layer) > 0)
                .count()
        })
        .sum()
}

/// The deterministic revocation scenario (the hypothesis' main claim) and the nondeterministic
/// negative control.
pub fn deterministic_arm(tally: &mut Tally, world: &World) -> Res<()> {
    let backend = lab::backend()?;
    let mut authority = Authority::new(&[])?;
    birth_all(&mut authority)?;
    let binding = world.binding(&authority, true);
    let masked = world.masked(&authority);
    let (p, p72, p72b) = (prompt(), extended(1), extended(2));

    // Non-vacuity: masking the expert changes the output.
    let cold_full = lab::cold(&backend, &raw("det-cold", &p), &binding)?;
    let cold_masked = lab::cold(&backend, &raw("det-cold-masked", &p), &masked)?;
    if cold_full.logits_digest == cold_masked.logits_digest {
        return Err("masking the revoked expert does not change the logits".to_string());
    }
    tally.set("revoked_expert_routed_tokens", world.routed_tokens as i64);

    // The pages are cached on both sides of layer j.
    let mut pre = begin(&backend, &raw("det-pre", &p), &binding, &authority)
        .map_err(fail("the pre-revocation session"))?;
    let out = complete(&mut pre, &authority)?;
    compare(tally, &out, &cold_full);
    let cached = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries;
    if cached != LAYERS * PAGES {
        return Err(format!(
            "the warm run cached {cached} entries, not {}",
            LAYERS * PAGES
        ));
    }
    let above = (LAYERS - 1 - J) * PAGES;
    let at_or_below = (J + 1) * PAGES;
    tally.add("pages_above_j_before_revocation", above as i64);
    tally.add("pages_at_or_below_j_before_revocation", at_or_below as i64);
    tally.add("pages_above_layer_cached_before_revocation", above as i64);
    // Derived from the structure the run's own insert count confirmed: every layer's page of
    // every chunk above j.
    let cached_above: BTreeSet<(usize, usize)> = ((J + 1)..LAYERS)
        .flat_map(|layer| (0..PAGES).map(move |page| (layer, page)))
        .collect();

    // A deterministic reuse event: the extended prompt forks all 64 cached tokens.
    let reuse_request = raw("det-reuse", &p72);
    let mut reuse =
        begin(&backend, &reuse_request, &binding, &authority).map_err(fail("the reuse session"))?;
    if reuse.hit_tokens() > 0 {
        tally.add("deterministic_reuse_events", 1);
    }
    let reuse_out = complete(&mut reuse, &authority)?;
    compare(
        tally,
        &reuse_out,
        &lab::cold(&backend, &reuse_request, &binding)?,
    );
    tally.add(
        "adoption_mismatches",
        reuse_out
            .layered
            .as_ref()
            .map_or(0, |l| l.adoption_mismatches) as i64,
    );
    drop(reuse);

    nondeterministic_control(tally, world, &authority, &backend, &p)?;

    // The revocation of expert (j, i): a committed Retire; the catalog learns of it.
    let mark = authority.mark();
    experts::retire(&mut authority.runtime, J, world.index)?;
    lab::apply_events(&backend, &authority.events_since(mark))?;
    tally.add("deterministic_revocation_attempts", 1);
    tally.add("expert_generation_revocations_performed", 1);
    let after = backend
        .catalog_stats()
        .map_err(fail("catalog stats"))?
        .entries;
    if after != at_or_below {
        tally.add("matrix_selectivity_mismatches", 1);
    }

    // Probe 1: the masked router (the live binding) on the extended prompt. The pages at or
    // below j are reused, none above j.
    let mut first =
        begin(&backend, &reuse_request, &masked, &authority).map_err(fail("the masked probe"))?;
    tally.add("post_revocation_reuse_probes", 1);
    tally.add("matching_post_revocation_lookups", 1);
    let probed: BTreeSet<(usize, usize)> = ((J + 1)..LAYERS)
        .flat_map(|layer| (0..(p72.len() / CHUNK_TOKENS)).map(move |page| (layer, page)))
        .collect();
    tally.add("pages_above_j_probed_after_revocation", probed.len() as i64);
    tally.set_json(
        "probed_page_set_digest",
        Json::str(page_set_digest(&probed)),
    );
    tally.set_json(
        "cached_page_set_digest_above_j",
        Json::str(page_set_digest(&cached_above)),
    );
    let out = complete(&mut first, &authority)?;
    // The physical sharing is read after the run: a partial hit is adopted at the end of the
    // prefill.
    let retained = whole_pages_shared(&first, &pre, 0..(J + 1));
    let stale = any_pages_shared(&first, &pre, (J + 1)..LAYERS);
    tally.add("retained_pages_reused_after_revocation", retained as i64);
    tally.add(
        "wrongly_dropped_pages",
        at_or_below.saturating_sub(retained) as i64,
    );
    tally.add("stale_pages_served", stale as i64);
    tally.add("stale_expert_generation_hits", stale as i64);
    compare(tally, &out, &lab::cold(&backend, &reuse_request, &masked)?);
    tally.add(
        "adoption_mismatches",
        out.layered.as_ref().map_or(0, |l| l.adoption_mismatches) as i64,
    );

    // Probe 2: a request citing the revoked expert's generation is refused before any lookup.
    let late = begin(&backend, &reuse_request, &binding, &authority);
    tally.add("post_revocation_reuse_probes", 1);
    tally.add("matching_post_revocation_lookups", 1);
    match &late {
        Ok(session) => tally.add(
            "stale_expert_generation_hits",
            1 + i64::from(session.hit_tokens() > 0),
        ),
        Err(_) if refused_not_live(&late) => {}
        Err(error) => return Err(format!("the late request: {error:?}")),
    }
    drop(late);

    // Probe 3: pages recomputed after the revocation are served to the next request.
    let recompute_request = raw("det-recompute", &p72b);
    let mut second = begin(&backend, &recompute_request, &masked, &authority)
        .map_err(fail("the recompute probe"))?;
    tally.add("post_revocation_reuse_probes", 1);
    tally.add("matching_post_revocation_lookups", 1);
    let served = whole_pages_shared(&second, &first, (J + 1)..LAYERS);
    tally.add("recomputed_pages_above_j_served", served as i64);
    let out = complete(&mut second, &authority)?;
    compare(
        tally,
        &out,
        &lab::cold(&backend, &recompute_request, &masked)?,
    );
    lab::note_scope(tally, &backend)?;
    Ok(())
}

/// The negative control: batch-variant routing is never tagged Exact, never touches the Exact
/// catalog, and its outputs diverge with the co-batch.
fn nondeterministic_control(
    tally: &mut Tally,
    world: &World,
    authority: &Authority,
    backend: &ReferenceKvReuseBackend,
    tokens: &[u32],
) -> Res<()> {
    let variant = world.variant(authority);
    let request = raw("nondet", tokens);
    let before = backend.catalog_stats().map_err(fail("catalog stats"))?;
    // Exact reuse is refused outright.
    tally.add("nondeterministic_reuse_attempts", 1);
    match backend.infer_scoped_bound(&lab::scope(), &request, None, &variant) {
        Err(ReferenceKvError::Binding(ptr_runtime::BindingError::ApproximateNotAllowed)) => {}
        Ok(out) => {
            if out
                .run
                .kv_report
                .as_ref()
                .is_some_and(|report| report.mode() == KvReuseMode::Exact)
            {
                tally.add("nondeterministic_exact_tags", 1);
            }
        }
        Err(error) => return Err(format!("the exact-only attempt: {error:?}")),
    }
    // With the policy that allows it, the run is Approximate and moves nothing.
    let hint = KvReuseHint {
        requested_prefix_tokens: 0,
        policy: KvReusePolicy::AllowApproximate,
    };
    tally.add("nondeterministic_reuse_attempts", 1);
    let out = backend
        .infer_scoped_bound(&lab::scope(), &request, Some(&hint), &variant)
        .map_err(fail("the approximate run"))?;
    if out
        .run
        .kv_report
        .as_ref()
        .is_none_or(|report| report.mode() == KvReuseMode::Exact)
    {
        tally.add("nondeterministic_exact_tags", 1);
    }
    let after = backend.catalog_stats().map_err(fail("catalog stats"))?;
    if after != before || out.hit_tokens != 0 || out.inserted_chunks != 0 {
        tally.add("approximate_catalog_movement", 1);
    }
    // The divergence: two requests co-batched in both orders. Capacity-limited routing
    // changes the logits with the order, dropless routing does not.
    let batch = [prompt(), second_prompt()];
    for (moe, label) in [
        (variant_moe(&variant), "variant"),
        (world.moe.clone(), "dropless"),
    ] {
        let stack = pins::reference_stack()?
            .with_moe(moe)
            .map_err(fail("the co-batch stack"))?;
        let (forward, dropped) = pins::cobatched_last_logits(&stack, &batch, &[0, 1])?;
        let (reverse, _) = pins::cobatched_last_logits(&stack, &batch, &[1, 0])?;
        let differs = forward.iter().zip(&reverse).any(|(a, b)| {
            a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits())
        });
        if label == "variant" {
            tally.add("nondeterministic_divergence_observed", i64::from(differs));
            tally.add("variant_dropped_selections", dropped as i64);
        } else {
            tally.add("batch_invariance_mismatches", i64::from(differs));
        }
    }
    Ok(())
}

fn variant_moe(binding: &StackBinding) -> Qwen2Moe {
    binding.moe.as_ref().expect("a MoE binding").moe.clone()
}

/// The quarantine arm: Exact KV is populated, the expert is fenced by the real authority path,
/// a matching request must not reuse it, and a readmitted expert never revives the entries
/// recorded under an earlier fence epoch.
pub fn quarantine_arm(tally: &mut Tally, world: &World) -> Res<()> {
    let backend = lab::backend()?;
    let mut authority = Authority::new(&[])?;
    birth_all(&mut authority)?;
    let state = experts::state_of(&authority.runtime, J, world.index);
    let Some((ExpertState::Serving, 1, _)) = state else {
        return Err(format!("the expert is not freshly serving: {state:?}"));
    };
    // Entries record the epoch the expert's state record stood at.
    let recorded = world.binding(&authority, false);
    let masked = world.masked(&authority);
    let (p, p72) = (prompt(), extended(1));
    let mut pre = begin(&backend, &raw("quar-pre", &p), &recorded, &authority)
        .map_err(fail("the pre-fence session"))?;
    let cold = lab::cold(&backend, &raw("quar-pre", &p), &recorded)?;
    let out = complete(&mut pre, &authority)?;
    compare(tally, &out, &cold);
    let request = raw("quar-match", &p72);
    let control = begin(&backend, &request, &recorded, &authority).map_err(fail("the control"))?;
    let control_hit = control.hit_tokens();
    drop(control);
    if control_hit != PROMPT {
        return Err(format!("the quarantine control hit {control_hit} tokens"));
    }
    tally.set("quarantine_control_hit_tokens", control_hit as i64);

    // The fence: a committed AuthorityQuarantine; the catalog's tombstoning follows it.
    let mark = authority.mark();
    experts::fence(
        &mut authority.runtime,
        J,
        world.index,
        ExpertFenceCause::AuthorityQuarantine,
        None,
    )?;
    lab::apply_events(&backend, &authority.events_since(mark))?;
    if !matches!(
        experts::state_of(&authority.runtime, J, world.index),
        Some((ExpertState::Fenced, 1, _))
    ) {
        return Err("the fence did not leave the expert Fenced at its generation".to_string());
    }
    tally.add("expert_quarantine_kv_cases", 1);
    // A matching request on the binding that cites the fenced expert is refused.
    let refused = begin(&backend, &request, &recorded, &authority);
    if !refused_not_live(&refused) {
        tally.add("kv_reuse_after_expert_fence", 1);
    }
    drop(refused);
    // The router with the expert masked reuses nothing above j from before the fence.
    let mut masked_session =
        begin(&backend, &request, &masked, &authority).map_err(fail("the masked request"))?;
    tally.add(
        "kv_reuse_after_expert_fence",
        any_pages_shared(&masked_session, &pre, (J + 1)..LAYERS) as i64,
    );
    let out = complete(&mut masked_session, &authority)?;
    compare(tally, &out, &lab::cold(&backend, &request, &masked)?);
    drop(masked_session);

    // The readmission: the expert serves again at the next epoch. The entries recorded under
    // the earlier epoch stay dead, and a binding at the new epoch computes its own pages.
    experts::readmit(
        &mut authority.runtime,
        J,
        world.index,
        ExpertFenceCause::AuthorityQuarantine,
    )?;
    let Some((ExpertState::Serving, 1, new_version)) =
        experts::state_of(&authority.runtime, J, world.index)
    else {
        return Err("the readmission did not return the expert to Serving".to_string());
    };
    let stale = begin(&backend, &request, &recorded, &authority);
    if !refused_not_live(&stale) {
        tally.add("kv_reuse_after_expert_fence", 1);
    }
    drop(stale);
    let fresh_binding = world.binding(&authority, false);
    let mut fresh = begin(&backend, &request, &fresh_binding, &authority)
        .map_err(fail("the readmitted binding"))?;
    tally.add(
        "kv_reuse_after_expert_fence",
        any_pages_shared(&fresh, &pre, (J + 1)..LAYERS) as i64,
    );
    let out = complete(&mut fresh, &authority)?;
    compare(tally, &out, &lab::cold(&backend, &request, &fresh_binding)?);
    tally.set("readmitted_state_version", new_version as i64);

    // Any cause: an expert fenced because an input it was trained on was revoked is not live
    // either, whichever epoch its dependency recorded.
    lab::commit_revocation(&mut authority.runtime, "input:layered-kv", 1)?;
    experts::fence(
        &mut authority.runtime,
        LAYERS - 1,
        EXPERTS - 1,
        ExpertFenceCause::InputRevoked,
        Some(("input:layered-kv", 1)),
    )?;
    let dependency = KvDependency::expert(
        &ProjectId::from(lab::PROJECT),
        &experts::pod_of(LAYERS - 1, EXPERTS - 1),
        Generation(1),
        [0; 32],
    );
    use ptr_runtime::KvLivenessOracle as _;
    tally.add(
        "fence_cause_input_revoked_live_after_fence",
        i64::from(authority.is_live(&dependency)),
    );
    tally.add("fence_cause_cases", 1);
    lab::note_scope(tally, &backend)?;
    Ok(())
}

// ---- the matrix ---------------------------------------------------------------------------

/// The matrix's binding-carried dependency: expert `(j, i)` of the MoE binding.
pub struct R010Rig {
    pub world: World,
}

impl Rig for R010Rig {
    fn dependency_kind(&self) -> &'static str {
        "expert_generation"
    }

    fn binding(&self, authority: &Authority) -> StackBinding {
        self.world.binding(authority, true)
    }

    fn resolved_binding(&self, ty: Invalidation, authority: &Authority) -> StackBinding {
        match ty {
            Invalidation::Revocation | Invalidation::Fence => self.world.masked(authority),
            Invalidation::Supersession => {
                let backend = lab::backend().expect("a backend");
                self.world.successor(&backend, authority)
            }
        }
    }

    fn resolution_holds(&self, record: &BindingRecord, ty: Invalidation) -> bool {
        let one = vec![(J, self.world.index)];
        match ty {
            Invalidation::Revocation | Invalidation::Fence => {
                record.masked_experts == one && record.experts_masked_by_resolution == one
            }
            Invalidation::Supersession => {
                record.experts_superseded == vec![(J, self.world.index, 1, 2)]
                    && record.experts_masked_by_resolution.is_empty()
            }
        }
    }

    fn identity(&self) -> String {
        experts::identity_of(J, self.world.index)
    }

    fn capsules(&self) -> Vec<String> {
        Vec::new()
    }

    fn commit_live(&self, authority: &mut Authority) -> Res<()> {
        birth_all(authority)
    }

    fn commit_event(&self, authority: &mut Authority, ty: Invalidation) -> Res<()> {
        let runtime = &mut authority.runtime;
        match ty {
            Invalidation::Revocation => experts::retire(runtime, J, self.world.index),
            Invalidation::Supersession => {
                experts::supersede(runtime, J, self.world.index).map(|_| ())
            }
            Invalidation::Fence => experts::fence(
                runtime,
                J,
                self.world.index,
                ExpertFenceCause::AuthorityQuarantine,
                None,
            ),
        }
    }

    fn expected_dropped(&self, layers: usize, chunks: usize) -> usize {
        (layers - 1 - J) * chunks
    }

    fn superseded_generation(&self) -> u64 {
        2
    }

    fn procedure_mismatches(&self, observed: &Observed) -> Vec<String> {
        let mut mismatches = Vec::new();
        let mut check = |name: &str, ok: bool| {
            if !ok {
                mismatches.push(name.to_string());
            }
        };
        check(
            "test_model_layers",
            observed.layers as u64 == params::TEST_MODEL_LAYERS,
        );
        check(
            "hidden_size",
            observed.hidden_size as u64 == params::HIDDEN_SIZE,
        );
        check("num_heads", observed.num_heads as u64 == params::NUM_HEADS);
        check(
            "num_kv_heads",
            observed.num_kv_heads as u64 == params::NUM_KV_HEADS,
        );
        check(
            "vocab_size",
            observed.vocab_size as u64 == params::VOCAB_SIZE,
        );
        check("weight_seed", observed.weight_seed == params::WEIGHT_SEED);
        check(
            "page_tokens",
            observed.page_tokens as u64 == params::PAGE_TOKENS,
        );
        check(
            "intermediate_size",
            observed.intermediate_size == pins::INTERMEDIATE_SIZE,
        );
        check(
            "max_new_tokens",
            observed.max_new_tokens as u64 == params::MAX_NEW_TOKENS,
        );
        check("adapter", observed.adapter.is_none());
        check(
            "moe_topology",
            observed
                .moe
                .as_ref()
                .is_some_and(|(layers, experts, top_k, mode)| {
                    layers.len() == LAYERS
                        && *experts as u64 == params::MOE_TOPOLOGY_NUM_EXPERTS
                        && *top_k as u64 == params::MOE_TOPOLOGY_TOP_K
                        && mode == "Dropless"
                }),
        );
        mismatches
    }

    fn procedure_json(&self, observed: &Observed) -> Json {
        Json::obj([
            ("test_model_layers", Json::Int(observed.layers as i64)),
            ("revoked_expert_layer_j", Json::Int(J as i64)),
            ("revoked_expert_index", Json::Int(self.world.index as i64)),
            ("hidden_size", Json::Int(observed.hidden_size as i64)),
            ("num_heads", Json::Int(observed.num_heads as i64)),
            ("num_kv_heads", Json::Int(observed.num_kv_heads as i64)),
            ("vocab_size", Json::Int(observed.vocab_size as i64)),
            ("weight_seed", Json::Int(observed.weight_seed as i64)),
            ("page_tokens", Json::Int(observed.page_tokens as i64)),
            (
                "moe_topology",
                observed.moe.as_ref().map_or(
                    Json::str("none"),
                    |(layers, experts, top_k, mode)| {
                        Json::obj([
                            ("moe_layers", Json::Int(layers.len() as i64)),
                            ("num_experts", Json::Int(*experts as i64)),
                            ("top_k", Json::Int(*top_k as i64)),
                            ("mode", Json::str(mode.clone())),
                        ])
                    },
                ),
            ),
        ])
    }

    fn note_cell(&self, tally: &mut Tally, cell: &CellRecord) {
        let Some(inflight) = cell.inflight else {
            return;
        };
        let ty = cell.ty.name();
        if cell.is_dependency {
            tally.add(&format!("inflight_expert_revocation_race_cases.{ty}"), 1);
            tally.add(&format!("inflight_race_cases_by_cell.{ty}_expert"), 1);
            tally.add(
                &format!("inflight_outputs_after_invalidation.{ty}_expert"),
                inflight.escaped,
            );
            tally.add("outputs_after_expert_revocation", inflight.escaped);
            tally.add("expert_inflight_cancelled", i64::from(inflight.cancelled));
            tally.add(
                "expert_inflight_restarted_fresh",
                i64::from(inflight.restarted_fresh == Some(true)),
            );
            tally.add(
                "expert_restart_output_mismatches",
                i64::from(inflight.restarted_fresh == Some(false)),
            );
        } else if cell.kind != "base_model" {
            if cell.is_semantic_input {
                tally.add(&format!("inflight_input_race_cases.{ty}"), 1);
                tally.add(
                    &format!("inflight_race_cases_by_cell.{ty}_semantic_input"),
                    1,
                );
                tally.add(
                    &format!("inflight_outputs_after_invalidation.{ty}_semantic_input"),
                    inflight.escaped,
                );
            }
            tally.add("outputs_after_input_invalidation", inflight.escaped);
        } else {
            tally.add("base_model_inflight_cases", 1);
            tally.add("base_model_inflight_escaped_outputs", inflight.escaped);
        }
    }
}

/// Run every cell of the matrix for one seed (44 at the pinned scale).
pub fn run_matrix(tally: &mut Tally, world: World, seed: u64, scale: Scale) -> Res<Vec<CellInfo>> {
    let rig = R010Rig { world };
    let mut scratch = Ids::default();
    let mut infos = Vec::new();
    let kinds: Vec<Kind> = std::iter::once(Kind::BaseModel)
        .chain(std::iter::once(Kind::Dependency))
        .chain(SEGMENT_KINDS.iter().map(Kind::Segment))
        .collect();
    for kind in kinds {
        if scale == Scale::Reduced && !matches!(kind, Kind::Dependency) {
            continue;
        }
        for path in Path::ALL {
            let types: &[Invalidation] = if matches!(kind, Kind::Dependency) {
                &Invalidation::EXPERT
            } else {
                &Invalidation::ALL
            };
            for ty in types {
                matrix::run_cell(tally, &mut scratch, &mut infos, &rig, seed, kind, *ty, path)?;
            }
        }
    }
    Ok(infos)
}

/// The plain-restart arm, reported separately: an empty catalog after a restart serves nothing
/// and computes the right answer.
pub fn empty_restart(tally: &mut Tally, world: &World) -> Res<()> {
    let first = lab::backend()?;
    let mut authority = Authority::new(&[])?;
    birth_all(&mut authority)?;
    let binding = world.binding(&authority, false);
    let request = raw("restart-probe", &prompt());
    let warmed = lab::run_completed(&first, &request, &binding, &authority)?;
    drop(first);
    let restarted = lab::backend()?;
    let outcome = lab::run_completed(&restarted, &request, &binding, &authority)?;
    let cold = lab::cold(&restarted, &request, &binding)?;
    let equal = compare(tally, &outcome, &cold);
    let stats = restarted.catalog_stats().map_err(fail("catalog stats"))?;
    tally.set_json(
        "empty_restart_arm",
        Json::obj([
            ("role", Json::str(params::EMPTY_RESTART_ARM_ROLE)),
            ("entries_before_lookup", Json::Int(0)),
            (
                "pre_restart_inserted_chunks",
                Json::Int(warmed.inserted_chunks as i64),
            ),
            ("hit_tokens", Json::Int(outcome.hit_tokens as i64)),
            ("catalog_hits", Json::Int(stats.hits as i64)),
            ("equals_cold", Json::Bool(equal)),
            ("counts_toward_restore_coverage", Json::Bool(false)),
        ]),
    );
    Ok(())
}

#[allow(dead_code)]
fn unused(_: &PodId) {}
