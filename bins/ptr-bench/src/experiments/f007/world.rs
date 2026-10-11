//! The worlds of F007: which Pod is how good at which finding code.
//!
//! EVERY NUMBER HERE IS THE ASSUMPTION UNDER TEST, not a measurement. The primary world
//! (`hetero`) declares that a Pod registered for a repair role clears the codes of that role much
//! more often than a generalist and that an off-role Pod almost never does; whether a verifier-
//! routed loop helps then depends on that assumption, and the experiment says how large the help
//! is if the assumption holds. `homog` declares every Pod equally good at every code: finding
//! routing has nothing to exploit there and must tie. `mild` shrinks the specialisation and `ctx`
//! additionally lets a Pod read the workspace broadcast; both are descriptive.

use std::collections::BTreeMap;
use std::sync::Arc;

use ptr_ledger::integrity::sha256;
use ptr_pods::DynPod;
use ptr_router::{FindingCode, PodEligibility, PodRef};
use ptr_runtime::vcw::experts::{Guarded, SimulatedExperts, SimulationConfig};
use ptr_runtime::vcw::{ProposeCall, RepairCall, RolePods, Roster};
use ptr_types::{PodId, ProjectId, TypeId};
use ptr_verifier::{repair_role_for, RepairRole};

use super::params;
use super::task::{Kind, Served};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WorldId {
    Hetero,
    Homog,
    Mild,
    Ctx,
}

impl WorldId {
    pub const ALL: [WorldId; 4] = [Self::Hetero, Self::Homog, Self::Mild, Self::Ctx];

    pub fn name(self) -> &'static str {
        match self {
            Self::Hetero => "hetero",
            Self::Homog => "homog",
            Self::Mild => "mild",
            Self::Ctx => "ctx",
        }
    }

    /// The worlds whose results can decide a verdict.
    pub fn gated(self) -> bool {
        matches!(self, Self::Hetero | Self::Homog)
    }

    /// Whether one code (`code.timeout`) has no Pod that can repair it at the start, so that
    /// L3 has something to do. The null world has no such gap: an L3 benefit there would be a
    /// capability the world adds, not specialisation.
    pub fn gap(self) -> bool {
        !matches!(self, Self::Homog)
    }

    /// Whether the proposers differ by kind. The null world has identical proposers.
    pub fn proposers_differ(self) -> bool {
        !matches!(self, Self::Homog)
    }

    /// The share (per mille) of failed repairs a Pod fixes anyway when it can read the
    /// workspace broadcast.
    pub fn second_chance(self) -> u64 {
        if matches!(self, Self::Ctx) {
            params::BROADCAST_SECOND_CHANCE_PERMILLE
        } else {
            0
        }
    }

    fn skill_table(self) -> (u64, u64, u64, u64) {
        let row = params::SKILL_PERMILLE
            .split(';')
            .find_map(|group| {
                group
                    .split_once('=')
                    .filter(|(name, _)| *name == self.name())
                    .map(|(_, rest)| rest)
            })
            .expect("every world has a skill row");
        let values: Vec<u64> = row
            .split(',')
            .map(|v| v.parse().expect("integer"))
            .collect();
        (values[0], values[1], values[2], values[3])
    }
}

/// What a repair Pod is registered for.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Role {
    Format,
    Value,
    Schema,
    Code,
    Generalist,
}

impl Role {
    fn from_repair(role: RepairRole) -> Option<Self> {
        match role {
            RepairRole::Format => Some(Self::Format),
            RepairRole::Value => Some(Self::Value),
            RepairRole::Schema => Some(Self::Schema),
            RepairRole::Code => Some(Self::Code),
            RepairRole::Resource | RepairRole::Fact => None,
        }
    }

    /// The role a deliberately wrong map sends a code of this role to.
    fn shifted(self) -> Self {
        match self {
            Self::Format => Self::Value,
            Self::Value => Self::Schema,
            Self::Schema => Self::Code,
            Self::Code => Self::Format,
            Self::Generalist => Self::Generalist,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RepairPod {
    pub id: &'static str,
    pub role: Role,
}

pub const REPAIRERS: [RepairPod; 9] = [
    RepairPod {
        id: "fmt_1",
        role: Role::Format,
    },
    RepairPod {
        id: "fmt_2",
        role: Role::Format,
    },
    RepairPod {
        id: "val_1",
        role: Role::Value,
    },
    RepairPod {
        id: "val_2",
        role: Role::Value,
    },
    RepairPod {
        id: "sch_1",
        role: Role::Schema,
    },
    RepairPod {
        id: "sch_2",
        role: Role::Schema,
    },
    RepairPod {
        id: "cod_1",
        role: Role::Code,
    },
    RepairPod {
        id: "cod_2",
        role: Role::Code,
    },
    RepairPod {
        id: "gen_r",
        role: Role::Generalist,
    },
];

pub const PROPOSERS: [&str; 3] = ["gen_1", "gen_2", "gen_3"];

/// The code nobody can repair at the start of a world with a gap.
pub const GAP_CODE: &str = "code.timeout";

/// The role of a code in `world`: the real verifier's table, except that without a gap the
/// timeout code is an ordinary code-repair code. `None` is the gap.
pub fn role_of(world: WorldId, code: &str) -> Option<Role> {
    let role = repair_role_for(code).expect("a plantable code is in the table");
    if role == RepairRole::Resource {
        return if world.gap() { None } else { Some(Role::Code) };
    }
    Role::from_repair(role)
}

/// Probability (per mille) that `pod` clears `code` in one repair call.
pub fn skill_permille(world: WorldId, pod: &RepairPod, code: &str) -> u64 {
    let Some(role) = role_of(world, code) else {
        return params::GAP_SKILL_PERMILLE;
    };
    let (on, span, generalist, off) = world.skill_table();
    match pod.role {
        Role::Generalist => generalist,
        own if own == role => {
            if span == 0 {
                return on;
            }
            let hash =
                sha256(format!("f007-skill|{}|{}|{}", world.name(), pod.id, code).as_bytes());
            let steps = span / 10;
            let pick = u64::from(hash[0]) % (2 * steps + 1);
            on + pick * 10 - steps * 10
        }
        _ => off,
    }
}

/// Which Pods the finding registry names for each code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistryMode {
    /// Each code to the Pods of its role and to the generalist.
    True,
    /// Each code to the Pods of the wrong role (the role shifted by one) and to the generalist:
    /// the sensitivity control.
    Shuffled,
    /// Every repair Pod for every code: there is no finding map.
    Generic,
    /// No repair Pod: a failed candidate can only be regenerated (best-of-N).
    NoRepair,
    /// Each code to the one Pod that is best at it (the world's table, read with hindsight), and
    /// only the proposer that is good at the kind: an upper reference, not an arm anyone can build.
    Oracle,
}

/// The repair Pod with the highest skill at `code` in `world` (ties: the first).
pub fn best_pod(world: WorldId, code: &str) -> RepairPod {
    let mut best = REPAIRERS[0];
    for pod in REPAIRERS {
        if skill_permille(world, &pod, code) > skill_permille(world, &best, code) {
            best = pod;
        }
    }
    best
}

/// `(pod, codes)` for the loop's roster, in a fixed order.
pub fn repairer_roster(
    world: WorldId,
    kind: Kind,
    mode: RegistryMode,
) -> Vec<(PodId, Vec<String>)> {
    let mut roster = Vec::new();
    if mode == RegistryMode::NoRepair {
        return roster;
    }
    for pod in REPAIRERS {
        let codes: Vec<String> = kind
            .codes()
            .iter()
            .filter(|code| match mode {
                RegistryMode::Generic => true,
                RegistryMode::NoRepair => false,
                RegistryMode::Oracle => {
                    role_of(world, code).is_some() && best_pod(world, code).id == pod.id
                }
                RegistryMode::True | RegistryMode::Shuffled => {
                    let Some(role) = role_of(world, code) else {
                        return false;
                    };
                    let target = if mode == RegistryMode::Shuffled {
                        role.shifted()
                    } else {
                        role
                    };
                    pod.role == Role::Generalist || pod.role == target
                }
            })
            .map(|code| (*code).to_string())
            .collect();
        if !codes.is_empty() {
            roster.push((PodId::from(pod.id), codes));
        }
    }
    roster
}

pub fn loop_roster(world: WorldId, kind: Kind, mode: RegistryMode) -> Roster {
    Roster {
        proposers: PROPOSERS
            .iter()
            .filter(|id| mode != RegistryMode::Oracle || **id == PROPOSERS[kind.index()])
            .map(|id| PodId::from(*id))
            .collect(),
        repairers: repairer_roster(world, kind, mode)
            .into_iter()
            .map(|(pod, codes)| {
                (
                    pod,
                    codes.iter().map(|c| FindingCode::new(c.as_str())).collect(),
                )
            })
            .collect(),
    }
}

/// Proposal defect rate (per mille) of `proposer` for `code`.
pub fn error_permille(world: WorldId, proposer: &str, kind: Kind, code: &str) -> u64 {
    let table = params::parse_table(params::ERROR_PERMILLE);
    let base: u64 = table
        .iter()
        .find(|(name, _)| *name == code)
        .map(|(_, rate)| rate.parse().expect("integer"))
        .expect("every code has a rate");
    if !world.proposers_differ() {
        return base;
    }
    let good = PROPOSERS[kind.index()] == proposer;
    let multiplier = if good {
        params::PROPOSER_GOOD_MULTIPLIER_PERMILLE
    } else {
        params::PROPOSER_OTHER_MULTIPLIER_PERMILLE
    };
    (base * multiplier / 1000).min(950)
}

/// The simulated Pods of one lane: one adapter per proposer (their defect rates differ) and one
/// for every repairer.
pub struct PodBank {
    pub props: BTreeMap<PodId, SimulatedExperts<Served>>,
    pub rep: SimulatedExperts<Served>,
    /// Whether the workspace broadcast reaches the Pods. Withheld, a Pod sees `None`.
    pub broadcast: bool,
    pub second_chance: u64,
    /// Repair calls in which a Pod that reads the broadcast saw a refusal naming its code.
    pub reads: u64,
    /// The proposer of the shared first draft: the kind's specialised one, the same in every arm.
    first_draft: PodId,
    /// Whether round 0 is the shared first draft (evaluation) rather than routed (training).
    pub shared_first_draft: bool,
    seed: u64,
}

impl PodBank {
    pub fn new(world: WorldId, kind: Kind, seed: u64, broadcast: bool) -> Self {
        let mut props = BTreeMap::new();
        for proposer in PROPOSERS {
            let mut config = SimulationConfig::new(seed);
            for code in kind.codes() {
                let rate = error_permille(world, proposer, kind, code);
                config = config.with_error_rate(code, rate as f64 / 1000.0);
            }
            let mut experts = SimulatedExperts::new(config, "task").expect("valid rates");
            experts
                .add_proposer(PodId::from(proposer))
                .expect("fresh Pod");
            props.insert(PodId::from(proposer), experts);
        }
        let mut config = SimulationConfig::new(seed);
        config.decline_on_failure = false;
        for pod in REPAIRERS {
            for code in kind.codes() {
                config = config.with_pod_repair_success(
                    pod.id,
                    code,
                    skill_permille(world, &pod, code) as f64 / 1000.0,
                );
            }
        }
        let mut rep = SimulatedExperts::new(config, "task").expect("valid rates");
        for pod in REPAIRERS {
            rep.add_repairer(PodId::from(pod.id), kind.codes())
                .expect("fresh Pod");
        }
        Self {
            props,
            rep,
            broadcast,
            second_chance: world.second_chance(),
            reads: 0,
            first_draft: PodId::from(PROPOSERS[kind.index()]),
            shared_first_draft: true,
            seed,
        }
    }

    /// Every Pod of the bank, as registry entries for payload type `kind`.
    pub fn registry_entries(&self, project: &ProjectId, kind: Kind) -> Vec<Arc<dyn DynPod>> {
        let payload = TypeId::from(kind.answer_type());
        let mut entries = Vec::new();
        for experts in self.props.values() {
            entries.extend(experts.pods(project, &payload));
        }
        entries.extend(self.rep.pods(project, &payload));
        entries
    }

    pub fn base_pods() -> Vec<PodId> {
        PROPOSERS
            .iter()
            .map(|id| PodId::from(*id))
            .chain(REPAIRERS.iter().map(|pod| PodId::from(pod.id)))
            .collect()
    }
}

/// The loop's `RolePods` over a [`PodBank`]: routes each call to the adapter of the Pod and
/// withholds or delivers the broadcast. Every call still goes through the adapters' call-time
/// eligibility guard.
pub struct Dispatch<'a> {
    pub bank: &'a mut PodBank,
    pub standing: &'a dyn PodEligibility,
}

impl RolePods<Served> for Dispatch<'_> {
    fn propose(&mut self, call: ProposeCall<'_>) -> Option<Served> {
        // The round-0 draft is shared by every arm: its proposer is fixed by
        // the kind and its defects are keyed by (seed, request id, Pod, code) inside the adapter, so
        // neither the arm's scorer nor the router's nonce sequence can reach it. Later proposals
        // (a fallback asked for a fresh one) go to the Pod the router chose. The training streams route
        // round 0 freely (`shared_first_draft` false), so the learned parts are fitted on traffic
        // in which the proposer choice shows.
        let first_draft = PodRef::new(self.bank.first_draft.clone(), call.pod.generation);
        let mut call: ProposeCall<'_> = call;
        if !self.bank.broadcast {
            call.snapshot = None;
        }
        if call.round == 0 && self.bank.shared_first_draft {
            call.pod = &first_draft;
        }
        let experts = self.bank.props.get_mut(&call.pod.pod)?;
        Guarded::new(experts, self.standing).propose(call)
    }

    fn repair(&mut self, mut call: RepairCall<'_, Served>) -> Option<Served> {
        let informed = self.bank.broadcast
            && self.bank.second_chance > 0
            && call.snapshot.is_some_and(|snapshot| {
                snapshot.refusals().iter().any(|refusal| {
                    refusal
                        .findings()
                        .iter()
                        .any(|finding| finding.code() == call.code.as_str())
                })
            });
        if informed {
            self.bank.reads += 1;
        }
        if !self.bank.broadcast {
            call.snapshot = None;
        }
        let pod = call.pod.pod.clone();
        let code = call.code.as_str().to_owned();
        let request = call.request_id.0.clone();
        let round = call.round;
        let mut produced = Guarded::new(&mut self.bank.rep, self.standing).repair(call);
        if informed {
            if let Some(candidate) = &produced {
                use ptr_runtime::vcw::experts::DefectPayload;
                if candidate.defects().contains(&code) {
                    let key = format!(
                        "f007-ctx|{}|{request}|{}|{code}|{round}",
                        self.bank.seed, pod.0
                    );
                    let hash = sha256(key.as_bytes());
                    let draw = u64::from_le_bytes(hash[..8].try_into().expect("8 bytes")) % 1000;
                    if draw < self.bank.second_chance {
                        produced = Some(candidate.without_defect(&code));
                    }
                }
            }
        }
        produced
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hetero_world_makes_each_role_pod_good_on_its_codes_and_the_gap_unrepairable() {
        let world = WorldId::Hetero;
        for pod in REPAIRERS {
            for kind in super::super::task::KINDS {
                for code in kind.codes() {
                    let skill = skill_permille(world, &pod, code);
                    match (role_of(world, code), pod.role) {
                        (None, _) => assert_eq!(skill, 0, "{code}"),
                        (Some(_), Role::Generalist) => assert_eq!(skill, 600),
                        (Some(role), own) if role == own => {
                            assert!((720..=880).contains(&skill), "{} {code} {skill}", pod.id);
                        }
                        (Some(_), _) => assert_eq!(skill, 300),
                    }
                }
            }
        }
    }

    #[test]
    fn the_homogeneous_world_has_one_skill_everywhere_and_no_gap() {
        let world = WorldId::Homog;
        for pod in REPAIRERS {
            for kind in super::super::task::KINDS {
                for code in kind.codes() {
                    assert_eq!(skill_permille(world, &pod, code), 600, "{} {code}", pod.id);
                }
            }
        }
        assert!(!world.gap() && !world.proposers_differ());
        for proposer in PROPOSERS {
            assert_eq!(
                error_permille(world, proposer, Kind::Json, "json.type"),
                error_permille(world, "gen_1", Kind::Json, "json.type")
            );
        }
    }

    #[test]
    fn the_true_map_names_role_pods_and_the_shuffled_map_names_none_of_them() {
        for kind in super::super::task::KINDS {
            let truth = repairer_roster(WorldId::Hetero, kind, RegistryMode::True);
            let shuffled = repairer_roster(WorldId::Hetero, kind, RegistryMode::Shuffled);
            for code in kind.codes() {
                let named = |roster: &Vec<(PodId, Vec<String>)>| -> Vec<String> {
                    roster
                        .iter()
                        .filter(|(_, codes)| codes.iter().any(|c| c == code))
                        .map(|(pod, _)| pod.0.clone())
                        .collect()
                };
                let (t, s) = (named(&truth), named(&shuffled));
                if *code == GAP_CODE {
                    assert!(t.is_empty() && s.is_empty(), "the gap is unregistered");
                    continue;
                }
                assert!(
                    t.contains(&"gen_r".to_string()) && t.len() == 3,
                    "{code}: {t:?}"
                );
                assert!(
                    s.contains(&"gen_r".to_string()) && s.len() == 3,
                    "{code}: {s:?}"
                );
                let specialists_t: Vec<_> = t.iter().filter(|p| *p != "gen_r").collect();
                let specialists_s: Vec<_> = s.iter().filter(|p| *p != "gen_r").collect();
                assert!(
                    specialists_t.iter().all(|p| !specialists_s.contains(p)),
                    "{code}: {t:?} vs {s:?}"
                );
            }
        }
        assert!(repairer_roster(WorldId::Hetero, Kind::Json, RegistryMode::NoRepair).is_empty());
        let generic = repairer_roster(WorldId::Hetero, Kind::Json, RegistryMode::Generic);
        assert_eq!(generic.len(), 9);
        assert!(generic.iter().all(|(_, codes)| codes.len() == 4));
    }

    #[test]
    fn a_bank_holds_every_pod_and_validates_its_roster() {
        let bank = PodBank::new(WorldId::Hetero, Kind::Arithmetic, 1, true);
        assert_eq!(bank.props.len(), 3);
        bank.rep.validate_roster().expect("a consistent roster");
        let entries = bank.registry_entries(&ProjectId::from("p"), Kind::Arithmetic);
        assert_eq!(entries.len(), 12);
    }

    #[test]
    fn proposers_are_better_at_their_own_kind_in_the_differing_worlds() {
        let own = error_permille(WorldId::Hetero, "gen_2", Kind::Json, "json.type");
        let other = error_permille(WorldId::Hetero, "gen_1", Kind::Json, "json.type");
        assert!(own < other, "{own} {other}");
    }
}
