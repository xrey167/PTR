//! The two negative controls: the harness' own oracle must be able to see what it claims to.
//!
//! - **bypass.** For each effect kind the executor's log is fed a stale action directly, as if
//!   the hard boundary did not exist (the harness cannot hand the real executor a dispatch, so
//!   the call goes to the same logging function the real executor uses): the mirror must flag
//!   it stale by revision, flag another by generation, and flag a fresh one fresh. A log that
//!   does not flag them could not count a stale execution, and S002 would pass vacuously.
//! - **ledger scan.** A stale `EffectAttempted` record is committed directly and the scan of the
//!   ledger must count it; the scan of a clean ledger must count none.
//!
//! A control that does not trip is `Err`: exit 2, the harness cannot tell a broken boundary
//! from a sound one.

use super::metrics::Counters;
use super::oracle::{self, Staleness};
use super::params;
use super::world::{self, World};
use crate::experiments::layered_kv::emit::Json;
use ptr_ledger::LedgerEvent;
use ptr_types::{CapabilityId, Generation, ProjectId, Revision, VerificationLevel};

/// Both controls' JSON, `(bypass, ledger_scan)`, and the trips added to `counters`.
pub fn run(counters: &mut Counters) -> Result<(Json, Json), String> {
    let mut flagged = Vec::new();
    for effect in params::EFFECTS {
        let world = World::new(false, effect)?;
        let now = world.with_mirror(|m| m.revision());
        let verdicts = world.with_mirror(|mirror| {
            (
                mirror.record_call(&world::action_at(effect, Revision(now - 1))),
                mirror.record_call(&{
                    let mut action = world::action_at(effect, Revision(now));
                    action.generation = Generation(2);
                    action
                }),
                mirror.record_call(&world::action_at(effect, Revision(now))),
            )
        });
        if verdicts != (Staleness::Revision, Staleness::Generation, Staleness::Fresh) {
            return Err(format!(
                "the bypass control does not trip for {}: the executor log judged {verdicts:?}",
                params::effect_name(effect)
            ));
        }
        let stale = world.with_mirror(|m| m.stale_calls());
        if stale != 2 {
            return Err(format!(
                "the bypass control counted {stale} stale calls for {}, not 2",
                params::effect_name(effect)
            ));
        }
        counters.control_trips += 1;
        flagged.push(params::effect_name(effect));
    }

    let mut world = World::new(false, ptr_types::Effect::Mutation)?;
    let clean = oracle::scan(world.runtime.committed_events());
    let current = world.runtime.revision();
    world
        .runtime
        .commit(LedgerEvent::EffectAttempted {
            key: None,
            project: ProjectId::from(world::PROJECT),
            principal: "s002-control".into(),
            target: world::TARGET.into(),
            operation: world::OPERATION.into(),
            capability: CapabilityId::from(world::CAPABILITY),
            effect: ptr_types::Effect::Mutation,
            generation: Generation(1),
            revision: Revision(current.0 - 1),
            verification: VerificationLevel::Deterministic,
            action_digest: [7; 32],
        })
        .map_err(|error| {
            format!("the ledger-scan control could not plant its record: {error:?}")
        })?;
    let planted = oracle::scan(world.runtime.committed_events());
    if clean.attempts != 0
        || clean.stale_attempts != 0
        || planted.attempts != 1
        || planted.stale_attempts != 1
    {
        return Err(format!(
            "the ledger-scan control does not trip: clean {clean:?}, planted {planted:?}"
        ));
    }
    counters.control_trips += 1;

    let bypass = Json::obj([
        ("required_rule", Json::str("stale_actions_executed <= 0")),
        ("required_rule_tripped", Json::Bool(true)),
        ("effects", Json::strs(&flagged)),
    ]);
    let ledger_scan = Json::obj([
        (
            "required_rule",
            Json::str("stale_effect_attempts_committed <= 0"),
        ),
        ("required_rule_tripped", Json::Bool(true)),
        ("clean_stale_attempts", Json::Int(clean.stale_attempts)),
        ("planted_stale_attempts", Json::Int(planted.stale_attempts)),
    ]);
    Ok((bypass, ledger_scan))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_controls_trip_and_count_one_trip_per_effect_and_one_for_the_scan() {
        let mut counters = Counters::default();
        let (bypass, scan) = run(&mut counters).expect("controls run");
        assert_eq!(counters.control_trips, 5);
        assert!(matches!(
            crate::experiments::layered_kv::emit::resolve(&bypass, "required_rule_tripped"),
            Some(Json::Bool(true))
        ));
        assert!(matches!(
            crate::experiments::layered_kv::emit::resolve(&scan, "planted_stale_attempts"),
            Some(Json::Int(1))
        ));
    }
}
