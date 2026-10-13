//! The typed dependency block of a version 3 `ExpertCommitRecord`: construction, canonical
//! encoding, strict decoding and compatibility with versions 1 and 2.

use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertInputError,
    ExpertInputRef, ExpertInputs, ExpertLevel, ExpertPodSpec, ExpertProposal, ExpertProposalError,
    ExpertState, ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest,
    StructuralOp, MAX_EXPERT_INPUTS_PER_ROLE, MAX_EXPERT_INPUT_IDENTITY_BYTES,
};
use ptr_types::{CommitIndex, Generation, PodId, ProjectId};

const EVIDENCE: [u8; 32] = [0xE1; 32];
const CONFIG: [u8; 32] = [0xC1; 32];

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn base() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new([9; 32]).unwrap(),
        1,
    )
    .unwrap()
}

fn spec(pod: &str) -> ExpertPodSpec {
    ExpertPodSpec::new(
        project(),
        PodId::from(pod),
        ManifestDigest::new([2; 32]).unwrap(),
        ExpertLevel::Segment,
        base(),
        Generation(1),
    )
    .unwrap()
}

fn proposal(op: StructuralOp) -> ExpertProposal {
    ExpertProposal::new(
        project(),
        ProposerRef::new(project(), PodId::from("meta"), Generation(1)).unwrap(),
        CommitIndex(10),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap()
}

fn birth() -> ExpertProposal {
    proposal(StructuralOp::Birth {
        result: spec("new"),
    })
}

fn input(identity: &str, generation: u64, digest: u8) -> ExpertInputRef {
    ExpertInputRef::new(identity, Generation(generation), [digest; 32]).unwrap()
}

fn inputs() -> ExpertInputs {
    ExpertInputs::new(
        vec![input("data:b", 2, 0x22), input("data:a", 1, 0x11)],
        vec![input("eval:x", 3, 0x33)],
    )
    .unwrap()
}

fn v3(proposal: ExpertProposal, inputs: ExpertInputs) -> ExpertCommitRecord {
    ExpertCommitRecord::new_with_inputs(proposal, EVIDENCE, 7, CONFIG, b"report".to_vec(), inputs)
        .unwrap()
}

fn malformed(bytes: &[u8]) -> &'static str {
    match ExpertCommitRecord::decode_canonical(bytes) {
        Err(ExpertProposalError::MalformedRecord(reason)) => reason,
        other => panic!("expected a malformed record, got {other:?}"),
    }
}

#[test]
fn input_refs_validate_identity_generation_and_digest() {
    let long = "x".repeat(MAX_EXPERT_INPUT_IDENTITY_BYTES + 1);
    let at_bound = "x".repeat(MAX_EXPERT_INPUT_IDENTITY_BYTES);
    assert!(ExpertInputRef::new(at_bound, Generation(1), [1; 32]).is_ok());
    for (identity, generation, digest, expected) in [
        ("", 1, [1; 32], ExpertInputError::BlankIdentity),
        (long.as_str(), 1, [1; 32], ExpertInputError::IdentityTooLong),
        (
            "a\nb",
            1,
            [1; 32],
            ExpertInputError::ControlCharacterInIdentity,
        ),
        ("a", 0, [1; 32], ExpertInputError::ZeroGeneration),
        ("a", 1, [0; 32], ExpertInputError::ZeroDigest),
    ] {
        assert_eq!(
            ExpertInputRef::new(identity, Generation(generation), digest),
            Err(expected)
        );
    }
}

#[test]
fn an_input_block_is_canonically_ordered_bounded_and_conflict_free() {
    let block = inputs();
    let training: Vec<_> = block
        .training()
        .iter()
        .map(|i| (i.identity().to_owned(), i.generation().0))
        .collect();
    assert_eq!(
        training,
        vec![("data:a".to_owned(), 1), ("data:b".to_owned(), 2)]
    );
    assert_eq!(block.len(), 3);
    assert!(!block.is_empty());
    assert!(ExpertInputs::default().is_empty());

    // The same identity at two generations is two inputs; the same pair twice is a duplicate.
    assert!(ExpertInputs::new(vec![input("a", 1, 1), input("a", 2, 1)], vec![]).is_ok());
    assert!(matches!(
        ExpertInputs::new(vec![input("a", 1, 1), input("a", 1, 1)], vec![]),
        Err(ExpertInputError::DuplicateInput { .. })
    ));
    // One (identity, generation) with two digests, across the lists, is a conflict; with one
    // digest it is fine.
    assert!(matches!(
        ExpertInputs::new(vec![input("a", 1, 1)], vec![input("a", 1, 2)]),
        Err(ExpertInputError::DigestConflict { .. })
    ));
    assert!(ExpertInputs::new(vec![input("a", 1, 1)], vec![input("a", 1, 1)]).is_ok());

    // The bound is per list, inclusive.
    let many = |n: usize| -> Vec<ExpertInputRef> {
        (0..n).map(|i| input(&format!("in:{i:05}"), 1, 1)).collect()
    };
    assert!(ExpertInputs::new(many(MAX_EXPERT_INPUTS_PER_ROLE), many(1)).is_ok());
    assert_eq!(
        ExpertInputs::new(many(MAX_EXPERT_INPUTS_PER_ROLE + 1), vec![]),
        Err(ExpertInputError::TooManyInputs {
            found: MAX_EXPERT_INPUTS_PER_ROLE + 1
        })
    );
    assert_eq!(
        ExpertInputs::new(vec![], many(MAX_EXPERT_INPUTS_PER_ROLE + 1)),
        Err(ExpertInputError::TooManyInputs {
            found: MAX_EXPERT_INPUTS_PER_ROLE + 1
        })
    );
}

#[test]
fn the_block_digest_covers_every_part() {
    let base_block = inputs().canonical_digest();
    assert_eq!(base_block, inputs().canonical_digest());
    let changed = [
        ExpertInputs::new(
            vec![input("data:b", 2, 0x22), input("data:a", 1, 0x12)],
            vec![input("eval:x", 3, 0x33)],
        ),
        ExpertInputs::new(
            vec![input("data:b", 2, 0x22), input("data:a", 1, 0x11)],
            vec![input("eval:x", 4, 0x33)],
        ),
        // Moving an input from the evaluation list to the training list changes the digest.
        ExpertInputs::new(
            vec![
                input("data:b", 2, 0x22),
                input("data:a", 1, 0x11),
                input("eval:x", 3, 0x33),
            ],
            vec![],
        ),
        Ok(ExpertInputs::default()),
    ];
    for block in changed {
        assert_ne!(block.unwrap().canonical_digest(), base_block);
    }
}

#[test]
fn the_base_model_has_an_input_form() {
    let reference = ExpertInputRef::from_base_model(&base());
    assert_eq!(reference.identity(), "base-model:base-a");
    assert_eq!(reference.generation(), Generation(1));
    assert_eq!(reference.digest(), &[9; 32]);
}

#[test]
fn a_version_3_record_round_trips_and_keeps_the_proposal_ids() {
    let record = v3(birth(), inputs());
    let bytes = record.encode_canonical();
    assert_eq!(&bytes[..8], b"PTREXR03");
    let decoded = ExpertCommitRecord::decode_canonical(&bytes).unwrap();
    assert_eq!(decoded, record);
    assert_eq!(decoded.encode_canonical(), bytes);
    assert_eq!(decoded.inputs(), Some(&inputs()));
    assert_eq!(decoded.evidence_digest(), &EVIDENCE);
    assert_eq!(decoded.gate_epoch(), 7);
    assert!(decoded.evidence().is_some());
    // Inputs are not part of the proposal, so the ids a report binds to do not move.
    assert_eq!(decoded.proposal().proposal_id(), birth().proposal_id());
    // An empty block is a valid version 3 record, distinct from version 2 bytes.
    let empty = v3(birth(), ExpertInputs::default());
    let v2 =
        ExpertCommitRecord::new_with_evidence(birth(), EVIDENCE, 7, CONFIG, b"report".to_vec())
            .unwrap();
    assert_ne!(empty.encode_canonical(), v2.encode_canonical());
    assert_eq!(
        ExpertCommitRecord::decode_canonical(&empty.encode_canonical()).unwrap(),
        empty
    );
}

#[test]
fn versions_1_and_2_decode_exactly_as_before_and_name_no_inputs() {
    let v1 = ExpertCommitRecord::new(birth(), EVIDENCE, 7).unwrap();
    let v2 =
        ExpertCommitRecord::new_with_evidence(birth(), EVIDENCE, 7, CONFIG, b"report".to_vec())
            .unwrap();
    assert_eq!(&v1.encode_canonical()[..8], b"PTREXR01");
    assert_eq!(&v2.encode_canonical()[..8], b"PTREXR02");
    for record in [v1, v2] {
        let decoded = ExpertCommitRecord::decode_canonical(&record.encode_canonical()).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(decoded.inputs(), None);
    }
}

#[test]
fn inputs_need_an_operation_that_produces_an_expert() {
    let target = ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from("a"),
        generation: Generation(1),
        level: ExpertLevel::Segment,
        base_model: base(),
        manifest_digest: ManifestDigest::new([1; 32]).unwrap(),
        state: ExpertState::Serving,
        state_version: 1,
    })
    .unwrap();
    let freeze = proposal(StructuralOp::Freeze { target });
    assert_eq!(
        ExpertCommitRecord::new_with_inputs(
            freeze,
            EVIDENCE,
            7,
            CONFIG,
            b"report".to_vec(),
            inputs()
        ),
        Err(ExpertProposalError::MalformedRecord(
            "inputs on an operation without a result"
        ))
    );
}

#[test]
fn version_3_construction_keeps_the_version_2_refusals() {
    assert!(matches!(
        ExpertCommitRecord::new_with_inputs(birth(), [0; 32], 7, CONFIG, b"r".to_vec(), inputs()),
        Err(ExpertProposalError::MalformedRecord(_))
    ));
    assert!(matches!(
        ExpertCommitRecord::new_with_inputs(birth(), EVIDENCE, 7, [0; 32], b"r".to_vec(), inputs()),
        Err(ExpertProposalError::MalformedRecord(_))
    ));
    assert!(matches!(
        ExpertCommitRecord::new_with_inputs(birth(), EVIDENCE, 7, CONFIG, Vec::new(), inputs()),
        Err(ExpertProposalError::MalformedRecord(_))
    ));
}

#[test]
fn truncated_and_padded_version_3_records_are_rejected() {
    let bytes = v3(birth(), inputs()).encode_canonical();
    for len in 0..bytes.len() {
        assert!(
            matches!(
                ExpertCommitRecord::decode_canonical(&bytes[..len]),
                Err(ExpertProposalError::MalformedRecord(_))
            ),
            "prefix of {len} bytes decoded"
        );
    }
    let mut padded = bytes.clone();
    padded.push(0);
    assert_eq!(malformed(&padded), "trailing bytes");
}

fn list_len(list: &[ExpertInputRef]) -> usize {
    4 + list
        .iter()
        .map(|i| 4 + i.identity().len() + 8 + 32)
        .sum::<usize>()
}

#[test]
fn a_non_canonical_or_invalid_input_block_is_rejected() {
    let record = v3(birth(), inputs());
    let bytes = record.encode_canonical();
    let eval_len = list_len(record.inputs().unwrap().evaluation());
    let train_len = list_len(record.inputs().unwrap().training());
    let start = bytes.len() - eval_len - train_len;
    assert_eq!(&bytes[start..start + 4], &2u32.to_le_bytes());

    // Swapping the two training entries leaves them unsorted: the one canonical order is
    // ascending, so the bytes are refused.
    let entry_a = 4 + "data:a".len() + 8 + 32;
    let entry_b = 4 + "data:b".len() + 8 + 32;
    let first = start + 4;
    let mut swapped = bytes.clone();
    let (a, b) = (
        bytes[first..first + entry_a].to_vec(),
        bytes[first + entry_a..first + entry_a + entry_b].to_vec(),
    );
    swapped.splice(first..first + entry_a + entry_b, b.into_iter().chain(a));
    assert_eq!(malformed(&swapped), "non-canonical encoding");

    // A duplicated entry is an invalid block.
    let mut duplicated = bytes.clone();
    let second_start = first + entry_a;
    let first_entry = bytes[first..first + entry_a].to_vec();
    duplicated.splice(second_start..second_start + entry_b, first_entry);
    assert_eq!(malformed(&duplicated), "invalid input block");

    // A zero digest, a zero generation and an empty identity are invalid references.
    let mut zero_digest = bytes.clone();
    zero_digest[first + entry_a - 32..first + entry_a].fill(0);
    assert_eq!(malformed(&zero_digest), "invalid input reference");
    let mut zero_generation = bytes.clone();
    zero_generation[first + 4 + "data:a".len()..first + 4 + "data:a".len() + 8].fill(0);
    assert_eq!(malformed(&zero_generation), "invalid input reference");
    let mut blank = bytes.clone();
    blank[first..first + 4].copy_from_slice(&0u32.to_le_bytes());
    assert!(ExpertCommitRecord::decode_canonical(&blank).is_err());

    // A list count above the bound is refused before any item is read.
    let mut over = bytes.clone();
    over[start..start + 4]
        .copy_from_slice(&((MAX_EXPERT_INPUTS_PER_ROLE as u32) + 1).to_le_bytes());
    assert!(ExpertCommitRecord::decode_canonical(&over).is_err());
    // A count larger than the remaining bytes is refused.
    let mut huge = bytes.clone();
    huge[start..start + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(ExpertCommitRecord::decode_canonical(&huge).is_err());

    // One digest for one pair across the lists is fine, two are a conflict.
    let conflict = ExpertInputs::new(vec![input("a", 1, 1)], vec![input("a", 1, 1)]).unwrap();
    let bytes = v3(birth(), conflict).encode_canonical();
    let mut tampered = bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert_eq!(malformed(&tampered), "invalid input block");
}

#[test]
fn the_input_block_is_part_of_the_canonical_bytes() {
    let a = v3(birth(), inputs()).encode_canonical();
    let b = v3(
        birth(),
        ExpertInputs::new(vec![input("data:a", 1, 0x11)], vec![]).unwrap(),
    )
    .encode_canonical();
    assert_ne!(a, b);
}
