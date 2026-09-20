//! The codebook as a cross-layer contract, and the validity mask as something the
//! model cannot learn around.
use ptr_types::{
    CodeFamily, Codebook, CodebookVersion, EpistemicState, ReasoningOperator, SemanticRole,
    UncertaintyKind, Validity, ValidityMask,
};

/// Decode `canonical_bytes` back into the assignment it commits to.
///
/// Written independently of the encoder so a change to either side has to be
/// deliberate, in the same spirit as the repository's hashlib golden vectors.
fn decode(bytes: &[u8]) -> (u32, Vec<(String, Vec<String>)>) {
    let mut offset = 0usize;
    let magic = b"PTRCODEBOOK\x00";
    assert_eq!(&bytes[..magic.len()], magic);
    offset += magic.len();
    let version = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let families = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
    offset += 2;

    let text = |offset: &mut usize| {
        let length = u16::from_le_bytes(bytes[*offset..*offset + 2].try_into().unwrap()) as usize;
        *offset += 2;
        let value = String::from_utf8(bytes[*offset..*offset + length].to_vec()).unwrap();
        *offset += length;
        value
    };

    let mut decoded = Vec::new();
    for _ in 0..families {
        let family = text(&mut offset);
        let cardinality =
            u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;
        let members: Vec<String> = (0..cardinality).map(|_| text(&mut offset)).collect();
        decoded.push((family, members));
    }
    assert_eq!(offset, bytes.len(), "every byte is accounted for");
    (version, decoded)
}

#[test]
fn canonical_bytes_are_a_golden_record_of_the_v1_assignment() {
    let (version, families) = decode(&Codebook::V1.canonical_bytes());
    assert_eq!(version, 1);
    assert_eq!(
        families,
        vec![
            (
                "semantic_role".to_owned(),
                vec![
                    "goal",
                    "constraint",
                    "claim",
                    "evidence",
                    "resource",
                    "capability",
                    "relation",
                    "procedure",
                    "action",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<String>>(),
            ),
            (
                "epistemic_state".to_owned(),
                vec![
                    "unknown",
                    "assumed",
                    "hypothesis",
                    "observed",
                    "inferred",
                    "verified",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            ),
            (
                "uncertainty_kind".to_owned(),
                vec!["point", "interval", "distribution"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            ),
            (
                "reasoning_operator".to_owned(),
                vec![
                    "semantic",
                    "deductive",
                    "probabilistic",
                    "statistical",
                    "temporal",
                    "causal",
                    "search",
                    "optimization",
                    "simulation",
                    "symbolic",
                    "external_pod",
                ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            ),
            (
                "validity".to_owned(),
                vec!["live", "superseded", "revoked", "disputed"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            ),
        ]
    );
}

#[test]
fn the_order_committed_by_the_bytes_is_the_code_order() {
    let book = Codebook::V1;
    let (_, families) = decode(&book.canonical_bytes());
    let (_, roles) = &families[0];
    // The position of a name in the committed bytes is exactly its tensor index, so
    // hashing these bytes detects any reassignment a checkpoint would be built on.
    for (index, name) in roles.iter().enumerate() {
        let member: SemanticRole = book
            .member_of(book.assignment_of::<SemanticRole>().unwrap()[index].0)
            .unwrap();
        assert_eq!(
            ptr_types::CognitiveType::name(member),
            name.as_str(),
            "index {index}"
        );
    }
}

#[test]
fn a_stored_code_is_meaningless_without_its_version() {
    let book = Codebook::V1;
    let stored = book.code_of(ReasoningOperator::Causal).unwrap();
    assert_eq!(stored.index(), 5);

    // Reading it back needs the same version; an unknown one is refused outright
    // rather than interpreted under whatever tables this build happens to have.
    assert_eq!(
        book.member_of::<ReasoningOperator>(stored).unwrap(),
        ReasoningOperator::Causal
    );
    let error = Codebook::at(CodebookVersion(2)).unwrap_err();
    assert_eq!(error.code(), "PTR_CODEBOOK_UNKNOWN_VERSION");
}

#[test]
fn embedding_tables_can_be_sized_from_the_codebook_alone() {
    let book = Codebook::V1;
    // What a model needs to allocate its metadata channels, taken from the
    // contract rather than hard-coded beside it.
    assert_eq!(book.cardinality_of::<SemanticRole>(), 9);
    assert_eq!(book.cardinality_of::<EpistemicState>(), 6);
    assert_eq!(book.cardinality_of::<UncertaintyKind>(), 3);
    assert_eq!(book.cardinality_of::<ReasoningOperator>(), 11);
    assert_eq!(book.cardinality_of::<Validity>(), 4);
    for family in CodeFamily::ALL {
        assert!(book.cardinality(family) > 0, "{}", family.name());
    }
}

#[test]
fn a_revoked_slot_keeps_its_codes_and_still_cannot_participate() {
    let book = Codebook::V1;
    // Four typed slots as a model would receive them: metadata per slot, plus the
    // lifecycle state that decides admission.
    let slots = [
        (
            SemanticRole::Claim,
            EpistemicState::Verified,
            Validity::Live,
        ),
        (
            SemanticRole::Evidence,
            EpistemicState::Observed,
            Validity::Revoked,
        ),
        (SemanticRole::Goal, EpistemicState::Assumed, Validity::Live),
        (
            SemanticRole::Claim,
            EpistemicState::Inferred,
            Validity::Disputed,
        ),
    ];

    let mask = ValidityMask::from_validities(
        &slots
            .iter()
            .map(|(_, _, validity)| *validity)
            .collect::<Vec<Validity>>(),
    );

    // Every slot still has well-formed metadata codes, including the excluded ones.
    // That separation is the architectural point: admission is not encoded in the
    // embedding, so no amount of training can recover an excluded slot.
    for (role, state, _) in slots {
        assert!(book.code_of(role).is_ok());
        assert!(book.code_of(state).is_ok());
    }
    assert_eq!(
        book.code_of(slots[1].0).unwrap(),
        book.code_of(SemanticRole::Evidence).unwrap()
    );

    assert_eq!(mask.admitted_slots().collect::<Vec<_>>(), [0, 2]);
    let bias = mask.attention_bias();
    assert_eq!(bias.iter().filter(|value| **value == 0.0).count(), 2);
    assert!(bias[1].is_infinite() && bias[3].is_infinite());

    // A later stage may narrow further, never reopen.
    let mut narrowed = mask.clone();
    narrowed.exclude(0).unwrap();
    assert!(narrowed.is_narrowing_of(&mask));
    assert!(!mask.is_narrowing_of(&narrowed));
    narrowed.narrow(&ValidityMask::admitting_all(4)).unwrap();
    assert_eq!(narrowed.admitted_slots().collect::<Vec<_>>(), [2]);
}
