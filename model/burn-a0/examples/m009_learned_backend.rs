//! Fail-closed M009 learned-backend contract probe.
//!
//! This executable is intentionally narrow: a future M009 entrypoint must use
//! it (or a byte-identical successor bound by a new study) so the frozen
//! contract is read before construction and its digest is embedded in, then
//! checked while reading, a checkpoint.  It is not a production backend and it
//! makes no quality claim.

use ptr_burn_a0::{load_bound, save_bound, M002V5Contract};
use std::{collections::BTreeMap, process::ExitCode};

const VOCABULARY: usize = 216;

fn arguments() -> Result<BTreeMap<String, String>, String> {
    let mut result = BTreeMap::new();
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let key = flag
            .strip_prefix("--")
            .ok_or_else(|| format!("expected --flag, got {flag:?}"))?;
        let value = values
            .next()
            .ok_or_else(|| format!("--{key} requires a value"))?;
        if result.insert(key.to_owned(), value).is_some() {
            return Err(format!("--{key} is duplicated"));
        }
    }
    Ok(result)
}

fn run() -> Result<(), String> {
    let args = arguments()?;
    let path = args
        .get("m002-v5-contract")
        .ok_or("--m002-v5-contract is required")?;
    let digest = args
        .get("m002-v5-contract-sha256")
        .ok_or("--m002-v5-contract-sha256 is required")?;
    if args.len() != 2 {
        return Err("M009 contract probe accepts no unbound override".to_owned());
    }
    let contract = M002V5Contract::read(path, digest).map_err(|error| error.to_string())?;
    contract
        .verify_training(0.10, 0.05, 0.10)
        .map_err(|error| error.to_string())?;
    let config = contract.config(VOCABULARY);
    let device = Default::default();
    let model = contract
        .init(&config, &device)
        .map_err(|error| error.to_string())?;
    let checkpoint = save_bound(&model, &config, &contract).map_err(|error| error.to_string())?;
    let _loaded =
        load_bound(&checkpoint, &config, &contract, &device).map_err(|error| error.to_string())?;
    println!(
        r#"{{"row":"m009-contract-bound","checkpoint_bytes":{}}}"#,
        checkpoint.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR: {error}");
            ExitCode::from(2)
        }
    }
}
