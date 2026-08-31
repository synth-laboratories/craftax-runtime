use craftax_gamebench_gold::CraftaxRustSession;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

fn main() {
    let scenarios_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("gold_rust has task directory parent")
                .join("fixtures/gold/scenarios/scenarios.json")
        });
    match run(&scenarios_path) {
        Ok((scenarios, checks)) => {
            println!("RESTORE_EQUIVALENCE_OK scenarios={scenarios} checks={checks}");
        }
        Err(error) => {
            eprintln!("RESTORE_EQUIVALENCE_FAILED {error}");
            std::process::exit(1);
        }
    }
}

fn run(path: &PathBuf) -> anyhow::Result<(usize, usize)> {
    let data: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let scenarios = data
        .get("scenarios")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("scenarios must be array"))?;
    let mut checks = 0;
    for entry in scenarios {
        let actions = entry
            .get("actions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if actions.is_empty() {
            let session = CraftaxRustSession::reset_from_entry(entry)?;
            verify_immediate_restore(&session)?;
            checks += 1;
            continue;
        }
        let mut session = CraftaxRustSession::reset_from_entry(entry)?;
        for (index, action) in actions.iter().enumerate() {
            if session.is_done() {
                break;
            }
            session.step(action)?;
            verify_immediate_restore(&session)?;
            checks += 1;
            verify_split_rollout(entry, &actions, index + 1)?;
            checks += 1;
        }
    }
    Ok((scenarios.len(), checks))
}

fn verify_immediate_restore(session: &CraftaxRustSession) -> anyhow::Result<()> {
    let expected = snapshot(session);
    let blob = session.checkpoint_bytes()?;
    let mut restored = session.clone();
    restored.restore_checkpoint_bytes(&blob)?;
    assert_equivalent("immediate_restore", &expected, &snapshot(&restored))
}

fn verify_split_rollout(
    entry: &Value,
    actions: &[Value],
    split_after: usize,
) -> anyhow::Result<()> {
    let mut reference = CraftaxRustSession::reset_from_entry(entry)?;
    for action in actions {
        if reference.is_done() {
            break;
        }
        reference.step(action)?;
    }
    let expected = snapshot(&reference);

    let mut branch = CraftaxRustSession::reset_from_entry(entry)?;
    for action in actions.iter().take(split_after) {
        if branch.is_done() {
            break;
        }
        branch.step(action)?;
    }
    let blob = branch.checkpoint_bytes()?;
    branch.restore_checkpoint_bytes(&blob)?;
    for action in actions.iter().skip(split_after) {
        if branch.is_done() {
            break;
        }
        branch.step(action)?;
    }
    assert_equivalent(
        &format!(
            "{}@split={split_after}",
            entry
                .get("scenario_id")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        ),
        &expected,
        &snapshot(&branch),
    )
}

fn snapshot(session: &CraftaxRustSession) -> Value {
    let readout = session.readout();
    json!({
        "public": session.public,
        "private": session.private,
        "grid_hash": readout.get("grid_hash").cloned().unwrap_or(Value::Null),
        "valid_actions": readout.get("valid_actions").cloned().unwrap_or(Value::Null),
        "nev_cursor": session.events.len(),
    })
}

fn assert_equivalent(label: &str, expected: &Value, actual: &Value) -> anyhow::Result<()> {
    if expected == actual {
        return Ok(());
    }
    let expected_object = expected.as_object().cloned().unwrap_or_default();
    let actual_object = actual.as_object().cloned().unwrap_or_default();
    let mut keys: Vec<String> = expected_object
        .keys()
        .chain(actual_object.keys())
        .cloned()
        .collect();
    keys.sort();
    keys.dedup();
    for key in keys {
        if expected_object.get(&key) != actual_object.get(&key) {
            anyhow::bail!(
                "{label}: mismatch at {key}: {:?} != {:?}",
                expected_object.get(&key),
                actual_object.get(&key)
            );
        }
    }
    anyhow::bail!("{label}: snapshots differ")
}
