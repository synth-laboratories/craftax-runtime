use craftax_gamebench_gold::CraftaxRustSession;
use serde_json::{json, Value};
use std::thread;
use std::time::Instant;

const ACTION_PATTERN: [&str; 6] = ["right", "do", "down", "do", "left", "up"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--throughput") {
        println!(
            "{}",
            serde_json::to_string_pretty(&throughput_report(&args)).unwrap()
        );
        return;
    }
    let iterations = args
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(50);
    println!(
        "{}",
        serde_json::to_string_pretty(&checkpoint_report(iterations)).unwrap()
    );
}

fn checkpoint_report(iterations: usize) -> Value {
    let entry = json!({
        "scenario_id": "bench_policy_dev",
        "seed": 101,
        "world": {"use_default": "policy_dev_small", "seed": 101, "max_steps": 120},
        "rules": {"base": "symbolic_no_homeostasis"},
        "actions": []
    });
    let mut session = CraftaxRustSession::reset_from_entry(&entry).expect("reset bench session");
    let started = Instant::now();
    for idx in 0..120 {
        if session.is_done() {
            break;
        }
        session
            .step(&Value::String(
                ACTION_PATTERN[idx % ACTION_PATTERN.len()].to_string(),
            ))
            .expect("bench step");
    }
    let step_elapsed_s = started.elapsed().as_secs_f64();
    let mut save_ms = Vec::with_capacity(iterations);
    let mut restore_ms = Vec::with_capacity(iterations);
    let mut sizes = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let blob = session.checkpoint_bytes().expect("checkpoint encode");
        save_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        sizes.push(blob.len() as f64);
        let started = Instant::now();
        session
            .restore_checkpoint_bytes(&blob)
            .expect("checkpoint restore");
        restore_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    json!({
        "schema": "gamebench.craftax.rust_bench.v1",
        "lane": "rust",
        "mode": "checkpoint",
        "iterations": iterations,
        "step_elapsed_s": step_elapsed_s,
        "steps_per_s": if step_elapsed_s > 0.0 { 120.0 / step_elapsed_s } else { 0.0 },
        "checkpoint_bytes_p50": percentile(&mut sizes.clone(), 0.50),
        "checkpoint_bytes_p99": percentile(&mut sizes.clone(), 0.99),
        "checkpoint_save_p50_ms": percentile(&mut save_ms.clone(), 0.50),
        "checkpoint_save_p99_ms": percentile(&mut save_ms.clone(), 0.99),
        "checkpoint_restore_p50_ms": percentile(&mut restore_ms.clone(), 0.50),
        "checkpoint_restore_p99_ms": percentile(&mut restore_ms.clone(), 0.99),
        "last_blob_bytes": sizes.last().copied().unwrap_or(0.0) as usize
    })
}

fn throughput_report(args: &[String]) -> Value {
    let episodes = parse_flag(args, "--episodes", 20);
    let max_steps = parse_flag(args, "--max-steps", 120);
    let workers = parse_flag(args, "--workers", 4).max(1).min(episodes.max(1));
    let started = Instant::now();
    let chunk = episodes.div_ceil(workers);
    let mut handles = Vec::new();
    for worker in 0..workers {
        let start = worker * chunk;
        let end = ((worker + 1) * chunk).min(episodes);
        if start >= end {
            continue;
        }
        handles.push(thread::spawn(move || {
            let mut results = Vec::new();
            for index in start..end {
                results.push(run_episode(101 + index as i64, max_steps));
            }
            results
        }));
    }
    let mut outcomes = Vec::new();
    for handle in handles {
        outcomes.extend(handle.join().expect("worker joined"));
    }
    let elapsed_s = started.elapsed().as_secs_f64();
    let total_steps: i64 = outcomes
        .iter()
        .map(|outcome| outcome["steps"].as_i64().unwrap_or(0))
        .sum();
    json!({
        "schema": "gamebench.craftax.rollout_throughput.v1",
        "lane": "rust",
        "engine_mode": "rust_in_process",
        "episode_count": outcomes.len(),
        "workers": workers,
        "max_steps": max_steps,
        "total_steps": total_steps,
        "elapsed_s": elapsed_s,
        "steps_per_s": if elapsed_s > 0.0 { total_steps as f64 / elapsed_s } else { 0.0 },
        "episodes_per_s": if elapsed_s > 0.0 { outcomes.len() as f64 / elapsed_s } else { 0.0 },
        "sample_episodes": outcomes.into_iter().take(5).collect::<Vec<_>>()
    })
}

fn run_episode(seed: i64, max_steps: usize) -> Value {
    let task = json!({
        "schema": "gamebench.task.craftax.v1",
        "task_id": format!("throughput_policy_dev_{seed}"),
        "scenario_id": format!("throughput_policy_dev_{seed}"),
        "world": {"use_default": "policy_dev_small", "seed": seed, "max_steps": max_steps},
        "rules": {"base": "symbolic_no_homeostasis"},
        "readouts": {"profile": "symbolic_compact"}
    });
    let mut session =
        CraftaxRustSession::reset_from_task(&task, Some(seed)).expect("reset episode");
    let mut cursor = 0;
    while !session.is_done()
        && session
            .private
            .get("step_index")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            < max_steps as i64
    {
        session
            .step(&Value::String(
                ACTION_PATTERN[cursor % ACTION_PATTERN.len()].to_string(),
            ))
            .expect("episode step");
        cursor += 1;
    }
    json!({
        "seed": seed,
        "steps": session.private.get("step_index").cloned().unwrap_or(json!(0)),
        "reward": session.private.get("total_reward").cloned().unwrap_or(json!(0.0)),
        "done_reason": session.private.get("done_reason").cloned().unwrap_or(Value::Null),
        "grid_hash": session.readout().get("grid_hash").cloned().unwrap_or(Value::Null)
    })
}

fn parse_flag(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values[index]
}
