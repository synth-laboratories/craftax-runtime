use anyhow::Result;
use craftax_gamebench_gold::run_entry;
use serde_json::Value;
use std::io::{self, Read};

fn main() -> Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let entry: Value = serde_json::from_str(&input)?;
    let output = run_entry(&entry)?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
