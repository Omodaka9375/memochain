//! Execute the MemoChain guest in the zkVM without proving (fast dev loop).
//! Verifies the committed public values against the Python parity vectors.

use sp1_sdk::{Prover, ProverClient, SP1Stdin};
use std::time::Instant;

/// Built by `cargo prove build` (or script/build.rs) into the elf-compilation dir.
fn load_elf() -> Vec<u8> {
    let candidates = [
        "../program/target/elf-compilation/riscv64im-succinct-zkvm-elf/release/memochain-program",
        "../../program/target/elf-compilation/riscv64im-succinct-zkvm-elf/release/memochain-program",
    ];
    for c in candidates {
        if let Ok(bytes) = std::fs::read(c) {
            return bytes;
        }
    }
    panic!("memochain-program ELF not found; run `cargo prove build` in crates/sp1/program");
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    sp1_sdk::utils::setup_logger();
    let (input, accs) = memochain_sp1_script::common::load_demo_input()?;

    println!("executing MemoChain guest on {} docs ...", input.n_docs);
    let client = ProverClient::builder().cpu().build().await;
    let mut stdin = SP1Stdin::new();
    stdin.write(&input);

    let elf = load_elf();
    let start = Instant::now();
    let (mut public, exec_info) = client.execute(elf.into(), stdin).await?;
    println!(
        "executed in {:.2}s, cycles: {}",
        start.elapsed().as_secs_f64(),
        exec_info.total_instruction_count()
    );

    // Public values: step, state, state_digest, accumulator (commit order).
    let step: u64 = public.read();
    let state: Vec<i16> = public.read();
    let state_digest: String = public.read();
    let accumulator: String = public.read();

    println!("step: {step}");
    println!("state bytes: {}", state.len() * 2);
    println!("state_digest: {state_digest}");
    println!("accumulator:  {accumulator}");

    // Parity: the zkVM must produce the same chain as the Python reference.
    assert_eq!(state.len(), chain_logic::STATE_PARAMS);
    assert_eq!(accumulator, accs.last().unwrap().as_str(), "accumulator differs from Python");
    println!("values match the Python reference chain!");

    Ok(())
}
