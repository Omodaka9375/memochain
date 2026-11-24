//! Prove the MemoChain write in the zkVM and verify the proof.
//! Default prover is local CPU; set SP1_PROVER=network for the hosted prover.

use sp1_sdk::{HashableKey, ProveRequest, Prover, ProverClient, ProvingKey, SP1Stdin};
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

    println!("proving MemoChain write on {} docs ...", input.n_docs);
    let client = ProverClient::builder().cpu().build().await;
    let mut stdin = SP1Stdin::new();
    stdin.write(&input);

    let elf = load_elf();
    let pk = client.setup(elf.into()).await?;
    println!("vk: {:?}", pk.verifying_key().hash_u32());

    let compressed = std::env::args().any(|a| a == "--compressed");
    let start = Instant::now();
    let mut proof = if compressed {
        println!("mode: compressed (recursive STARK)");
        client.prove(&pk, stdin).compressed().await?
    } else {
        println!("mode: core");
        client.prove(&pk, stdin).core().await?
    };
    let prove_secs = start.elapsed().as_secs_f64();
    println!("proved in {prove_secs:.2}s");

    let proof_path = if compressed {
        "memochain_compressed_proof.bin"
    } else {
        "memochain_core_proof.bin"
    };
    proof
        .save(proof_path)
        .map_err(|e| eyre::eyre!("failed to save proof: {e}"))?;
    let proof_bytes = std::fs::metadata(proof_path)?.len();
    println!("saved proof to {proof_path} ({proof_bytes} bytes)");

    let vstart = Instant::now();
    client.verify(&proof, pk.verifying_key(), None)?;
    println!("proof verified in {:.2}ms!", vstart.elapsed().as_secs_f64() * 1000.0);

    let public = &mut proof.public_values;
    let step: u64 = public.read();
    let state: Vec<i16> = public.read();
    let state_digest: String = public.read();
    let accumulator: String = public.read();
    println!("step:         {step}");
    println!("state bytes:  {}", state.len() * 2);
    println!("state_digest: {state_digest}");
    println!("accumulator:  {accumulator}");
    assert_eq!(state.len(), chain_logic::STATE_PARAMS);
    assert_eq!(accumulator, accs.last().unwrap().as_str(), "accumulator differs from Python");
    println!("proof carries the verified 22 KB memory write.");
    Ok(())
}
