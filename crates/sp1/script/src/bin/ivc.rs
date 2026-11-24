//! Mina-style recursive IVC host CLI for MemoChain's quantized student writer.
//!
//! Subcommands:
//!   execute  - Run the zkVM guest (with deferred proof verification if --prev-proof)
//!              and report exact RISC-V cycle count + committed digests.
//!   prove    - Generate a compressed recursive STARK proof (genesis or +N doc
//!              recursive update from --prev-proof), verify it, save proof, and
//!              optionally write a MemoChain Capsule v1 extracted directly from
//!              the verified proof's public values.
//!   verify   - Cryptographically verify a saved compressed proof from disk and
//!              print its committed step, weights_digest, state_digest, accumulator.

use chain_logic::ivc::{encode_capsule_v1, IvcPublicValues};
use clap::{Parser, Subcommand};
use memochain_core::student::{compute_projections_raw, StudentWriter, EMBED_DIM, FREIVALDS_THRESHOLD};
use sp1_sdk::{
    HashableKey, ProveRequest, Prover, ProverClient, ProvingKey, SP1Proof,
    SP1ProofWithPublicValues, SP1Stdin,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "ivc", about = "MemoChain Mina-style recursive SP1 IVC prover/verifier")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Execute {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        docs: PathBuf,
        #[arg(long)]
        prev_proof: Option<PathBuf>,
    },
    Prove {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        docs: PathBuf,
        #[arg(long)]
        prev_proof: Option<PathBuf>,
        #[arg(long)]
        out_proof: PathBuf,
        #[arg(long)]
        out_capsule: Option<PathBuf>,
        #[arg(long)]
        out_meta: Option<PathBuf>,
    },
    Verify {
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        out_capsule: Option<PathBuf>,
    },
}

fn load_ivc_elf() -> Vec<u8> {
    let candidates = [
        "../ivc-program/target/elf-compilation/riscv64im-succinct-zkvm-elf/release/memochain-ivc-program",
        "../../ivc-program/target/elf-compilation/riscv64im-succinct-zkvm-elf/release/memochain-ivc-program",
    ];
    for c in candidates {
        if let Ok(bytes) = fs::read(c) {
            return bytes;
        }
    }
    panic!("memochain-ivc-program ELF not found; run `cargo prove build` in crates/sp1/ivc-program");
}

fn build_stdin(
    vk: &sp1_sdk::SP1VerifyingKey,
    weights_path: &PathBuf,
    docs_path: &PathBuf,
    prev_proof_path: Option<&PathBuf>,
) -> eyre::Result<(SP1Stdin, usize, bool)> {
    let weights_raw = fs::read(weights_path)?;
    let docs_raw = fs::read(docs_path)?;
    let writer = StudentWriter::from_bytes(&weights_raw).map_err(|e| eyre::eyre!(e))?;
    eyre::ensure!(
        !docs_raw.is_empty() && docs_raw.len() % (EMBED_DIM * 2) == 0,
        "invalid docs_raw size: {}",
        docs_raw.len()
    );
    let n_docs = docs_raw.len() / (EMBED_DIM * 2);

    let proj_raw = if n_docs > FREIVALDS_THRESHOLD {
        println!("precomputing exact int64 Freivalds projections for {n_docs} docs ...");
        compute_projections_raw(&writer, &docs_raw)
    } else {
        Vec::new()
    };

    let mut stdin = SP1Stdin::new();
    let self_vk_digest = vk.hash_u32();
    stdin.write(&self_vk_digest);
    stdin.write_vec(weights_raw);
    stdin.write_vec(docs_raw);
    stdin.write_vec(proj_raw);

    let is_recursive = if let Some(pp) = prev_proof_path {
        let prev = SP1ProofWithPublicValues::load(pp)
            .map_err(|e| eyre::eyre!("failed to load prev proof {}: {e}", pp.display()))?;
        let prev_pv_bytes = prev.public_values.as_slice().to_vec();
        let prev_inner = match prev.proof {
            SP1Proof::Compressed(inner) => *inner,
            _ => eyre::bail!("prev proof must be a compressed SP1 proof for recursion"),
        };
        stdin.write_vec(prev_pv_bytes);
        stdin.write_proof(prev_inner, vk.vk.clone());
        true
    } else {
        stdin.write_vec(Vec::new());
        false
    };

    Ok((stdin, n_docs, is_recursive))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    sp1_sdk::utils::setup_logger();
    let cli = Cli::parse();

    let client = ProverClient::builder().cpu().build().await;
    let elf = load_ivc_elf();
    let pk = client.setup(elf.clone().into()).await?;
    let vk = pk.verifying_key().clone();
    println!("ivc vk_digest: {:?}", vk.hash_u32());

    match cli.cmd {
        Cmd::Execute { weights, docs, prev_proof } => {
            let (stdin, n_docs, is_rec) = build_stdin(&vk, &weights, &docs, prev_proof.as_ref())?;
            let mode = if is_rec { "recursive" } else { "genesis" };
            println!("executing IVC ({mode}) on {n_docs} docs ...");
            let t0 = Instant::now();
            let (pv_buf, report) = client.execute(elf.into(), stdin).await?;
            let secs = t0.elapsed().as_secs_f64();
            let pv: IvcPublicValues = bincode::deserialize(pv_buf.as_slice())?;
            println!("executed in {secs:.2}s, total cycles: {}", report.total_instruction_count());
            println!("step:           {}", pv.step);
            println!("weights_digest: {}", pv.weights_digest);
            println!("state_digest:   {}", pv.state_digest);
            println!("accumulator:    {}", pv.accumulator);
        }

        Cmd::Prove { weights, docs, prev_proof, out_proof, out_capsule, out_meta } => {
            let (stdin, n_docs, is_rec) = build_stdin(&vk, &weights, &docs, prev_proof.as_ref())?;
            let mode = if is_rec { "recursive" } else { "genesis" };
            println!("proving compressed IVC ({mode}) on {n_docs} docs ...");
            let t0 = Instant::now();
            let proof = client.prove(&pk, stdin).compressed().await?;
            let prove_secs = t0.elapsed().as_secs_f64();
            println!("proved ({mode}, {n_docs} docs) in {prove_secs:.2}s");

            if let Some(parent) = out_proof.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            proof
                .save(&out_proof)
                .map_err(|e| eyre::eyre!("failed to save proof: {e}"))?;
            let proof_bytes = fs::metadata(&out_proof)?.len();
            println!("saved proof to {} ({proof_bytes} bytes)", out_proof.display());

            let v0 = Instant::now();
            client.verify(&proof, &vk, None)?;
            let verify_ms = v0.elapsed().as_secs_f64() * 1000.0;
            println!("verified compressed proof in {verify_ms:.2}ms!");

            let pv: IvcPublicValues = bincode::deserialize(proof.public_values.as_slice())?;
            println!("step:           {}", pv.step);
            println!("weights_digest: {}", pv.weights_digest);
            println!("state_digest:   {}", pv.state_digest);
            println!("accumulator:    {}", pv.accumulator);

            if let Some(cap_path) = out_capsule {
                let cap_bytes = encode_capsule_v1(&pv).map_err(|e| eyre::eyre!(e))?;
                if let Some(parent) = cap_path.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(&cap_path, &cap_bytes)?;
                println!("wrote proven capsule to {} ({} bytes)", cap_path.display(), cap_bytes.len());
            }

            if let Some(meta_path) = out_meta {
                let meta = serde_json::json!({
                    "mode": mode,
                    "n_docs_folded": n_docs,
                    "step": pv.step,
                    "vk_digest": pv.vk_digest,
                    "weights_digest": pv.weights_digest,
                    "state_digest": pv.state_digest,
                    "accumulator": pv.accumulator,
                    "proof_bytes": proof_bytes,
                    "prove_secs": (prove_secs * 100.0).round() / 100.0,
                    "verify_ms": (verify_ms * 100.0).round() / 100.0,
                });
                if let Some(parent) = meta_path.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                fs::write(&meta_path, serde_json::to_string_pretty(&meta)? + "\n")?;
                println!("wrote metadata to {}", meta_path.display());
            }
        }

        Cmd::Verify { proof: proof_path, out_capsule } => {
            let proof = SP1ProofWithPublicValues::load(&proof_path)
                .map_err(|e| eyre::eyre!("failed to load proof: {e}"))?;
            let v0 = Instant::now();
            client.verify(&proof, &vk, None)?;
            let verify_ms = v0.elapsed().as_secs_f64() * 1000.0;
            let pv: IvcPublicValues = bincode::deserialize(proof.public_values.as_slice())?;
            println!("VERIFIED in {verify_ms:.2}ms!");
            println!("step:           {}", pv.step);
            println!("weights_digest: {}", pv.weights_digest);
            println!("state_digest:   {}", pv.state_digest);
            println!("accumulator:    {}", pv.accumulator);
            if let Some(cap_path) = out_capsule {
                let cap_bytes = encode_capsule_v1(&pv).map_err(|e| eyre::eyre!(e))?;
                fs::write(&cap_path, &cap_bytes)?;
                println!("extracted verified capsule to {} ({} bytes)", cap_path.display(), cap_bytes.len());
            }
        }
    }

    Ok(())
}
