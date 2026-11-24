//! Native runner: fold a collection with receipts and emit the public values
//! the SP1 guest will commit. Takes a ChainInput JSON on stdin (or --input
//! file) and writes PublicValues JSON to stdout.
//!
//! This is the pre-proof development loop: verify the chain logic locally,
//! then hand the identical input to the SP1 host for proving.

use chain_logic::{run_chain, ChainInput};
use std::io::Read;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = if args.len() > 2 && args[1] == "--input" {
        std::fs::read_to_string(&args[2]).unwrap_or_else(|e| {
            eprintln!("cannot read input file {}: {e}", args[2]);
            exit(1);
        })
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .unwrap_or_else(|e| {
                eprintln!("cannot read stdin: {e}");
                exit(1);
            });
        buf
    };

    let input: ChainInput = serde_json::from_str(&json).unwrap_or_else(|e| {
        eprintln!("invalid ChainInput JSON: {e}");
        eprintln!("expected: {{ w_key: [i16], w_val: [i16], beta: i64, decay: i64, n_docs: usize, docs_buf: [i16] }}");
        exit(1);
    });

    match run_chain(&input) {
        Ok(out) => {
            println!("{}", serde_json::json!({
                "step": out.step,
                "state_len": out.state.len(),
                "state_digest": out.state_digest,
                "accumulator": out.accumulator,
            }));
        }
        Err(e) => {
            eprintln!("chain failed: {e}");
            exit(1);
        }
    }
}
