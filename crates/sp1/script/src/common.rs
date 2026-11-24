//! Shared host utilities: build a demo ChainInput (the parity vectors) and
//! read one from a JSON file.

use chain_logic::ChainInput;
use std::fs;
use std::path::Path;

/// Load the parity-vector ChainInput produced by tools/gen_rust_vectors.py.
/// Returns (input, expected_accumulators).
pub fn load_demo_input() -> eyre::Result<(ChainInput, Vec<String>)> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../crates/memochain-core/tests/data");
    let text = fs::read_to_string(data_dir.join("parity.txt"))?;

    let mut tokens = text.split_whitespace();
    let mut sections: Vec<(String, Vec<String>)> = Vec::new();
    let mut section = String::new();
    let mut count = 0usize;
    let mut values: Vec<String> = Vec::new();
    loop {
        let Some(tok) = tokens.next() else { break };
        if count == 0 {
            if !section.is_empty() {
                sections.push((std::mem::take(&mut section), std::mem::take(&mut values)));
            }
            section = tok.to_string();
            count = tokens.next().expect("count").parse()?;
        } else {
            values.push(tok.to_string());
            count -= 1;
        }
    }
    if !section.is_empty() {
        sections.push((section, values));
    }
    let get = |name: &str| -> eyre::Result<&Vec<String>> {
        sections
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
            .ok_or_else(|| eyre::eyre!("missing section {name}"))
    };

    let to_i16 = |xs: &Vec<String>| -> eyre::Result<Vec<i16>> {
        xs.iter().map(|s| Ok(s.parse::<i64>()? as i16)).collect()
    };
    let n_docs: usize = get("N_DOCS")?[0].parse()?;
    let doc_dim: usize = get("DOC_DIM")?[0].parse()?;
    let docs_flat = to_i16(get("DOCS")?)?;
    let docs: Vec<Vec<i16>> = (0..n_docs)
        .map(|i| docs_flat[i * doc_dim..(i + 1) * doc_dim].to_vec())
        .collect();

    let input = ChainInput::from_vecs(
        to_i16(get("W_KEY")?)?,
        to_i16(get("W_VAL")?)?,
        get("BETA")?[0].parse()?,
        get("DECAY")?[0].parse()?,
        docs,
    );

    let acc_text = fs::read_to_string(data_dir.join("parity_accumulators.txt"))?;
    let toks: Vec<&str> = acc_text.split_whitespace().collect();
    assert_eq!(toks[0], "N");
    let accs: Vec<String> = toks[2..].iter().map(|s| s.to_string()).collect();

    Ok((input, accs))
}

/// Read a ChainInput JSON file.
pub fn load_input_file(path: &str) -> eyre::Result<ChainInput> {
    let json = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}
