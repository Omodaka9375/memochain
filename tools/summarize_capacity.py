"""Summarize the capacity ablation sweep into runs/capacity_ablation/summary.json.

Run from the repo root on the training host:
    python tools/summarize_capacity.py [runs/capacity_ablation]
"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/capacity_ablation")

rows = []
for d in sorted(root.iterdir()):
    meta_p = d / "train_e2e_meta.json"
    if not meta_p.exists():
        continue
    m = json.loads(meta_p.read_text())
    h, k, v = m["num_heads"], m["key_dim"], m["val_dim"]
    history = m.get("history") or []
    best = min(history, key=lambda e: e["val_nll"]) if history else {}
    rows.append(
        {
            "config": d.name,
            "heads": h,
            "key_dim": k,
            "val_dim": v,
            "fact_slots": h * k,
            "state_bytes": m["state_bytes"],
            "state_kib": round(m["state_bytes"] / 1024, 1),
            "steps": m["steps"],
            "best_val_nll": best.get("val_nll"),
            "zero_nll": best.get("val_zero_nll"),
            "best_gain_nats": best.get("val_nll_gain"),
            "final_gain_nats": history[-1]["val_nll_gain"] if history else None,
            "elapsed_sec": m.get("elapsed_sec"),
            "beta_learned_mean": round(sum(m.get("beta_learned", [])) / max(len(m.get("beta_learned", [1])), 1), 4),
            "decay_learned_mean": round(sum(m.get("decay_learned", [])) / max(len(m.get("decay_learned", [1])), 1), 5),
        }
    )

out = root / "summary.json"
out.write_text(json.dumps(rows, indent=2) + "\n")
print(f"{'config':<10}{'KiB':>7}{'slots':>7}{'gain@last':>11}{'best_nll':>10}")
for r in rows:
    print(
        f"{r['config']:<10}{r['state_kib']:>7}{r['fact_slots']:>7}"
        f"{r['final_gain_nats']:>+11.4f}{r['best_val_nll']:>10.4f}"
    )
print(f"\nwrote {out}")
