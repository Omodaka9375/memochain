#!/usr/bin/env python3
"""Benchmark: one identical growing conversation replayed against a direct
model endpoint and one or more MemoChain proxy endpoints.

Per turn, the harness records what the BACKEND itself reports (usage.prompt_tokens),
plus wall-clock time to first token (TTFT) and total request time. Fold events
come from the proxy's memochain stats block. Output: JSON + Markdown report.

No hostnames are hardcoded; pass endpoints on the command line.

Example (local models, free to run):
  python tools/benchmark_savings.py \
      --turns 60 --tokens-per-turn 1200 --model MODEL \
      --endpoint direct=http://HOST:8000/v1 \
      --endpoint proxy=http://HOST:9000/v1 \
      --out tools/bench_results

Real-dollar version (OpenRouter meters usage.cost per request; pass the same
BYOK key to both sides, the proxy forwards it to the backend):
  python tools/benchmark_savings.py \
      --turns 60 --tokens-per-turn 1200 --model google/gemini-3.8-flash \
      --endpoint direct=https://openrouter.ai/api/v1 \
      --endpoint via-proxy=http://HOST:9001/v1 \
      --auth "Bearer sk-or-v1-YOURKEY" \
      --out tools/bench_results_openrouter
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

FILLER_WORD = "logistics"


def build_user_msg(turn: int, tokens_per_turn: int) -> dict:
    # 4 chars per token is the standard estimate; repeat a neutral word.
    words = max(1, tokens_per_turn * 4 // (len(FILLER_WORD) + 1))
    return {
        "role": "user",
        "content": f"Benchmark note {turn}: " + (FILLER_WORD + " ") * words,
    }


def run_endpoint(
    name: str,
    base_url: str,
    model: str,
    turns: int,
    tokens_per_turn: int,
    auth: str | None,
    max_out_tokens: int,
    timeout: float,
) -> dict:
    headers = {}
    if auth:
        headers["Authorization"] = auth
    session_id = f"bench-{name}-{int(time.time())}"
    headers["x-session-id"] = session_id
    history: list[dict] = []
    rows: list[dict] = []
    print(f"[{name}] starting session={session_id} url={base_url}", flush=True)
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for i in range(turns):
            user_msg = build_user_msg(i, tokens_per_turn)
            body = {
                "model": model,
                "max_tokens": max_out_tokens,
                "temperature": 0.0,
                "messages": history + [user_msg],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            t0 = time.perf_counter()
            ttft = None
            usage = None
            memo = None
            reply = ""
            with client.stream("POST", "/chat/completions", json=body, headers=headers) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"[{name}] turn {i}: HTTP {r.status_code}: {r.read()[:400]!r}")
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(f"[{name}] turn {i}: backend error: {chunk['error']}")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if chunk.get("memochain"):
                        memo = chunk["memochain"]
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        reply += delta.get("content") or ""
            total = time.perf_counter() - t0
            row = {
                "turn": i,
                "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                "cost": usage.get("cost") if usage else None,
                "ttft_s": round(ttft, 3) if ttft is not None else None,
                "total_s": round(total, 3),
            }
            if memo:
                row["memochain"] = memo
                if memo.get("fold"):
                    print(f"[{name}] turn {i}: FOLD {memo['fold']}", flush=True)
            rows.append(row)
            print(
                f"[{name}] turn {i + 1}/{turns} "
                f"prompt_tokens={row['prompt_tokens']} ttft={row['ttft_s']}s",
                flush=True,
            )
            history.append(user_msg)
            history.append({"role": "assistant", "content": reply or "(ok)"})
    return {"name": name, "base_url": base_url, "session_id": session_id, "turns": rows}


def _fold_turns(rows: list[dict]) -> set[int]:
    """Fold turns = the request carried a fold (memochain fold_now), or the
    prompt dropped below the previous turn (fold detection for backends that
    do not return the stats block)."""
    fold_turns: set[int] = set()
    prev_pt = None
    for r in rows:
        memo = r.get("memochain") or {}
        if memo.get("fold_now") or memo.get("fold"):
            fold_turns.add(r["turn"])
        pt = r.get("prompt_tokens")
        if pt is not None and prev_pt is not None and pt < prev_pt * 0.9:
            fold_turns.add(r["turn"])
        if pt is not None:
            prev_pt = pt
    return fold_turns


def summarize(result: dict) -> dict:
    rows = [r for r in result["turns"] if r["prompt_tokens"] is not None]
    n = len(rows)
    third = max(1, n // 3)
    fold_turns = _fold_turns(rows)
    normal = [r for r in rows if r["turn"] not in fold_turns]
    fold_rows = [r for r in rows if r["turn"] in fold_turns]
    n_ttft = [r["ttft_s"] for r in normal if r["ttft_s"] is not None]
    f_ttft = [r["ttft_s"] for r in fold_rows if r["ttft_s"] is not None]
    costs = [r["cost"] for r in rows if r.get("cost") is not None]
    return {
        "total_input_tokens": sum(r["prompt_tokens"] for r in rows),
        "final_prompt_tokens": rows[-1]["prompt_tokens"] if rows else None,
        "avg_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows)) if rows else None,
        "ttft_normal_mean_s": round(statistics.mean(n_ttft), 3) if n_ttft else None,
        "ttft_normal_median_s": round(statistics.median(n_ttft), 3) if n_ttft else None,
        "ttft_fold_mean_s": round(statistics.mean(f_ttft), 3) if f_ttft else None,
        "final_ttft_s": rows[-1]["ttft_s"] if rows else None,
        "total_cost": round(sum(costs), 6) if costs else None,
        "fold_count": len(fold_turns),
        "fold_turns": sorted(fold_turns),
    }


PRICE_POINTS = [  # USD per 1M input tokens
    ("Claude Sonnet class ($3.00)", 3.00),
    ("GPT/Gemini Pro class ($1.25)", 1.25),
    ("DeepSeek class ($0.27)", 0.27),
    ("Qwen open class ($0.20)", 0.20),
]


def write_report(out_dir: Path, meta: dict, results: list[dict]) -> Path:
    sums = {r["name"]: summarize(r) for r in results}
    direct = results[0]
    d = sums[direct["name"]]
    lines: list[str] = []
    lines.append(f"# MemoChain Savings Benchmark\n")
    lines.append(
        f"Generated {meta['generated_at']} | model `{meta['model']}` | "
        f"{meta['turns']} turns x ~{meta['tokens_per_turn']} tokens per turn "
        f"(identical conversation replayed against each endpoint).\n"
    )
    lines.append("Input tokens are as reported by the BACKEND `usage.prompt_tokens`, "
                 "not estimated by the harness. TTFT is wall-clock time to the first "
                 "streamed token.\n")

    lines.append("## Measured results\n")
    lines.append("| Endpoint | Total input tokens | Avg prompt/turn | Final-turn prompt | TTFT normal turns | TTFT fold turns | Folds |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        s = sums[r["name"]]
        fold_ttft = f"{s['ttft_fold_mean_s']} s" if s["ttft_fold_mean_s"] else "-"
        lines.append(
            f"| {r['name']} ({r['base_url']}) | {s['total_input_tokens']:,} "
            f"| {s['avg_prompt_tokens']:,} | {s['final_prompt_tokens']:,} "
            f"| {s['ttft_normal_median_s']} s (median) | {fold_ttft} | {s['fold_count']} |"
        )

    lines.append("\n## Savings vs direct (measured, this session length)\n")
    lines.append("| Endpoint | Input token reduction | TTFT reduction (normal turns vs direct) |")
    lines.append("| --- | --- | --- |")
    for r in results[1:]:
        s = sums[r["name"]]
        tok_red = 100.0 * (1 - s["total_input_tokens"] / d["total_input_tokens"])
        ttft_red = (
            100.0 * (1 - s["ttft_normal_median_s"] / d["ttft_normal_median_s"])
            if d["ttft_normal_median_s"] and s["ttft_normal_median_s"]
            else None
        )
        ttft_txt = f"{ttft_red:.0f}%" if ttft_red is not None else "n/a"
        lines.append(f"| {r['name']} | {tok_red:.0f}% | {ttft_txt} |")
    lines.append(
        "\nFold turns pay a one-request-per-~40k-folded-tokens summary call on top "
        "(measured here as the fold-turn TTFT); every other turn is a plain forward.\n"
    )

    lines.append("\n## Dollar cost of the measured input tokens (published price points)\n")
    lines.append("| Endpoint | " + " | ".join(p[0] for p in PRICE_POINTS) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in PRICE_POINTS) + " |")
    for r in results:
        s = sums[r["name"]]
        cells = " | ".join(f"${s['total_input_tokens'] * price / 1e6:.2f}" for _, price in PRICE_POINTS)
        lines.append(f"| {r['name']} | {cells} |")
    if d["total_cost"] is not None:
        lines.append("\nOpenRouter-metered cost: "
                     + ", ".join(f"{r['name']}: ${sums[r['name']]['total_cost']}" for r in results))

    lines.append("\n## Projection (same fold cadence, longer sessions)\n")
    lines.append("Post-fold prompt size is constant, so savings grow with session length. "
                 "Projection uses the measured average prompt of each endpoint:\n")
    lines.append("| Session length | Direct input tokens | Best-proxy input tokens | Reduction |")
    lines.append("| --- | --- | --- | --- |")
    best = min(results[1:], key=lambda r: sums[r["name"]]["avg_prompt_tokens"])
    bs = sums[best["name"]]
    for session in (200, 600, 2000):
        direct_tokens = session * meta["tokens_per_turn"] * (session + 1) / 2
        proxy_tokens = bs["avg_prompt_tokens"] * session
        red = 100.0 * (1 - proxy_tokens / direct_tokens)
        lines.append(
            f"| {session} turns (~{session * meta['tokens_per_turn'] / 1000:.0f}k tokens) "
            f"| {direct_tokens / 1e6:.1f}M | {proxy_tokens / 1e6:.1f}M | {red:.0f}% |"
        )

    lines.append("\n## Reproduce the real-dollar version (OpenRouter meters usage.cost)\n")
    lines.append("```\n")
    lines.append("python tools/benchmark_savings.py --turns 60 --tokens-per-turn 1200 \\")
    lines.append("  --model google/gemini-3.8-flash \\")
    lines.append("  --endpoint direct=https://openrouter.ai/api/v1 \\")
    lines.append("  --endpoint via-proxy=<your proxy URL>/v1 \\")
    lines.append('  --auth "Bearer sk-or-v1-YOURKEY" --out tools/bench_results_openrouter')
    lines.append("```\n")
    lines.append("The same BYOK key is sent to both sides; the proxy forwards it. "
                 "The report then shows OpenRouter's own metered cost for direct vs proxied traffic.\n")

    lines.append("## Raw data\n")
    for r in results:
        s = sums[r["name"]]
        turns = ', '.join(str(t) for t in s["fold_turns"]) or 'none'
        lines.append(f"- `{r['name']}` session id: `{r['session_id']}`, fold turns: {turns}")

    report = out_dir / "savings_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", action="append", required=True,
                    help="name=url, e.g. direct=http://localhost:8000/v1. First endpoint is the baseline.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=60)
    ap.add_argument("--tokens-per-turn", type=int, default=1200)
    ap.add_argument("--max-out-tokens", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--auth", default=None, help='Authorization header value, e.g. "Bearer sk-..."')
    ap.add_argument("--out", default="tools/bench_results")
    args = ap.parse_args()

    endpoints = []
    for spec in args.endpoint:
        name, _, url = spec.partition("=")
        if not url:
            ap.error(f"--endpoint must be name=url, got: {spec}")
        endpoints.append((name, url))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "turns": args.turns,
        "tokens_per_turn": args.tokens_per_turn,
        "max_out_tokens": args.max_out_tokens,
    }
    results = []
    for name, url in endpoints:
        results.append(
            run_endpoint(name, url, args.model, args.turns, args.tokens_per_turn,
                         args.auth, args.max_out_tokens, args.timeout)
        )
        time.sleep(1.0)

    payload = {"meta": meta,
               "results": [{**r, "summary": summarize(r)} for r in results]}
    (out_dir / "savings_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = write_report(out_dir, meta, results)
    print(f"\nreport: {report}")
    print(f"data:   {out_dir / 'savings_data.json'}")


if __name__ == "__main__":
    main()
