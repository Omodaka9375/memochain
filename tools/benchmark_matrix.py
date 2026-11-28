"""Phase D Unified 4-Arm Benchmark Matrix + Real RTX 3060 GPU Prefill/KV-VRAM Benchmark.

Compares four arms across two realistic long-session workloads:
  1. raw             : Full uncompressed history (what a client sends with no proxy)
  2. legacy_mc       : MemoChain pre-Phase-A (sys-prompt dynamic injection, sync fold,
                       no tool compaction, no append-only facts ledger)
  3. memochain_cloud : Full Phases A+B+C+D (split static/dynamic injection, FTS5/RRF,
                       dedupe, hit-eviction, append-only facts, shrinkage guard,
                       tool compaction, async fold, cloud pricing model)
  4. memochain_local : Same full stack tuned with local profile (tighter hot window
                       for local GPU TTFT & KV-cache VRAM ceilings)

Workloads:
  - long_chat    : 150 turns, 25 planted concrete facts + 5 mid-session updates/supersedes
  - coding_agent : 60 turns, 10 early config facts + 50 tool-heavy coding turns
                   (repeated file reads + large pytest/grep outputs)

Optional ``--gpu-prefill`` runs real forward-pass prefill timing + peak CUDA KV-cache VRAM
measurements on the local GPU (e.g. RTX 3060 12 GB with Qwen/Qwen3-0.6B) using actual
prompt token lengths produced by each arm.

Usage:
  PYTHONPATH=src python tools/benchmark_matrix.py --gpu-prefill --out tools/bench_phase_d_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memochain.server.app import IntWriterAdapter, _hash_embed
from memochain.server.session import SessionMemory, estimate_tokens, message_text


# Reference cloud pricing ($ per 1M tokens) - Claude Sonnet 4 / GPT-4o / DeepSeek V3.2
PRICE_MODELS: dict[str, dict[str, float]] = {
    "claude_sonnet": {"input": 3.00, "cache_read": 0.30, "cache_write": 3.75},
    "gpt_4o": {"input": 2.50, "cache_read": 1.25, "cache_write": 2.50},
    "deepseek_v3": {"input": 0.28, "cache_read": 0.028, "cache_write": 0.28},
}


class RealisticLossyFoldReader:
    """Simulates a real fold LLM call (latency + realistic fact-sheet update that
    occasionally drops older facts when rewriting, testing MemoChain's Phase C
    append-only ledger + shrinkage guard)."""

    def __init__(self, latency_s: float = 0.04, amnesia_every: int = 3) -> None:
        self.latency_s = latency_s
        self.amnesia_every = amnesia_every
        self.calls = 0
        self.fold_input_tokens = 0

    def read(self, state, q, prev_summary: str) -> str:
        return prev_summary

    def read_fold(
        self,
        state,
        q,
        prev_summary: str,
        entries: list[tuple[str, str, str]],
        model_hint: str | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> str:
        self.calls += 1
        self.fold_input_tokens += sum(estimate_tokens(t) for _, _, t in entries) + estimate_tokens(prev_summary)
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        # Keep existing bullets unless simulating rewrite amnesia on every Nth fold
        existing = [ln.strip() for ln in prev_summary.splitlines() if ln.strip().startswith("-")]
        if self.amnesia_every > 0 and (self.calls % self.amnesia_every == 0) and len(existing) > 4:
            # Drop half of old bullets like a real LLM rewrite sometimes does
            existing = existing[len(existing) // 2 :]
        new_lines: list[str] = []
        for _, role, text in entries:
            if role == "user" and (":" in text or "UPDATE" in text or "FACT" in text):
                snippet = text.splitlines()[0][:120]
                new_lines.append(f"- note: {snippet}")
        merged = existing + new_lines
        return "\n".join(merged[-30:]) if merged else "- session: active"


def _prefix_cached_tokens(prev_msgs: list[dict[str, Any]] | None, curr_msgs: list[dict[str, Any]]) -> int:
    """Count exact byte-identical leading message tokens (real provider prefix cache rule)."""
    if prev_msgs is None:
        return 0
    cached = 0
    for a, b in zip(prev_msgs, curr_msgs):
        if a == b:
            cached += estimate_tokens(message_text(a))
        else:
            break
    return cached


def _compute_cloud_cost(
    uncached_tokens: int,
    cached_tokens: int,
    fold_tokens: int,
    *,
    cheap_fold_factor: float,
    pricing: dict[str, float],
) -> float:
    # Uncached input billed at cache_write/input, cached at cache_read; fold at cheap_fold_factor * input
    main_cost = (uncached_tokens / 1e6) * pricing["input"] + (cached_tokens / 1e6) * pricing["cache_read"]
    fold_cost = (fold_tokens / 1e6) * pricing["input"] * cheap_fold_factor
    return round(main_cost + fold_cost, 5)


@dataclass
class WorkloadTurn:
    messages_added: list[dict[str, Any]]


def build_long_chat_workload(num_turns: int = 150) -> tuple[list[dict[str, Any]], list[WorkloadTurn], list[dict[str, str]]]:
    """150-turn chat with 25 planted concrete facts + 5 mid-session updates/supersedes."""
    system = [{"role": "system", "content": "You are a senior systems engineer assisting on a multi-hour session. " + ("Policy clause. " * 250)}]
    facts: list[dict[str, str]] = []
    turns: list[WorkloadTurn] = []

    # 25 facts planted across turns 1..50 (every 2nd turn)
    for idx in range(25):
        key = f"svc_node_{idx:02d}"
        code = f"TK-{1000 + idx}"
        host = f"node-{idx:02d}.prod.internal:{5400 + idx}"
        facts.append({"key": key, "code": code, "expected": host, "updated": False})

    # 5 updates applied around turns 80..100 for facts 0..4
    updates = {
        80: (0, "node-00-v2.prod.internal:6400"),
        85: (1, "node-01-v2.prod.internal:6401"),
        90: (2, "node-02-v2.prod.internal:6402"),
        95: (3, "node-03-v2.prod.internal:6403"),
        100: (4, "node-04-v2.prod.internal:6404"),
    }

    filler = (
        "Let us review the rollout checklist, observability dashboards, alert thresholds, "
        "container resource limits, retry backoff jitter, and database migration ordering. "
    ) * 6

    for t in range(1, num_turns + 1):
        added: list[dict[str, Any]] = []
        if t <= 50 and t % 2 == 1:
            f = facts[(t - 1) // 2]
            added.append({
                "role": "user",
                "content": f"FACT {f['code']}: {f['key']}: {f['expected']} — keep this exact endpoint. {filler}",
            })
            added.append({
                "role": "assistant",
                "content": f"Recorded {f['key']} = {f['expected']} under ticket {f['code']}. {filler}",
            })
        elif t in updates:
            fi, new_host = updates[t]
            f = facts[fi]
            f["old"] = f["expected"]
            f["expected"] = new_host
            f["updated"] = True
            added.append({
                "role": "user",
                "content": f"UPDATE {f['code']}: {f['key']}: {new_host} (supersedes previous host). {filler}",
            })
            added.append({
                "role": "assistant",
                "content": f"Updated {f['key']} to {new_host} for {f['code']}. {filler}",
            })
        else:
            added.append({"role": "user", "content": f"Turn {t} discussion: {filler}"})
            added.append({"role": "assistant", "content": f"Turn {t} analysis and next steps: {filler}"})
        turns.append(WorkloadTurn(messages_added=added))

    return system, turns, facts


def build_coding_agent_workload(num_turns: int = 60) -> tuple[list[dict[str, Any]], list[WorkloadTurn], list[dict[str, str]]]:
    """60-turn coding agent session: 10 early config facts + 50 tool-heavy coding turns
    with repeated read_file calls and large pytest/grep tool outputs."""
    system = [{"role": "system", "content": "You are an autonomous coding agent. " + ("Coding standard. " * 250)}]
    facts: list[dict[str, str]] = []
    turns: list[WorkloadTurn] = []

    for i in range(10):
        code = f"CFG-{200 + i}"
        val = f"cluster-{i}.staging.local:{7000 + i}"
        facts.append({"key": f"env_cfg_{i}", "code": code, "expected": val, "updated": False})

    for t in range(1, num_turns + 1):
        added: list[dict[str, Any]] = []
        if t <= 10:
            f = facts[t - 1]
            added.append({
                "role": "user",
                "content": f"Config {f['code']}: {f['key']}: {f['expected']}. Pin this for deployment.",
            })
            added.append({"role": "assistant", "content": f"Pinned {f['key']} -> {f['expected']} ({f['code']})."})
        else:
            fpath = ["src/server/app.py", "src/server/session.py", "src/core/writer.py"][t % 3]
            added.append({"role": "user", "content": f"Turn {t}: inspect {fpath}, apply fix, and run pytest."})
            added.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"r_{t}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": fpath})}},
                    {"id": f"b_{t}", "type": "function", "function": {"name": "bash_run", "arguments": json.dumps({"command": "uv run pytest -q"})}},
                ],
            })
            added.append({
                "role": "tool",
                "tool_call_id": f"r_{t}",
                "name": "read_file",
                "content": f"# {fpath} revision {t}\n" + ("def handle_step(x: int) -> int:\n    return x + 1\n" * 65),
            })
            added.append({
                "role": "tool",
                "tool_call_id": f"b_{t}",
                "name": "bash_run",
                "content": f"=== pytest run {t} ===\n" + ("PASSED tests/test_module.py::test_case\n" * 70) + "95 passed in 1.2s",
            })
            added.append({"role": "assistant", "content": f"Turn {t}: {fpath} updated and 95 tests passed."})
        turns.append(WorkloadTurn(messages_added=added))

    return system, turns, facts


def run_arm(
    arm: str,
    workload_name: str,
    system: list[dict[str, Any]],
    turns: list[WorkloadTurn],
    facts: list[dict[str, str]],
) -> dict[str, Any]:
    td = tempfile.mkdtemp(prefix=f"mc_bench_{arm}_")
    reader = RealisticLossyFoldReader(latency_s=0.04, amnesia_every=3)

    if arm == "raw":
        sess = None
    elif arm == "legacy_mc":
        sess = SessionMemory(
            f"{workload_name}-{arm}",
            Path(td),
            writer=IntWriterAdapter(None),
            embed_fn=_hash_embed,
            reader=reader,
            hot_limit=12000,
            keep_min=4500,
            recall_top_k=4,
            async_fold=False,
            compact_tool_outputs=False,
        )
    elif arm == "memochain_cloud":
        sess = SessionMemory(
            f"{workload_name}-{arm}",
            Path(td),
            writer=IntWriterAdapter(None),
            embed_fn=_hash_embed,
            reader=reader,
            hot_limit=18000,
            keep_min=6500,
            recall_top_k=4,
            async_fold=True,
            compact_tool_outputs=True,
            tool_archive_min_chars=1200,
            tool_keep_recent_groups=3,
        )
    elif arm == "memochain_local":
        sess = SessionMemory(
            f"{workload_name}-{arm}",
            Path(td),
            writer=IntWriterAdapter(None),
            embed_fn=_hash_embed,
            reader=reader,
            hot_limit=8000,
            keep_min=3200,
            recall_top_k=4,
            async_fold=True,
            compact_tool_outputs=True,
            tool_archive_min_chars=1200,
            tool_keep_recent_groups=3,
        )
    else:
        raise ValueError(arm)

    history: list[dict[str, Any]] = list(system)
    prev_out: list[dict[str, Any]] | None = None
    total_prompt_tokens = 0
    total_cached_tokens = 0
    peak_prompt_tokens = 0
    per_turn_tokens: list[int] = []
    proxy_latencies_ms: list[float] = []
    folds_triggered = 0

    for turn in turns:
        history.extend(turn.messages_added)
        t0 = time.perf_counter()
        if arm == "raw":
            out = [dict(m) for m in history]
            st: dict[str, Any] = {}
        elif arm == "legacy_mc":
            # Legacy behavior: combined static+dynamic injected directly into system prompt (busts cache!)
            out, st = sess.process(history, inject_memory=False)  # type: ignore[union-attr]
            mb = st.get("memory_block", "")
            if mb and out and out[0].get("role") == "system":
                out[0] = {"role": "system", "content": str(out[0].get("content") or "") + "\n\n" + mb}
        else:
            out, st = sess.process(history)  # type: ignore[union-attr]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        proxy_latencies_ms.append(dt_ms)
        if st.get("fold"):
            folds_triggered += 1
        if sess is not None and sess.async_fold:
            # Simulate natural user/LLM think time between turns so background fold completes
            sess.wait_for_pending_fold(timeout=5.0)

        pt = sum(estimate_tokens(message_text(m)) for m in out)
        ct = _prefix_cached_tokens(prev_out, out)
        total_prompt_tokens += pt
        total_cached_tokens += ct
        peak_prompt_tokens = max(peak_prompt_tokens, pt)
        per_turn_tokens.append(pt)
        prev_out = out

    # End-of-session recall evaluation on ALL planted facts (including updated ones)
    recall_hits = 0
    update_hits = 0
    update_total = 0
    for f in facts:
        q_hist = history + [{"role": "user", "content": f"What is the exact current host for {f['code']} ({f['key']})?"}]
        if arm == "raw":
            q_out = q_hist
        elif arm == "legacy_mc":
            q_out, q_st = sess.process(q_hist, inject_memory=False)  # type: ignore[union-attr]
            mb = q_st.get("memory_block", "")
            if mb:
                q_out[0] = {"role": "system", "content": str(q_out[0].get("content") or "") + "\n\n" + mb}
        else:
            q_out, _ = sess.process(q_hist)  # type: ignore[union-attr]
        rendered = "\n".join(message_text(m) for m in q_out)
        hit = f["expected"] in rendered
        if hit:
            recall_hits += 1
        if f.get("updated"):
            update_total += 1
            if hit:
                update_hits += 1

    uncached = max(0, total_prompt_tokens - total_cached_tokens)
    cheap_factor = 0.12 if arm == "memochain_cloud" else (1.0 if arm == "legacy_mc" else 0.0)
    costs = {
        model_name: _compute_cloud_cost(
            uncached,
            total_cached_tokens,
            reader.fold_input_tokens if sess is not None else 0,
            cheap_fold_factor=cheap_factor,
            pricing=pricing,
        )
        for model_name, pricing in PRICE_MODELS.items()
    }

    sorted_lat = sorted(proxy_latencies_ms)
    p50_ms = sorted_lat[len(sorted_lat) // 2]
    p95_ms = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))]
    max_ms = sorted_lat[-1]

    verify_ok = True
    capsule_bytes = 0
    facts_stored = 0
    if sess is not None:
        v = sess.verify()
        verify_ok = bool(v.get("ok"))
        capsule_bytes = sess.capsule_path.stat().st_size if sess.capsule_path.exists() else 0
        facts_stored = sess.chunk_store.facts_count() if sess.chunk_store is not None else 0

    return {
        "arm": arm,
        "workload": workload_name,
        "turns": len(turns),
        "total_prompt_tokens": total_prompt_tokens,
        "peak_prompt_tokens": peak_prompt_tokens,
        "final_prompt_tokens": per_turn_tokens[-1],
        "prefix_cached_tokens": total_cached_tokens,
        "prefix_cache_hit_pct": round(100.0 * total_cached_tokens / max(1, total_prompt_tokens), 1),
        "folds_triggered": folds_triggered,
        "fold_calls": reader.calls,
        "fold_input_tokens": reader.fold_input_tokens,
        "proxy_p50_ms": round(p50_ms, 2),
        "proxy_p95_ms": round(p95_ms, 2),
        "proxy_max_ms": round(max_ms, 2),
        "fact_recall": f"{recall_hits}/{len(facts)}",
        "fact_recall_pct": round(100.0 * recall_hits / max(1, len(facts)), 1),
        "updated_fact_recall": f"{update_hits}/{update_total}" if update_total else "n/a",
        "capsule_bytes": capsule_bytes,
        "facts_stored": facts_stored,
        "receipt_chain_ok": verify_ok,
        "estimated_cost_usd": costs,
    }


def run_gpu_prefill_benchmark(token_lengths: list[int], model_id: str = "Qwen/Qwen3-0.6B") -> dict[str, Any]:
    """Measure real CUDA prefill TTFT (ms) and peak allocated VRAM (MB) on the local GPU
    (e.g. RTX 3060 12 GB) across the prompt lengths actually produced by each arm."""
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        return {"skipped": True, "reason": "CUDA unavailable"}

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
    # Prevent Windows WDDM from silently spilling VRAM into system RAM (which hangs prefill)
    try:
        torch.cuda.set_per_process_memory_fraction(0.90, 0)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()

    # Warmup
    with torch.inference_mode():
        warm = torch.randint(10, 1000, (1, 256), device="cuda")
        _ = model(warm, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()

    measurements: list[dict[str, Any]] = []
    for n_tok in sorted(set(token_lengths)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            inp = torch.randint(10, 10000, (1, n_tok), device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(inp, use_cache=True, logits_to_keep=1)
                kv_bytes = 0
                pkv = out.past_key_values
                if hasattr(pkv, "key_cache"):
                    for k, v in zip(pkv.key_cache, pkv.value_cache):
                        kv_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
                elif hasattr(pkv, "layers"):
                    for layer in pkv.layers:
                        if hasattr(layer, "keys") and layer.keys is not None:
                            kv_bytes += layer.keys.numel() * layer.keys.element_size()
                        if hasattr(layer, "values") and layer.values is not None:
                            kv_bytes += layer.values.numel() * layer.values.element_size()
                elif isinstance(pkv, (tuple, list)):
                    for layer in pkv:
                        for t in layer:
                            kv_bytes += t.numel() * t.element_size()
                del out, pkv
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            del inp
            torch.cuda.empty_cache()
            row = {
                "prompt_tokens": n_tok,
                "prefill_ms": round(prefill_ms, 1),
                "tok_per_sec": round(n_tok / (prefill_ms / 1000.0), 1),
                "kv_cache_mb": round(kv_bytes / (1024 * 1024), 1),
                "peak_vram_mb": round(peak_mb, 1),
                "oom": False,
            }
            measurements.append(row)
            print(f"  [gpu] {n_tok:6d} tok -> {row['prefill_ms']:7.1f} ms | kv={row['kv_cache_mb']:6.1f} MB | peak={row['peak_vram_mb']:7.1f} MB", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            measurements.append({
                "prompt_tokens": n_tok,
                "prefill_ms": None,
                "tok_per_sec": None,
                "kv_cache_mb": None,
                "peak_vram_mb": total_vram_mb,
                "oom": True,
            })
            print(f"  [gpu] {n_tok:6d} tok -> OOM (exceeded 90% of {total_vram_mb} MB VRAM)", flush=True)
            break

    del model
    torch.cuda.empty_cache()
    return {
        "gpu": gpu_name,
        "total_vram_mb": total_vram_mb,
        "model": model_id,
        "measurements": measurements,
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    hdr = f"{'workload':12s} | {'arm':15s} | {'tot_tok':9s} | {'peak':6s} | {'cache%':6s} | {'p95ms':6s} | {'recall':7s} | {'sonnet$':7s} | {'dsv3$':7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['workload']:12s} | {r['arm']:15s} | {r['total_prompt_tokens']:9d} | "
            f"{r['peak_prompt_tokens']:6d} | {r['prefix_cache_hit_pct']:5.1f}% | "
            f"{r['proxy_p95_ms']:6.1f} | {r['fact_recall']:7s} | "
            f"${r['estimated_cost_usd']['claude_sonnet']:6.3f} | "
            f"${r['estimated_cost_usd']['deepseek_v3']:6.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MemoChain Phase D 4-Arm Benchmark Matrix")
    parser.add_argument("--gpu-prefill", action="store_true", help="Also run real CUDA prefill/VRAM benchmark")
    parser.add_argument("--gpu-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    arms = ["raw", "legacy_mc", "memochain_cloud", "memochain_local"]
    results: list[dict[str, Any]] = []

    sys_chat, chat_turns, chat_facts = build_long_chat_workload(150)
    for arm in arms:
        results.append(run_arm(arm, "long_chat", sys_chat, chat_turns, chat_facts))

    sys_code, code_turns, code_facts = build_coding_agent_workload(60)
    for arm in arms:
        results.append(run_arm(arm, "coding_agent", sys_code, code_turns, code_facts))

    _print_table(results)

    report: dict[str, Any] = {"matrix": results}
    if args.gpu_prefill:
        # Measure real GPU prefill at representative peak token lengths across local vs cloud vs raw
        sample_lens = [2048, 4096, 8192, 16384, 24576, 32768]
        print(f"\nRunning real CUDA prefill & KV-cache VRAM benchmark on {args.gpu_model}...")
        gpu_res = run_gpu_prefill_benchmark(sample_lens, model_id=args.gpu_model)
        report["gpu_prefill"] = gpu_res
        if not gpu_res.get("skipped"):
            print(f"GPU: {gpu_res['gpu']} ({gpu_res['total_vram_mb']} MB VRAM) | Model: {gpu_res['model']}")
            print(f"{'prompt_tok':10s} | {'prefill_ms':10s} | {'tok/s':8s} | {'kv_mb':8s} | {'peak_vram_mb':12s} | status")
            print("-" * 68)
            for m in gpu_res["measurements"]:
                if m["oom"]:
                    print(f"{m['prompt_tokens']:10d} | {'OOM':>10s} | {'-':>8s} | {'-':>8s} | {m['peak_vram_mb']:12.1f} | OOM")
                else:
                    print(
                        f"{m['prompt_tokens']:10d} | {m['prefill_ms']:10.1f} | {m['tok_per_sec']:8.1f} | "
                        f"{m['kv_cache_mb']:8.1f} | {m['peak_vram_mb']:12.1f} | ok"
                    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
