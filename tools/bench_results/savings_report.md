# MemoChain Savings Benchmark

Generated 2026-11-25T20:33:12+00:00 | model `inkling-2-mini` | 120 turns x ~1200 tokens per turn (identical conversation replayed against each endpoint).

Input tokens are as reported by the BACKEND `usage.prompt_tokens`, not estimated by the harness. TTFT is wall-clock time to the first streamed token.

## Measured results

| Endpoint | Total input tokens | Avg prompt/turn | Final-turn prompt | TTFT normal turns | TTFT fold turns | Folds |
| --- | --- | --- | --- | --- | --- | --- |
| direct (http://localhost:8000/v1) | 7,086,360 | 59,053 | 117,125 | 0.35 s (median) | - | 0 |
| proxy-8k (http://localhost:9002/v1) | 674,016 | 5,617 | 6,746 | 0.375 s (median) | 9.844 s | 38 |

## Savings vs direct (measured, this session length)

| Endpoint | Input token reduction | TTFT reduction (normal turns vs direct) |
| --- | --- | --- |
| proxy-8k | 90% | -7% |

Fold turns pay a one-request-per-~40k-folded-tokens summary call on top (measured here as the fold-turn TTFT); every other turn is a plain forward.


## Dollar cost of the measured input tokens (published price points)

| Endpoint | Claude Sonnet class ($3.00) | GPT/Gemini Pro class ($1.25) | DeepSeek class ($0.27) | Qwen open class ($0.20) |
| --- | --- | --- | --- | --- |
| direct | $21.26 | $8.86 | $1.91 | $1.42 |
| proxy-8k | $2.02 | $0.84 | $0.18 | $0.13 |

## Projection (same fold cadence, longer sessions)

Post-fold prompt size is constant, so savings grow with session length. Projection uses the measured average prompt of each endpoint:

| Session length | Direct input tokens | Best-proxy input tokens | Reduction |
| --- | --- | --- | --- |
| 200 turns (~240k tokens) | 24.1M | 1.1M | 95% |
| 600 turns (~720k tokens) | 216.4M | 3.4M | 98% |
| 2000 turns (~2400k tokens) | 2401.2M | 11.2M | 100% |

## Reproduce the real-dollar version (OpenRouter meters usage.cost)

```

python tools/benchmark_savings.py --turns 60 --tokens-per-turn 1200 \
  --model google/gemini-3.8-flash \
  --endpoint direct=https://openrouter.ai/api/v1 \
  --endpoint via-proxy=<your proxy URL>/v1 \
  --auth "Bearer sk-or-v1-YOURKEY" --out tools/bench_results_openrouter
```

The same BYOK key is sent to both sides; the proxy forwards it. The report then shows OpenRouter's own metered cost for direct vs proxied traffic.

## Raw data

- `direct` session id: `bench-direct-1795638792`, fold turns: none
- `proxy-8k` session id: `bench-proxy-8k-1795638865`, fold turns: 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117