# MemoChain Savings Benchmark

Generated 2026-11-25T21:49:43+00:00 | model `deepseek/deepseek-v3.2` | 60 turns x ~1200 tokens per turn (identical conversation replayed against each endpoint).

Input tokens are as reported by the BACKEND `usage.prompt_tokens`, not estimated by the harness. TTFT is wall-clock time to the first streamed token.

## Measured results

| Endpoint | Total input tokens | Avg prompt/turn | Final-turn prompt | TTFT normal turns | TTFT fold turns | Folds |
| --- | --- | --- | --- | --- | --- | --- |
| direct (https://openrouter.ai/api/v1) | 919,770 | 15,330 | 30,168 | 0.721 s (median) | - | 0 |
| via-proxy (<local-proxy>/v1) | 245,181 | 4,086 | 4,852 | 1.058 s (median) | 4.373 s | 18 |

## Savings vs direct (measured, this session length)

| Endpoint | Input token reduction | TTFT reduction (normal turns vs direct) |
| --- | --- | --- |
| via-proxy | 73% | -47% |

Fold turns pay a one-request-per-~40k-folded-tokens summary call on top (measured here as the fold-turn TTFT); every other turn is a plain forward.


## Dollar cost of the measured input tokens (published price points)

| Endpoint | Claude Sonnet class ($3.00) | GPT/Gemini Pro class ($1.25) | DeepSeek class ($0.27) | Qwen open class ($0.20) |
| --- | --- | --- | --- | --- |
| direct | $2.76 | $1.15 | $0.25 | $0.18 |
| via-proxy | $0.74 | $0.31 | $0.07 | $0.05 |

## Actual OpenRouter-billed cost (`usage.cost` returned by OpenRouter)

| Endpoint | Metered `usage.cost` | Reduction vs direct |
| --- | --- | --- |
| direct | **$0.1252** | baseline |
| via-proxy | **$0.0416** | **66.8%** |

Note: OpenRouter `usage.cost` on proxied chat turns measures the main chat calls; each fold turn also makes one compact hybrid summary call (~1.5k in / ~300 out) using the same BYOK key.


## Projection (same fold cadence, longer sessions)

Post-fold prompt size is constant, so savings grow with session length. Projection uses the measured average prompt of each endpoint:

| Session length | Direct input tokens | Best-proxy input tokens | Reduction |
| --- | --- | --- | --- |
| 200 turns (~240k tokens) | 24.1M | 0.8M | 97% |
| 600 turns (~720k tokens) | 216.4M | 2.5M | 99% |
| 2000 turns (~2400k tokens) | 2401.2M | 8.2M | 100% |

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

- `direct` session id: `bench-direct-1795643383`, fold turns: none
- `via-proxy` session id: `bench-via-proxy-1795643442`, fold turns: 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57