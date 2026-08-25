# Stage 4 — same-window fairness check

Script: [`experiments/stage4_window9.py`](../experiments/stage4_window9.py) · Results: [`results/stage4_window9/`](../results/stage4_window9/) · Trains, hard regime, single seed. **Not in `run_all.sh` — run manually.** Back to [CLAUDE.md](../CLAUDE.md).

## What it does

[Stage 3](stage3.md) already feeds every method the same `WINDOW=8` span, but GrayST
and TC Reordering only ever read 3 of those 8 raw frames — so "same window" and "same
amount of frame content consumed" aren't the same thing. This script asks the literal
question: if every method is handed the exact same `T=9` window, what does each one do
with it?

| method | frames actually used |
|---|---|
| `rgb` | 1 (mid frame; window ignored entirely — appearance-only control) |
| `grayst` | last 3 of 9 (frames 6,7,8) — 6 discarded |
| `tc_reordering` | 3 of 9, spread (frames 0,4,8) — 6 discarded |
| `chunk_avg` | all 9, split into 3 contiguous groups, each averaged into one channel — nothing discarded, but no frequency decomposition |
| `tavg` | average of all 9 → 3 identical channels |
| `freqst` | DCT of all 9 → 3 coefficient channels (DC + 2 AC) |

Reuses the exact same underlying clips as [Stage 3](stage3.md)'s hard regime (identical
`build_dataset` seeds), just re-sliced to a 9-frame window — so any difference from
Stage 3's hard-regime numbers comes only from the window-size change, not different
data.

## Key finding

`results/stage4_window9/metrics.txt`:

| method | A (static) | B (pan) |
|---|---|---|
| RGB | 0.242 | 0.242 |
| GrayST (3 of 9) | 1.000 | 0.280 |
| TC Reordering (3 of 9) | 1.000 | 0.280 |
| Chunk-avg (9 of 9, no DCT) | 1.000 | 0.640 |
| Temporal avg (9 of 9) | 0.335 | 0.535 |
| **FreqST (9 of 9, DCT)** | **1.000** | **0.875** |

GrayST/TC are completely unaffected by having more frames available that they still
discard — they only ever look at the same 3. Chunk-avg (uses everything, still no
frequency decomposition) reaches 0.640 — better than raw stacking, but well short of
FreqST's 0.875. Frequency structure still earns its keep beyond "use the whole window."

## Output files

- `metrics.txt`, `window9_comparison.png`

## Why it matters

Rules out "GrayST/TC just weren't given enough of the window" as an explanation for
their collapse under pan — they had access to it and ignored it by construction. Sets
up [Stage 4b](stage4b_grayst_ensemble.md), which asks the harder question: what if
GrayST is *forced* to use the whole window, via ensembling instead of raw discarding?
