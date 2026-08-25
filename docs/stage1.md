# Stage 1 — transform sanity check

Script: [`experiments/stage1.py`](../experiments/stage1.py) · Results: [`results/stage1/`](../results/stage1/) · No training. Back to [CLAUDE.md](../CLAUDE.md).

## What it does

Pure math check on synthetic data: does the DCT transform behave as designed, before
any classifier touches it.

- Single bright Gaussian blob (sigma=4, intensity=0.8) moves horizontally across a
  96×96 textured background at a controlled speed (px/frame). 8-frame window, `freqst()`
  keeping 3 coefficients.
- `get_window()` picks the blob's start position so it sits at frame-center at the
  window's midpoint regardless of speed — keeps the comparison fair across speeds
  (otherwise a fast blob would just exit frame early within the window).

Three checks, run in `main()`:

1. **`plot_channel_grid`** — visualize ch0/ch1/ch2 for speeds `[1, 3, 6]`.
   Confirms: ch0 = blurred/averaged frame (background sharp, blob smeared like a
   long exposure), ch1/ch2 = flat except where the blob moved (motion-only, isolated
   from the static background).
2. **`compare_transforms_grid`** — FreqST vs GrayST vs TC Reordering side by side at
   one speed (6 px/frame). GrayST/TC channels are all "another raw snapshot of the
   scene"; FreqST's AC channels show only the change.
3. **`quantify_activation_vs_speed`** — mean `|ch1|`, `|ch2|` in the motion region
   across speeds `0..8`, plus the fraction of *total* AC energy (all 8 DCT coefficients,
   not just the 3 kept) captured by ch1+ch2.

## Key finding

Activation rises with speed then falls — but `captured_fraction` shows this isn't
because motion energy disappears: it leaks into DCT coefficients beyond the 3 FreqST
keeps, as speed increases. This is an aliasing-like effect bounded by window length W:
an 8-frame window can only resolve motion up to a speed roughly matched to the window
length in the first few coefficients. Real tuning knob — window length trades off
temporal context per frame against which motion speeds show up in the kept coefficients.

## Output files

- `channels_grid.png` — ch0/ch1/ch2 across 3 speeds
- `transform_comparison.png` — FreqST vs GrayST vs TC Reordering
- `activation_vs_speed.png` — activation + captured-fraction curves
- `metrics.txt` — raw numbers, monotonicity checks (in-regime vs. full range)

## Why it matters

Proves the core transform claim (DC=appearance, AC=motion, isolated from static
background) holds in a controlled setting, and surfaces the W-vs-speed tradeoff that
matters for choosing W on real data later. Feeds directly into [Stage 1b](stage1b.md)
(does this survive real pixels?) and the window-size sweep in [Stage 3](stage3.md).
