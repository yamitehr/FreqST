# Channel Sampling / GrayST + TC Reordering (paper summary)

**Title:** Capturing Temporal Information in a Single Frame: Channel Sampling Strategies for Action Recognition
**Authors:** Kiyoon Kim, Shreyank N Gowda, Oisin Mac Aodha, Laura Sevilla-Lara — University of Edinburgh (BMVC 2022)
**arXiv:** 2201.10394 · **Code:** https://github.com/kiyoon/channel_sampling
**Local:** `relevant papers/Kim-Channel-Sampling/channel_sampling.pdf` (+ `channel_sampling.md`, figures)

## Why it matters for FreqST
**This is the paper FreqST is positioned directly against** — GrayST and TC Reordering are the exact baselines FreqST benchmarks its DCT preprocessing against. Same core move (repurpose the 3 input channels of an unmodified 2D CNN to carry temporal info, no arch change, no extra FLOPs), but they use **raw-pixel stacks** where FreqST uses a **frequency (DCT) decomposition** over a longer window.

## The two strategies (both: reorder channels, no backbone change, no extra compute, no retrain-from-scratch)
1. **TC Reordering (Time-Color).** For an output "frame", take ONE color channel (R) from 3 consecutive frames and stack those 3 into the channel dim; repeat per color. Encodes short-term frame-to-frame change. **TC+2** variant samples 2 extra frames to avoid duplicating the last frames → consistent formation across all frames, notable gain.
2. **GrayST (Grayscale Short-Term Stacking).** Convert frames to grayscale, stack 3 consecutive grayscale frames as the 3 channels of one "RGB" image. Trades color for temporal reach: to produce T output frames it samples 3×T inputs (e.g. 8-frame GrayST sees 24 frames). Motivated by grayscale costing only ~0.5% ImageNet accuracy.

## Results
Up to **+24%** over standard RGB channel input across TSN, TRN/MTRN, TSM, MVFNet — gains largest on temporally-demanding datasets: **CATER** and Something-Something V1/V2. Narrows the gap to heavy models (TDN, Video Swin) at no extra inference cost.

## FreqST's critique / differentiation (from project CLAUDE.md)
- Both GrayST and TC Reordering are **raw-pixel stacks over only 3 frames** → their gains **collapse under camera motion** (can't separate "object moved" from "camera panned"). In FreqST's synthetic hard regime: GrayST/TC drop to ~0.31 (chance) under pan vs FreqST ~0.88–0.90.
- FreqST's bet: an explicit **DCT over a full W-frame window (e.g. 8)** separates local motion from global pan far better than a 3-frame raw stack. FreqST stage4/4b explicitly ruled out "GrayST just needs more frames" (ensemble/chunk-avg over 9 frames only reaches ~0.64) → it's the frequency decomposition, not frame count.
- Same backbones (TSN/TSM/MVFNet) + same benchmark (CATER, see [[CATER]]) → head-to-head comparison is apples-to-apples.
