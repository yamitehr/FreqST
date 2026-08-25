# CATER (paper summary)

**Title:** CATER: A Diagnostic Dataset for Compositional Actions & Temporal Reasoning
**Authors:** Rohit Girdhar, Deva Ramanan — CMU / Argo AI (ICLR 2020)
**arXiv:** 1910.04744 · **Project:** http://rohitgirdhar.github.io/CATER
**Local:** `relevant papers/CATER/cater.pdf` (+ `cater.md`, figures)

## Why it matters for FreqST
This is **the dataset FreqST's proposal targets** — the reduced-schedule pilot and full study both run on CATER Task 2, static-camera vs camera-motion splits. CATER's whole design purpose is FreqST's exact thesis: build a video benchmark where **single-frame / frame-averaging models fail and true spatiotemporal modeling is required**, and where **camera motion is a controllable, isolable variable**.

## What CATER is
Synthetic CLEVR-style tabletop videos (300 frames, 320×240, 24 FPS). Objects: CLEVR shapes (cube/sphere/cylinder) + cones + a special gold **"snitch"**. Four atomic actions: `rotate`, `pick-place`, `slide`, `contain` (cones can recursively contain objects → long-range state tracking). Free of object/scene bias (same object library every video), so temporal structure is the only signal that helps.

**Camera-motion split:** identical data distribution, but the camera jumps between predefined 3D coords every 30 frames (X,Y ∈ {−10,10}, Z ∈ {8,10,12}), always pointed at origin. This is the split where FreqST expects to win — the direct benchmark for its "separate local motion from global pan" claim.

## Three tasks
- **Task 1 — Atomic action recognition:** 14 classes (e.g. `slide(cone)`), multi-label, **mAP**. Debug task.
- **Task 2 — Compositional action recognition** (FreqST's target): pairs of atomic actions with temporal relation ∈ {before, during, after} → 14×14×3 = 588, dedup → **301 classes**, multi-label, **mAP**. Needs spatiotemporal composition reasoning; non-local blocks + LSTM help here (unlike Task 1).
- **Task 3 — Snitch localization** (flagship): predict snitch's final cell on a quantized 6×6 = 36-grid, single-label top-1/top-5 + mean L1. Occlusion + recursive containment force long-range reasoning.

Rendering: 5500 videos/task, 70:30 train:test (train further 80:20 for val). K=2 objects-with-actions for tasks 1–2, K=N for task 3.

## Key findings (baselines)
Even SOTA video models struggle. R3D / Non-Local / TSN / I3D: strong on static camera Task 1, **drop notably under camera motion** → the paper explicitly calls for "models agnostic to camera motion" (that is FreqST's pitch). Long-term reasoning: LSTM aggregation helps and optical flow alone is *not* effective — unlike prior datasets.

## Hooks into FreqST
- FreqST's `results/REPORT.md` claims robustness-to-pan at the classification level; CATER camera-motion Task 2 mAP is the real-data test of that synthetic-only result.
- Baselines to beat = the TSN/TSM/MVFNet numbers on this dataset (proposal deliverable).
- CATER is ~35 GB; needs Athena GPU — matches the "Next Steps" plan in the project CLAUDE.md.
