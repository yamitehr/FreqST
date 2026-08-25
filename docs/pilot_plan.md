# Pilot plan — FreqST vs GrayST on CATER (proposed, not yet run)

Not a stage — a forward-looking design doc for the first real-data experiment. Back to [CLAUDE.md](../CLAUDE.md).

## Purpose

Bridge between the synthetic CPU prototype (Stages 1–4b) and the proposal's full CATER
study. Answer one specific question, cheaply:

> Under a fair frame-budget comparison, does FreqST beat GrayST on CATER's
> camera-motion split when both are plugged into the same pretrained 2D backbone?

Positive → green light for the full CATER + TSN/TSM/MVFNet study.
Negative → the synthetic-blob pan-robustness result did not transfer, and the paper's
scope narrows before we spend the GPU budget.

## Dataset — CATER task 2 (cameramotion split), subset

**Why CATER, not SSv2/Kinetics.** It's the only common benchmark that ships an explicit
static-camera vs moving-camera split of the same task — exactly the setting FreqST is
designed for. SSv2 has no such split so it can't test the hypothesis.

**Why task 2, not task 1.** Task 2 is the proposal's target (301-way multi-label
compositional actions, mAP). We use it directly so the pilot rehearses the deliverable's
evaluation.

**Subset for pilot** (full CATER is ~3500 train / ~2000 test; too heavy for a pilot):

| set | clips | notes |
|---|---|---|
| Train | ~1000 | balanced coverage over the 301 compositional classes |
| Val | ~200 | held out, used for observation only |
| Test | ~500 | frozen, evaluated once |

Same subset composition for the cameramotion split; a static-camera subset run comes
later once the cameramotion result is in.

## Protocol — match Kim et al. exactly (verified from `third_party/channel_sampling/`)

**Sampling.** CATER uses DENSE sampling in their code, not sparse TSN.
`input_frame_length = 32` (CNN inputs per clip); dense recipe: 32 frames × stride 8.

**Optimizer / schedule.** SGD, momentum 0.9, weight decay 5e-4. Batch 24, LR 0.0024
(their 2×RTX-3090 recipe). Schedule: `ReduceLROnPlateauMultiple`, factor 0.1, patience
10 on val loss; early-stop after 20 plateau epochs. AMP FP16.

**Augmentation.** Train: short-side random jitter `[224, 336]`, random 224 crop,
horizontal flip. Test: **1 clip × 1 center crop** (NOT the 10-crop protocol used for
SSv2 — CATER's `val_num_spatial_crops = 1`, `val_num_ensemble_views = 1`).

**Metric.** mAP (multi-label, sigmoid + BCE). Report **last-epoch val mAP**, no
best-epoch selection, no early stopping on metric — matches Kim et al. and matches
Stage 3's own reporting rule.

**Reference numbers to beat** (Kim et al. Table 1, cameramotion / mAP):

| backbone | GrayST cameramotion mAP |
|---|---|
| TSN | 61.9 |
| TRN | 57.6 |
| TSM | 74.7 |
| MVFNet | 67.8 |

GrayST is SOTA on cameramotion across all four backbones. TSM is best; TSN is cleanest
for pilot (see backbone choice below).

## Methods to compare — matched frame budget

The key design decision. FreqST's default recipe would ingest 32 × 8 = 256 raw frames
per clip vs GrayST's 32 × 3 = 96 — a 2.7× temporal-coverage advantage that makes any
positive FreqST result uninterpretable ("did the DCT win, or did FreqST just see more
of the video?").

**Fix: cap FreqST at GrayST's 96-frame budget.** FreqST then produces fewer CNN inputs
(N=16 instead of 32), which is still a *handicap* — less TSN consensus at test time.
If FreqST still wins under this handicap, that's a stronger result than a matched-N run.

**Why W=6 (not W=8).** With `N × W = 96` fixed, picking W=6 gives N=16 anchors;
picking W=8 gives N=12. N=16 is a milder TSN-consensus handicap (2× fewer votes than
GrayST vs 2.7× for N=12) at negligible cost to the transform: a length-6 DCT still
splits DC from 2 low AC bins (we keep 3 of 6 bins = 50% of the spectrum, vs 3 of 8
= 37.5% for W=8). CATER's object motion (slide / contain / pick-place over multiple
seconds at 24 fps) sits well within the low-frequency band either way; the Stage 1
"speed vs kept-AC-energy" curve narrows slightly for W=6 but doesn't cross into
CATER-relevant motion speeds.

Three methods, all at 96 raw frames per clip:

| method | frames/clip | CNN inputs/clip | window per anchor | tests |
|---|---|---|---|---|
| **GrayST-vanilla** (baseline reference) | 96 | 32 | 3 grays spread across the whole clip | published baseline |
| **GrayST-W6** (representation control) | 96 | 32 | 3 grays sampled within a 6-frame window around anchor | isolates *representation* — same window, same budget, DCT vs raw pixels |
| **FreqST-N16** (ours, matched budget) | 96 | **16** | 6 consecutive → DCT → DC + 2 low AC | matched raw-frame budget, at a 2× CNN-input handicap |

The GrayST-W6 control is the important one. It matches FreqST's *window* (6-frame
region per anchor) AND its *frame budget* (96), differing only in representation:
GrayST-W6 keeps raw grayscale pixels; FreqST does the 1D DCT along time. If
FreqST beats GrayST-W6, the DCT itself is doing real work — which is exactly the
claim the report defends synthetically (see [stage4b](stage4b_grayst_ensemble.md)) but
has not been tested on real video.

**Deferred (add only if the above is positive):**
- FreqST-N32 at 192 frames (W=6) or 256 frames (W=8) — "extra frames help too, on top of the win."
- TCPlus2 and TC Reordering baselines — Kim et al. also compared against these.
- Static-camera CATER split — repeat the whole pilot on it for the second half of
  the paper's table.

## Backbone choice

**TSN-ResNet-50 (ImageNet-pretrained) for the primary comparison.** TSM's temporal-shift
module already encodes temporal information in-network; combining it with FreqST's
already-encoded temporal representation risks double-counting or destructive
interference. TSN is the cleaner isolation of the input-stub effect. TSM/MVFNet come
later as secondary comparisons if the TSN result is positive.

## Reporting

- **Single seed per method** for the pilot — cheapest scoping that still answers the
  branching decision. Multi-seed verification is deferred to the full-CATER stage,
  where a 3-way seed ablation matters for the paper.
- Paired significance test on per-clip predictions (McNemar or bootstrap CI on the
  500-clip test set). This is the only within-pilot noise signal we get; mAP
  differences of ~1-2 points on 500 clips are within resampling noise, so a p-value
  is worth reporting alongside the raw number.
- Also record wall-clock training + inference time per method (DCT compute cost).

## Total pilot cost

- 3 methods × 1 seed = **3 training runs** on the cameramotion subset.
- Estimated ~1-2 GPU-hours per run on Athena (subset scale, ResNet-50).
- **~3-6 GPU-hours total**, comfortably one workday on a single GPU.

## Success criteria — decide next step from these outcomes

| outcome | interpretation | next step |
|---|---|---|
| FreqST-N16 > GrayST-W6 > GrayST-vanilla | Best case: DCT wins on representation alone at a CNN-input handicap. | Run FreqST-N32 to confirm the extra frames also help, then commit to full CATER. |
| FreqST-N16 > GrayST-vanilla, ≈ GrayST-W6 | DCT is not doing extra work over raw pixels at matched window — the "advantage" is just window size. | Narrow the paper's claim, position FreqST as a way to get wider windows into a 3-channel input. Skip full CATER unless a stronger reformulation surfaces. |
| FreqST-N16 ≈ GrayST-vanilla | Frame budget was the whole story on the synthetic result. | Kill the pan-robustness claim. Investigate why synthetic result didn't transfer. |
| FreqST-N16 < GrayST-vanilla | FreqST is actively worse on real video. | Diagnose (implementation? DCT window mismatched to CATER's motion scale?) before writing anything. |

## Setup work not yet started

- Download CATER task 2 (~35GB) to Athena.
- Extract to per-frame JPEGs following Kim et al.'s pipeline (their
  `submodules/video_datasets_api/tools/` scripts — the two nested submodules are
  currently NOT initialized in our vendored copy of `third_party/channel_sampling/`
  because the sandbox blocked recursive submodule init).
- Wire our `freqst()` transform (from `transforms.py`) into PyVideoAI as a new
  sampling mode alongside `RGB / TC / GreyST / TCPlus2`.
- Implement the GrayST-W6 sampling variant (~20-line edit to the sparse-sample
  dataset's frame-index function).
- Add a paired-significance-test utility for per-clip mAP.

## Related docs

- [Stage 3](stage3.md) — synthetic learnability result FreqST is trying to reproduce on real data.
- [Stage 3 verify](stage3_verify.md) — the multi-seed + span-confound protocol this pilot mirrors.
- [Stage 4b](stage4b_grayst_ensemble.md) — the synthetic-side control that says the DCT is doing work beyond frame count. GrayST-W6 in this pilot is its real-data analog.
