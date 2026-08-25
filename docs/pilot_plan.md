# Pilot plan — FreqST on CATER (proposed, not yet run)

Not a stage — a forward-looking design doc for the first real-data experiment. Back to [CLAUDE.md](../CLAUDE.md).

## Purpose

Bridge between the synthetic CPU prototype (Stages 1–4b) and the proposal's full CATER
study. Answer one specific question:

> With a raw-frame budget matched to GrayST, does FreqST beat Kim et al.'s published
> GrayST numbers on CATER's camera-motion split, using the same pretrained 2D backbone
> and the same training protocol?

Positive → green light for the full CATER + TSN/TSM/MVFNet study.
Negative → the synthetic-blob pan-robustness result did not transfer, and the paper's
scope narrows before we spend more GPU budget.

## Dataset — CATER task 2 (cameramotion split), full

**Why CATER, not SSv2/Kinetics.** It's the only common benchmark that ships an explicit
static-camera vs moving-camera split of the same task — exactly the setting FreqST is
designed for. SSv2 has no such split so it can't test the hypothesis.

**Why task 2, not task 1.** Task 2 is the proposal's target (301-way multi-label
compositional actions, mAP). We use it directly so the pilot rehearses the deliverable's
evaluation.

**Full CATER, not a subset.** Kim et al.'s published GrayST numbers are trained on full
CATER (**3849 train / 1651 val**, verified from the extracted lists via
`scripts/download_cater.sh`). CATER ships no separate test split; the paper reports
last-epoch val mAP as its test metric, and we match. To compare our FreqST number
directly against theirs without a data-size caveat, we train FreqST on the same full
data. Storage: ~35 GB.

Static-camera split comes later once the cameramotion result is in.

## Protocol — match Kim et al. exactly (verified from `third_party/channel_sampling/`)

Every training-time choice matches Kim et al.'s CATER recipe, so any performance
difference is attributable to the input stub (FreqST vs GrayST), not to schedule or
augmentation drift.

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

**Baseline reference numbers** (Kim et al. Table 1, cameramotion / mAP):

| backbone | GrayST cameramotion mAP |
|---|---|
| TSN | 61.9 |
| TRN | 57.6 |
| **TSM** | **74.7** |
| MVFNet | 67.8 |

GrayST is SOTA on cameramotion across all four backbones. TSM has the highest number;
TSN is the cleanest backbone for isolating the input-stub effect (see below).

## Method — FreqST-N16 at matched raw-frame budget

The one design decision. FreqST's natural recipe would ingest 32 × 8 = 256 raw frames
per clip vs GrayST's 32 × 3 = 96 — a 2.7× temporal-coverage advantage that makes any
positive FreqST result uninterpretable ("did the DCT win, or did FreqST just see more
of the video?").

**Fix: cap FreqST at GrayST's 96-frame budget.** With `N × W = 96` fixed:

| FreqST config | anchors N | window W | CNN inputs/clip | handicap vs GrayST |
|---|---|---|---|---|
| **N=16, W=6 (chosen)** | 16 | 6 | 16 | 2× fewer TSN votes |
| N=12, W=8 | 12 | 8 | 12 | 2.7× fewer TSN votes |
| N=32, W=3 (degenerate — don't) | 32 | 3 | 32 | none, but only 3 DCT bins total |

N=16, W=6 gives the mildest TSN-consensus handicap at negligible cost to the transform:
a length-6 DCT still splits DC from 2 low AC bins (we keep 3 of 6 bins = 50% of the
spectrum, vs 3 of 8 = 37.5% for W=8). CATER's slow object motion (slide / contain /
pick-place over multiple seconds at 24 fps) sits well within the low-frequency band
either way; Stage 1's "speed vs kept-AC-energy" curve narrows slightly for W=6 but
doesn't cross into CATER-relevant motion speeds.

**FreqST-N16 details:**
- 16 anchors per clip, TSN-sampled from 16 equal segments (~15 raw frames each at
  CATER's ~240-frame clip length).
- For each anchor, take 6 consecutive raw grayscale frames around it.
- 1D DCT along time per pixel over the 6-frame window; keep coefficients 0, 1, 2 as
  R, G, B → one 3-channel tensor per anchor.
- Feed 16 3-channel tensors to the backbone; average logits (TSN consensus).
- **Raw frames loaded per clip: 96** — matches GrayST exactly.

**Comparison: direct to paper's published numbers.** No GrayST re-training. Our
FreqST-N16 mAP against Kim et al.'s reported GrayST mAP for the same backbone.

## Backbone choice

**Primary: TSN-ResNet-50 (ImageNet-pretrained).** TSM's temporal-shift module already
encodes temporal information in-network; combining it with FreqST's already-encoded
temporal representation risks double-counting or destructive interference. TSN is the
cleaner isolation of the input-stub effect.

**Secondary: TSM-ResNet-50.** If TSN wins, run TSM too — TSM+GrayST is Kim et al.'s
highest reported number (74.7 mAP), so beating it is the strongest headline the paper
could carry. If FreqST underperforms TSM+GrayST despite winning at TSN, that's still
informative (it suggests FreqST's advantage is complementary to raw-pixel methods but
not to temporal-shift-augmented ones).

## Reporting

- **Single seed.** Multi-seed verification is deferred to the full-CATER stage once we
  know FreqST is worth the compute.
- Wall-clock training + inference time per run (DCT compute cost, to preempt reviewer
  questions).
- All hyperparameters logged; any deviation from Kim et al.'s recipe called out
  explicitly.

## Caveats to acknowledge in the paper

Skipping GrayST re-training saves compute but has two costs worth being upfront about:

1. **No implementation-verification baseline.** If our pipeline diverges from Kim
   et al.'s in some silent way (dataloader quirk, augmentation implementation), we
   can't detect it by comparing our-GrayST to their-GrayST. Mitigation: build the
   pipeline as a thin FreqST-adapter around Kim et al.'s own PyVideoAI code
   (`third_party/channel_sampling/`), so almost every non-transform code path is
   theirs, not ours.
2. **Windowing-vs-representation confound.** FreqST-N16 differs from GrayST on two
   axes at once: window shape (6 contiguous vs 3 spread) and representation
   (DCT vs raw pixels). If FreqST wins, we can't cleanly attribute the win to the
   DCT specifically. Stage 4b provides synthetic evidence that the DCT does work
   beyond frame count, but a real-data GrayST-W6 control would be the direct
   analog. Add as a follow-up experiment if the pilot result is promising and
   reviewers push back.

## Total pilot cost

- **1 training run per backbone** (start: TSN-R50; optional: TSM-R50, MVFNet-R50).
- Full-CATER scale — much heavier than a subset. Rough estimate: 12–36 GPU-hours per
  run on Athena depending on backbone, throughput, and how many epochs to plateau.
- **Total: 12–36 GPU-hours for TSN-only, up to ~100 for all four backbones.**

## Success criteria — decide next step from these outcomes

| outcome (TSN-R50, cameramotion mAP) | interpretation | next step |
|---|---|---|
| FreqST-N16 > GrayST 61.9 by a clear margin (≥3 pts) | Method transfers to real video at matched budget. | Run TSM and MVFNet backbones; then run static-camera split; then start writing. |
| FreqST-N16 ≈ GrayST 61.9 (within ±2 pts) | Ambiguous — could be method noise or a genuine tie. | Add multi-seed (3×) TSN runs before deciding. If still tied, run GrayST-W6 control to isolate windowing vs representation. |
| FreqST-N16 < GrayST 61.9 by a clear margin (≤−3 pts) | FreqST is worse on real video despite the synthetic story. | Diagnose before writing: is it a pipeline bug (does our GrayST-vanilla reproduce 61.9)? is W=6 too narrow for CATER motion? Is the DCT window mismatched to object-motion timescales? |

## Setup work not yet started

- Download CATER task 2 (~35 GB) to Athena.
- Extract to per-frame JPEGs following Kim et al.'s pipeline (their
  `submodules/video_datasets_api/tools/` scripts — the two nested submodules are
  currently NOT initialized in our vendored copy of `third_party/channel_sampling/`
  because the sandbox blocked recursive submodule init).
- Wire our `freqst()` transform (from `transforms.py`) into PyVideoAI as a new
  sampling mode alongside `RGB / TC / GreyST / TCPlus2`. Sample N=16 anchors × W=6
  consecutive frames per anchor.

## Related docs

- [Stage 3](stage3.md) — synthetic learnability result FreqST is trying to reproduce on real data.
- [Stage 3 verify](stage3_verify.md) — multi-seed + span-confound protocol; the format the full-CATER stage will follow.
- [Stage 4b](stage4b_grayst_ensemble.md) — the synthetic-side control that argues the DCT does work beyond frame count. Its real-data analog (GrayST-W6) is deferred here.
