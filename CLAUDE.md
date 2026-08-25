# FreqST

MSc Computer Vision course project. 4 authors: Yamit Ehrlich, Sophie Feldman (you), Ron Libman, Rona Lavi.

## Idea

FreqST: per-pixel 1D DCT along the time axis over a window of W consecutive grayscale
frames, keep the DC coefficient + 2 low-frequency AC coefficients → one 3-channel,
image-shaped tensor. Feed straight into an unmodified 2D-CNN backbone (no architecture
change, no extra params). Channel 0 = average appearance over the window, channels 1-2 =
slow/fast motion energy.

Positioned against Kim et al. (BMVC 2022) "Capturing Temporal Information in a Single
Frame" — GrayST (stack 3 consecutive raw grayscale frames as channels) and TC Reordering
(stack 3 raw frames spread across the window). Both are raw-pixel stacks; their gains
collapse under camera motion (can't tell "object moved" from "camera panned"). FreqST's
bet: an explicit frequency decomposition separates local motion from global pan better,
because it draws on a full W-frame window (e.g. 8) instead of 3 raw snapshots.

## Proposal vs. current code — the gap

**Proposal target:** train TSN, TSM, and MVFNet end-to-end on CATER (Girdhar & Ramanan,
ICLR 2020) task 2, static-camera split + camera-motion split, compare mAP against Kim et
al.'s published baselines. Needs GPU + ~35GB storage. Plan: reduced-schedule pilot first
(fewer epochs, data subset) on both splits before committing to the full study.

**Current code is a from-scratch CPU prototype, not that study.** No CATER, no TSN/TSM/
MVFNet, no pretrained backbone — just a synthetic Gaussian-blob dataset + a tiny 4-block
CNN trained from scratch. It exists to sanity-check the transform and the camera-motion
hypothesis cheaply before spending GPU budget on the real thing. Read `results/REPORT.md`
in full before touching anything — it's dense and the reasoning matters (aliasing/
undersampling tradeoffs, why Stage 2 as originally scoped was reframed, fairness/ablation
controls). Summary below, but the report has the "why" that a summary loses.

**Remote GPU available:** Athena, via ssh. Use it once ready to move past the synthetic
CPU prototype toward CATER-scale training (see Next Steps). No CATER data or TSN/TSM/
MVFNet codebase has been fetched anywhere yet — that setup work hasn't started.

## Layout

```
transforms.py                              grayst / tc_reordering / chunk_avg / freqst,
                                            apply_windowed() for sliding-window application
models.py                                   TinyCNN — 3 conv blocks + GAP + linear head,
                                            same architecture used identically across every
                                            preprocessing variant (only input channels change)
synthetic_data.py                           Gaussian-blob clip generator (generate_labeled_clip)
                                            + generate_pan_over_real_background (real vtest.avi
                                            frame as texture, synthetic pan + local blob on top)
data/vtest.avi                              OpenCV sample clip (Apache-2.0), real pedestrian
                                            footage, static camera — used by stage1b
experiments/stage1.py                       transform sanity check, synthetic, no training
                                            → docs/stage1.md
experiments/stage1b.py                      real-video qualitative check, no training
                                            → docs/stage1b.md
experiments/stage3.py                       learnability: tiny CNN, easy+hard regimes,
                                            4 methods (rgb/grayst/tc_reordering/freqst) x
                                            2 datasets (static/pan) → docs/stage3.md
experiments/stage3_verify.py                multi-seed (x3) + temporal-average control,
                                            re-verifies stage3's hard-regime headline result
                                            → docs/stage3_verify.md
experiments/stage4_window9.py               fairness check: every method gets the same T=9
                                            window (stage3 gives GrayST/TC only 3 of 8 frames)
                                            → docs/stage4_window9.md
experiments/stage4b_grayst_ensemble.py      GrayST as a 3-glimpse train-time-augmented,
                                            test-time-averaged ensemble — tests whether
                                            GrayST just needed to see more of the window
                                            → docs/stage4b_grayst_ensemble.md
run_all.sh                                   bash, runs stage1 → stage1b → stage3 →
                                            stage3_verify only
results/REPORT.md                            full writeup — READ THIS FIRST
docs/stage*.md                               per-stage deep dives (what/why/findings),
                                            each links back here — see Stage docs below
```

**`run_all.sh` does NOT run stage4 or stage4b.** Run them manually:

```bash
python -m experiments.stage4_window9
python -m experiments.stage4b_grayst_ensemble
```

**stage4/stage4b results are not yet folded into `results/REPORT.md`.** They only exist
as `results/stage4_window9/metrics.txt` and `results/stage4b_grayst_ensemble/metrics.txt`.
Read those directly, or fold them into the report, before treating REPORT.md as the full
picture.

`run_all.sh` assumes a bash-style venv (`venv/bin/activate`) — works as-is on Linux/macOS,
and on Windows under Git Bash/WSL, but not in a plain PowerShell/cmd venv (that uses
`venv\Scripts\activate` instead). Safer cross-OS default: activate your venv however your
shell needs, then run modules directly from `FreqST/`, e.g. `python -m experiments.stage1`.

## Stage docs

Deep dive per stage — what each script does, why, exact numbers, output files:

- [docs/stage1.md](docs/stage1.md) — transform sanity check (no training)
- [docs/stage1b.md](docs/stage1b.md) — real-video qualitative check (no training)
- [docs/stage3.md](docs/stage3.md) — learnability check (headline result)
- [docs/stage3_verify.md](docs/stage3_verify.md) — multi-seed + span-confound controls
- [docs/stage4_window9.md](docs/stage4_window9.md) — same-window fairness check
- [docs/stage4b_grayst_ensemble.md](docs/stage4b_grayst_ensemble.md) — best-effort GrayST control

## Key results so far (hard regime: dim blob, small model, less data — forces methods apart)

| method | static (A) | +camera pan (B) |
|---|---|---|
| RGB (1 frame) | 0.26 | 0.22 (chance) |
| Temporal avg (8f, control) | 0.39 | 0.58 |
| GrayST (3 consecutive frames) | 0.99–1.00 | **0.31–0.32** (collapses) |
| TC Reordering (3 spread frames) | 1.00 | **0.31** (collapses) |
| GrayST 3-glimpse ensemble (uses all 9 frames, still raw pixels) | 1.00 | 0.64 |
| Chunk-avg (all frames, no DCT) | 1.00 | 0.64 |
| **FreqST (DCT)** | 1.00 | **0.875–0.90** |

Takeaways, condensed (full reasoning + caveats in REPORT.md):
- Stage 1: the DCT split behaves exactly as designed, but only within a speed range
  matched to window length W — faster motion leaks energy into coefficients FreqST
  doesn't keep. Real tuning knob for any real-data run.
- Stage 1b: on real video with a static camera, FreqST is clean (DC = sharp background,
  AC = isolated motion). Under a real-texture camera pan, it does **not** cleanly isolate
  the object at the pixel level — ties TC Reordering (~5% of motion energy on the actual
  object either way).
- Stage 3: despite that pixel-level finding, FreqST is dramatically more robust to pan at
  the *classification* level. Two mechanisms, separated via the temporal-average control:
  integrating the whole window (already recovers a lot) + frequency structure on top
  (recovers more, beyond what averaging alone gives).
- Stage 4/4b: ruled out "GrayST just needs more frames" as an alternative explanation —
  giving GrayST all 9 frames (via ensemble or chunk-averaging) only gets it to ~0.64,
  still well short of FreqST's ~0.88–0.90. It's the frequency decomposition specifically,
  not frame count.
- Everything above is fully synthetic (one blob, checkerboard background, rigid pan) or
  qualitative-only on one real clip. Zero evidence yet on real datasets, multiple objects,
  occlusion, parallax, non-rigid camera motion, or a pretrained backbone.

## Next steps

1. Fold `stage4_window9` and `stage4b_grayst_ensemble` results into `results/REPORT.md`.
2. Real-data pilot: FreqST vs. GrayST as the input stub to a pretrained 2D backbone on a
   subset of a real action dataset, ideally one with a camera-motion split. This is the
   recommended bridge before committing to the full CATER study — matches the proposal's
   own "reduced-schedule pilot first" plan. Athena (ssh) is available for this once GPU
   compute is needed.
3. Get CATER (task 2, 301 classes, static + camera-motion splits) and the TSN/TSM/MVFNet
   training codebase set up — this is the proposal's actual deliverable and hasn't been
   started. ~35GB storage needed.
4. Optional/exploratory ideas noted in REPORT.md but not yet implemented: a pan-speed
   sweep (quantify where the pixel-level separation degrades, not just static-vs-one-pan-
   speed) and motion-compensated FreqST (subtract global flow before the DCT, as a "fixed"
   version for the pan case).
