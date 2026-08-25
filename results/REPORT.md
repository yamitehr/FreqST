# FreqST prototype -- experiment notes

Quick disclaimer up front: everything here runs on a synthetic Gaussian-blob
video, not real footage. It can tell us whether the *transform* behaves the
way it's designed to behave in a controlled setting where we know the exact
ground-truth motion. It can't tell us anything about real backgrounds,
lighting, occlusion, multiple objects, or whether this actually helps a
real action-recognition backbone. Treat conclusions here as "the math checks
out," not "this will work in the wild."

## Stage 1 -- sanity check (no training)

**Setup:** a bright Gaussian blob moves horizontally across a textured
96x96 background at a controlled speed (px/frame). FreqST is applied to an
8-frame window (1D DCT along time per pixel, keep the first 3 coefficients).
Compared side-by-side against GrayST (stack 3 raw consecutive frames) and TC
Reordering (stack 3 raw frames spread across the window).

**Qualitative result (`channels_grid.png`, `transform_comparison.png`):**
Exactly as designed.
- Channel 0 (DC) looks like a blurred/averaged frame -- the checkerboard
  background is sharp (it doesn't move) and the blob is smeared out along
  its path, like a long-exposure photo.
- Channels 1 and 2 are flat/zero everywhere *except* where the blob moved --
  they isolate the motion instead of showing "yet another picture of the
  scene." Channel 1 shows a simple two-lobe (dipole) pattern; channel 2
  shows a finer three-lobe pattern, i.e. genuinely different (higher)
  spatial frequency content along the motion axis.
- Side-by-side with GrayST/TC Reordering: those two just show the blob
  itself (blurred slightly differently) in all three channels, because
  they're raw pixel values from different times. FreqST's AC channels
  instead show *only the change*, with the static background cancelled out.
  This is the qualitative payoff of doing the transform in frequency space.

**Quantitative result (`activation_vs_speed.png`):** mean |activation| in
the motion region rises with speed only up to a point, then falls off:

| speed (px/frame) | ch1 | ch2 | % of AC energy in ch1+ch2 |
|---|---|---|---|
| 0 | 0.000 | 0.000 | -- |
| 1 | 0.149 | 0.044 | 99% |
| 2 | 0.192 | 0.101 | 96% |
| 3 | 0.186 | 0.129 | 88% |
| 4 | 0.169 | 0.135 | 77% |
| 6 | 0.137 | 0.124 | 58% |
| 8 | 0.113 | 0.107 | 45% |

So it's monotonically increasing only over a modest speed range (roughly
0-2 px/frame for ch1, 0-4 px/frame for ch2 in this setup), not over the
whole tested range. We checked *why* using a diagnostic: the fraction of
total AC energy (summed across *all* 8 DCT coefficients, not just the 3
FreqST keeps) that lands in ch1+ch2 drops steadily as speed increases (99%
-> 45%). So the motion energy isn't disappearing -- it's shifting into
higher-frequency DCT bins that a 3-channel FreqST doesn't keep. This is an
aliasing/undersampling-style effect: an 8-frame window can only resolve
motion up to a speed roughly matched to the window length and blob size: go
faster than that and you need either more kept coefficients or a shorter
window to see it in the first few bins.

**Takeaway:** the core FreqST idea behaves exactly as intended (DC =
background/average, AC = motion, isolated from the static parts of the
scene) in a speed regime matched to the chosen window length. There's a
real tuning knob here -- window length W trades off "how much temporal
context per frame" against "what motion speeds show up in the first few
kept coefficients" -- worth keeping in mind for stages 2-3 and for real data
(faster/slower motion may need a different W).

---

## Stage 1b -- qualitative check on real video (no training)

Goal: see whether Stage 1's clean synthetic behaviour survives on real
footage, in the paper's Figure-3 layout (rows = method, columns = frames
across time).

**Data / license.**
- `data/vtest.avi`: OpenCV's own sample clip, downloaded from
  `github.com/opencv/opencv/raw/master/samples/data/vtest.avi` (~8 MB,
  768x576, 10 fps). It ships in the OpenCV repository, which is Apache-2.0
  licensed -- freely redistributable, no gating. (The task brief called it
  BSD; either way it's an unrestricted OpenCV sample.) It shows pedestrians
  walking across a fixed security-camera shot: clear moving foreground,
  static background. I used the most-active 9-frame stretch (auto-selected
  by max frame-to-frame difference; starts at frame 511) and took 16
  consecutive frames from there, converted to grayscale and resized to
  240x180 for display. Windowed transform: W=8, stride=1 -> 9 output columns.
- **Pan-over-real-background clip** (`generate_pan_over_real_background` in
  `synthetic_data.py`): I did NOT source a separately-licensed panning video.
  Instead I took a single static frame (frame 0) from vtest.avi as a large
  real-texture background, then panned a 240x180 viewport across it at a
  known **5.0 px/frame** (horizontal), and composited a synthetic Gaussian
  blob (sigma=7, intensity 0.7) moving locally at **3.0 px/frame** downward
  on top. So the clip has real background texture + exact-ground-truth global
  camera motion + exact-ground-truth local object motion, all at once.

**Figure style / faithfulness to the paper.** The comparison figures follow
Kim et al. Figure 3's layout (rows = method, columns = frames across time):
row 1 Original RGB (shown in true colour, as in the paper), row 2 Time-Color
Reordering, row 3 GrayST, and row 4 FreqST (ours, added). One caveat on
faithfulness: the paper's *Time-Color Reordering* keeps the original colour
channels and reorders which time each channel is taken from, so its row looks
noticeably more saturated than GrayST. Our whole pipeline is grayscale (so
that FreqST/GrayST/TC are compared grayscale-to-grayscale, and so Stage 3
trains on identical-modality inputs), which means our TC-Reordering row is a
grayscale-frame stack and ends up looking more similar to GrayST than the
paper's does. This only affects the *look* of the visual comparison, not the
quantitative claims below.

**Result on the static-camera real clip (`vtest_comparison.png`,
`vtest_channels_grid.png`): FreqST works as designed on real video.**
- DC channel (ch0) is a genuinely clean averaged frame: the static
  background (buildings, van, pavement, the pole) stays sharp, and the
  walking people are ghosted/smeared along their paths like a long exposure.
  Real lighting/shadows/compression noise did not muddy it noticeably --
  because those are static, they sit in DC cleanly.
- AC channels (ch1, ch2) are ~zero (flat gray) everywhere except on the
  moving pedestrians, where they light up with clear signed structure. They
  stay concentrated on the actually-moving people and do NOT pick up much
  from the static background texture, shadows, or compression. Quantitatively,
  80% of FreqST's AC energy sits in just ~4% of pixels (`metrics.txt`, [1]),
  essentially identical to TC Reordering (also ~4%) -- both concentrate on
  the movers when the camera is still.
- The pseudo-colour composite row (DC=R, AC1=G, AC2=B) shows a smooth
  appearance base with saturated colour only on the walkers, matching the
  spirit of the paper's Figure 3 (motion shows up as colour).

**Result on the pan clip (`panmotion_comparison.png`,
`panmotion_channels_grid.png`): the clean local/global separation DOES NOT
survive a global pan over real texture. This is the important finding.**
- DC channel is no longer a clean frame -- it becomes a directional
  motion-blur smear (horizontal streaks), because under a pan no pixel is
  static: different background texture slides through each pixel over the
  window, so the per-pixel time-average blurs along the pan direction.
- AC channels light up along *every* background edge that the pan drags
  across -- the pole, the sign, the diagonal pavement line, the bright lamp
  -- not just the local blob. In fact the local blob is faint relative to
  the pan-induced background activation.
- TC Reordering looks essentially the same failure mode (whole-frame colour
  fringing on every edge), which visually confirms the "global pan corrupts
  local pixel signals" story from the original paper -- but FreqST is
  corrupted the same way, not saved by its frequency split.
- Quantitatively the two methods are near-indistinguishable under pan:
  - spatial spread of motion energy jumps from ~4% of pixels (static) to
    ~12% for BOTH FreqST (0.118) and TC Reordering (0.117) (`metrics.txt` [1]);
  - fraction of motion energy actually landing on the local blob is only
    **~5% for both** (FreqST 0.053 vs TC 0.048) -- i.e. ~95% of the motion
    signal is the panning background for either method (`metrics.txt` [2]).
    FreqST is marginally better but nowhere near "isolating" the object.

**Why FreqST doesn't rescue the pan case (the intuition):** FreqST's DC/AC
split removes the *temporally constant* part per pixel. That cleanly isolates
motion only when the background is temporally constant at each pixel -- i.e.
a static camera. A global pan violates exactly that assumption: it injects
temporal variation into every textured pixel, so the AC channels fire
everywhere there's spatial contrast, and DC degrades into motion blur. The
DCT-along-time sees a pan the same way raw pixel-stacking does. So on
textured backgrounds, FreqST's advantage over TC Reordering under camera
motion appears to be roughly nil, at least for "clean separation."

**What this means for Stage 2 (recommendation).** Stage 2 as currently
scoped -- "does FreqST cleanly separate local vs global motion" -- is likely
to return a *negative* result on any realistic textured background, and this
Stage 1b evidence already points that way (blob-energy fraction ~5% for both
methods). I'd recommend one of:
  1. **Reframe Stage 2** around *spatial concentration / coherence* of
     activation rather than clean separation: e.g. does FreqST's motion
     activation at least stay more spatially coherent, or degrade more
     gracefully as pan speed grows, than TC's? (Our spread metric is a
     starting point; on this one clip they're tied, but a pan-speed sweep
     might reveal a gap.) Also worth testing motion-compensated FreqST
     (subtract global flow first) as the "fixed" version.
  2. **Keep Stage 2 but honestly expect/report a null result**, using it to
     document *when* the transform breaks (static-cam OK, moving-cam not),
     which is itself a useful, publishable-style caveat.
  3. **Narrow the claim**: position FreqST as a cheap static-camera temporal
     encoder (where Stage 1b shows it genuinely shines), and drop the
     camera-motion-robustness claim rather than trying to defend it.

My suggestion: do a quick pan-speed *sweep* version of Stage 2 (cheap, reuses
this generator) to confirm the null/att­enuation quantitatively, then move to
Stage 3 (learnability), which is the more decision-relevant experiment.

**Honest caveats.** This is qualitative/visual on a single real clip plus one
synthetic-pan construction -- no dataset-scale claims. vtest.avi's motion
statistics (upright walking pedestrians, fixed CCTV camera, mild compression)
are quite different from Something-Something-V2's close-up hand-object
interactions, so "FreqST looks clean here" is suggestive, not conclusive, for
the original hypothesis. The pan clip's background is a *single* real frame
panned rigidly, which is a clean idealization of camera motion (no parallax,
no rolling shutter, no independently-moving background objects) -- real camera
motion could be either easier or harder.

---

## Stage 3 -- learnability check: does FreqST help a CNN classify?

Task: 4-class blob motion-direction (left/right/up/down). One tiny CNN
(3 conv blocks + global avg pool + linear head), identical architecture and
hyperparameters across every run; only the input preprocessing changes.
Four preprocessings, all producing 3-channel inputs:
  - RGB (single trajectory-center frame, replicated -> appearance-only control)
  - GrayST (3 consecutive grayscale frames)
  - TC Reordering (3 frames spread across the window)
  - FreqST (DCT over the W=8 window, first 3 coefficients)

Two datasets: A = static camera; B = same task + a random global background
pan (speed comparable-to-exceeding the blob), a pure nuisance carrying no
label information (the Stage 1b camera-motion confound). Speed range for the
blob was U(1,3) px/frame -- chosen to sit mostly below the ~4 px/frame
energy-leakage breakpoint Stage 1 measured, so the "kept 3 coefficients" hold
most of the motion energy.

I ran it at two difficulty levels because the first (easy) version saturated:

**Easy regime (bright blob, low noise, bigger model/data).** All three
temporal methods hit 1.000 on BOTH datasets; RGB stayed at chance (0.25).
The task is trivially separable, so it can't rank the methods -- but it does
answer one question cleanly: when the task is easy, the global pan costs
*zero* accuracy for every method. The CNN just reads the compact moving blob
and ignores the diffuse background pan, even though (per Stage 1b) the raw
FreqST/TC channels are individually corrupted by that pan.

**Hard regime (dim sigma=2 blob, more noise, smaller model, less data, faster
pan U(3,6)).** This forced the methods apart and produced the key result,
which I then re-ran across 3 seeds + a control (see `verify_pan_robustness.png`,
`verify_metrics.txt`):

| method | A: static | B: + camera pan | drop A->B |
|---|---|---|---|
| RGB (1-frame) | 0.26 | 0.22 | ~0 (chance) |
| Temporal avg, 8 frames (control) | 0.39 | 0.58 | -0.19 |
| GrayST (3 frames) | 0.99 | **0.32** | 0.68 |
| TC Reordering (3 frames) | 1.00 | **0.31** | 0.69 |
| **FreqST (8 frames)** | 1.00 | **0.90** | **0.10** |

(mean over 3 seeds; FreqST's pan accuracy 0.903 +/- 0.013 -- very stable.)

**The verdict.**
- *Static camera:* FreqST matches GrayST/TC -- all essentially perfect. On
  this task, at least, FreqST is fully competitive with GrayST (the closest
  analog). No evidence it's worse.
- *Under camera pan:* FreqST is dramatically MORE robust. GrayST and TC
  **collapse to near chance** (~0.31) when a fast background pan is added,
  while FreqST holds at **0.90** -- a ~0.6 absolute accuracy gap, stable
  across seeds. The training curves (`hard/curves.png`) show it starkly: on
  Dataset B only FreqST's loss decreases and its val-accuracy climbs; the
  others flatline at chance.

**This partially reverses the worry from Stage 1b/2.** Stage 1b showed FreqST
does not *cleanly isolate* the moving object under pan (its AC channels light
up on panning background edges too). That's still true at the pixel level.
But it turns out not to matter downstream: FreqST is still far better for
*classification* under pan than the 3-frame methods. Two mechanisms, which
the temporal-average control lets us separate:
  1. **Temporal integration.** Plain 8-frame averaging (the "tavg" control)
     already recovers a lot: 0.58 on pan vs 0.31 for the 3-frame methods.
     Averaging a window washes the fast, random-direction background pan into
     a uniform blur, while the slower, consistent-direction blob leaves a
     readable directional smear. (Tellingly, tavg does *better* under pan
     (0.58) than static (0.39): under pan the background blurs away and the
     blob smear stands out more.) GrayST/TC get none of this -- 3 raw frames
     under fast pan are dominated by huge background displacement.
  2. **Frequency structure on top.** FreqST (0.90) clears the plain-average
     control (0.58) by a wide margin, so its AC channels carry real extra
     direction signal beyond mere averaging -- the blob's oriented,
     label-consistent AC signature survives even though diffuse background AC
     is also present.

### Stage 3 follow-up: two controls before treating the result as settled

**Check 1 -- is it "narrow time span" rather than "8-frame integration"?**
GrayST samples 3 *consecutive* frames (a narrow span); FreqST uses 8. Maybe
the 3-frame methods collapse only because their frames are bunched in time,
not because they use few frames. I added wide-span 3-frame controls (same
3 frames / 3 channels, just spread out) and re-ran all methods, 3 seeds,
same hard regime (`verify_metrics.txt`, `verify_pan_robustness.png`):

| method (all 3-channel) | A: static | B: pan (final) | drop |
|---|---|---|---|
| RGB (1 frame) | 0.26 | 0.22 | ~0 |
| GrayST (3f, consecutive, span 3) | 0.99 | 0.32 | 0.68 |
| GrayST-wide (3f, span 8: frames 0,4,7) | 1.00 | 0.31 | 0.69 |
| TC Reordering (3f, span 8: frames 0,4,7) | 1.00 | 0.31 | 0.69 |
| TC-wide (3f, span 12: frames 0,6,11) | 1.00 | 0.35 | 0.65 |
| Temporal avg (8f) | 0.39 | 0.58 | -0.19 |
| **FreqST (8f DCT)** | 1.00 | **0.90** | 0.10 |

**Verdict on Check 1: FreqST's advantage fully survives -- widening the span
does essentially nothing.** Stretching a 3-frame method's sampling from span-3
to span-8 to span-12 moves pan accuracy only from 0.32 to 0.31 to 0.35 -- all
still near chance. The big jumps come from *integrating the whole window*
(temporal-average 0.58) and then the *frequency decomposition* (FreqST 0.90),
not from how far apart the 3 samples sit. So the confound is ruled out: it's
about integration + frequency structure, not time span. (Note: in a grayscale
pipeline GrayST-wide and TC Reordering are literally the same operation --
stack grayscale frames 0,4,7 -- which is why their rows are identical; that
coincidence is itself a clean confirmation.)

**Check 2 -- which epoch's accuracy is reported, and is 0.90 cherry-picked?**
Every number in every table is the **final-epoch model's** test accuracy --
evaluated once after the last training epoch, with **no best-epoch selection
and no early stopping** -- and it is the *identical* rule for all methods
(same `train_one`). The FreqST-on-pan validation curve is visibly noisy, so I
also recorded the best test accuracy over epochs, per seed, to check the final
number isn't sitting in a dip:

| seed | FreqST B, final epoch | FreqST B, best epoch |
|---|---|---|
| 10 | 0.915 | 0.938 |
| 20 | 0.890 | 0.890 |
| 30 | 0.905 | 0.915 |

**Verdict on Check 2: the headline is not inflated -- if anything it's mildly
conservative.** Final trails best by at most 0.023 (mean 0.903 final vs 0.914
best). Reporting the final epoch slightly *understates* FreqST rather than
cherry-picking its peak, and the 0.6-accuracy gap over the 3-frame methods
(which also report final epoch, and whose best-epoch numbers are likewise
near chance, ~0.33) is far larger than this noise. The conclusion stands.

**Window-size sensitivity (FreqST, Dataset A):** W = 4, 8, 12 all gave 1.000.
For this speed range the choice of window didn't matter for accuracy (the
task is easy on static camera regardless). The Stage-1 finding that large W +
high speed leaks energy out of the kept coefficients didn't bite here because
blob speeds were kept in the resolvable range -- consistent with Stage 1, not
a contradiction. A speed-classification task (not run) would be the place
that window choice should actually start to matter.

**Fairness caveat (important).** FreqST integrates 8 frames into its 3
channels; GrayST/TC use 3 frames. Part of FreqST's pan advantage is simply
"more temporal context." But that's precisely FreqST's design claim -- it
compresses a W-frame window into 3 channels at *no extra cost to the
backbone*, which GrayST/TC (raw stacks) structurally cannot do. So this is a
fair advantage of the method, not an artifact; and the tavg-8 control shows
the frequency encoding contributes beyond frame count alone. (This was
probed directly -- see "Check 1" below: giving the 3-frame methods wider
frame spacing, up to a full 12-frame span, does NOT recover their accuracy,
so the advantage is integration + frequency structure, not span or frame
spacing.)

**Honest scope.** This is still a fully synthetic, controlled task: one
Gaussian blob, a checkerboard background, a rigid pan, coarse 4-way direction
labels. It shows FreqST *can* be competitive on static camera and
*substantially more pan-robust* than raw-frame stacking in a setting built to
stress exactly that -- it does **not** show this holds on real datasets
(Something-Something, CATER) with real backgrounds, multiple objects,
occlusion, parallax, non-rigid camera motion, and a pretrained backbone. The
natural next step, if this is promising enough to justify the cost, is a
small real-data test: e.g. FreqST vs GrayST as the input stub to a pretrained
2D ResNet on a subset of an action dataset, ideally one with a camera-motion
split. Everything here is a green light for that test, not a substitute for it.

---

## Stage 4 -- same-window fairness check

Goal: close off the fairness caveat flagged above. Stage 3 gives every method the
same WINDOW=8 span, but GrayST and TC Reordering only ever read 3 of those 8 raw
frames (see `transforms.py`) -- so "same window" and "same amount of frame content
consumed" are not the same thing. This script hands every method the literal same
T=9 window and asks what each one does with it. Adds one new control, `chunk_avg`:
split the 9 frames into 3 contiguous groups, average each group into one channel --
every frame contributes, nothing discarded, but still no frequency decomposition
(isolates "uses the whole window" from "uses a DCT"). Same underlying clips as
Stage 3's hard regime (identical `build_dataset` seeds), just re-sliced to 9 frames,
single seed. Not part of `run_all.sh` -- run manually via
`python -m experiments.stage4_window9`. Full writeup: `docs/stage4_window9.md`.

**Result** (`results/stage4_window9/metrics.txt`, `window9_comparison.png`):

| method | frames used | A: static | B: + pan | drop A->B |
|---|---|---|---|---|
| RGB (1 frame, window ignored) | 1 of 9 | 0.282 | 0.230 | 0.052 |
| GrayST (3f of 9, consecutive) | 3 of 9 (6 discarded) | 1.000 | 0.275 | 0.725 |
| TC Reordering (3f of 9, spread) | 3 of 9 (6 discarded) | 1.000 | 0.282 | 0.718 |
| Chunk-avg (3 groups of 9, none dropped) | 9 of 9 | 1.000 | 0.620 | 0.380 |
| Temporal avg (9f) | 9 of 9 | 0.485 | 0.515 | -0.030 |
| **FreqST (9f DCT)** | 9 of 9 | 1.000 | **0.880** | 0.120 |

**Verdict: the fairness objection doesn't hold up.** GrayST and TC Reordering are
completely unaffected by having 6 extra frames sitting right there in the window --
they still only look at 3, because that's how those methods are defined, not because
they were starved of data in Stage 3. Chunk-avg (uses everything, no DCT) climbs to
0.620 on pan -- better than raw stacking, confirming "use the whole window" helps on
its own -- but still well short of FreqST's 0.880. The DCT is doing real work beyond
just aggregating more frames.

Note on reproducibility: re-running this on a different machine (Windows/CPU, this
session) than the original results shifted a couple of numbers noticeably --
RGB (0.242->0.282 static, no real signal either way, both ~chance) and Temporal avg
(0.335->0.485 static) moved by more than seed noise alone would suggest, likely
library/BLAS-version differences affecting the untrained/lightly-trained baselines
most. The methods that matter for the headline claim (GrayST/TC collapsing, FreqST
holding at ~0.88) reproduced closely. Worth pinning package versions
(`requirements.txt` currently has no version pins) if exact reproducibility across
machines becomes important later.

---

### Summary across stages
- **Stage 1:** FreqST's DCT transform behaves exactly as designed (DC=appearance,
  AC=motion), monotonically in a speed range matched to the window length;
  energy leaks past the kept coefficients at high speed.
- **Stage 1b:** On real static-camera video it's clean; under a real-texture
  camera pan it does NOT cleanly isolate the object at the pixel level (ties
  TC Reordering, ~5% motion energy on the object).
- **Stage 3:** Despite that pixel-level corruption, FreqST is competitive with
  GrayST on a static-camera classification task and *much* more robust to a
  camera-pan nuisance (0.90 vs ~0.31), thanks to temporal integration plus
  frequency structure. Verified across seeds with a temporal-average control.
- **Stage 4:** Ruled out the fairness objection that FreqST wins only because it
  sees more raw frames -- GrayST/TC ignore extra frames even when handed the
  identical window, and a fairer "use everything, no DCT" control (chunk-avg,
  0.62) still trails FreqST (0.88) by a wide margin.
- **Overall:** the idea is worth a real-data test. It is not proven on real
  video, and its advantage is clearest specifically under camera motion --
  the exact case that motivated it.
