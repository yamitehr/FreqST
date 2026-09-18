# Diving48 — Dataset Choice and Integration Notes

## Why Diving48 instead of CATER

CATER's camera doesn't actually move smoothly — it jumps to a new position every 30 frames.
Within a typical FreqST window (e.g. 8 frames), the camera is mostly static. That means
CATER doesn't really test what FreqST is designed to do (separate smooth camera motion
from object motion).

Diving48 has **real, smooth broadcast camera tracking** — the camera follows the diver
throughout the dive. This is continuous motion, exactly the type FreqST should handle well.

Other advantages:
- ~10 GB (vs CATER's ~35 GB)
- Same background in all videos (diving pool) — forces the model to use temporal/motion info, can't cheat on appearance
- 48 action classes (dive types defined by takeoff, somersaults, twists, body position)
- 18,404 clips (16,067 train / 2,337 test)
- Well supported in existing tools (MMAction2, HuggingFace)

## Kim et al.'s codebase already supports Diving48

The channel_sampling submodule (`third_party/channel_sampling/`) is Kim et al.'s PyVideoAI
framework. It already has Diving48 configs ready to go:

- Dataset config: `dataset_configs/diving48_v2.py`
- Experiment configs in `exp_configs/ch_tcgrey/diving48_v2/`:
  - `tsn_resnet50-RGB_8frame.py`
  - `tsn_resnet50-GreyST_8frame.py`
  - `tsn_resnet50-TC_8frame.py`
  - (also TSM, TRN, MTRN variants)

## How training works

One command, three arguments:
```bash
python tools/run_train.py -D diving48_v2 -M tsn_resnet50 -E GreyST_8frame -c:e tcgrey
```
- `-D` = dataset
- `-M` = model (backbone)
- `-E` = experiment name (determines preprocessing: RGB / GrayST / TC / FreqST)
- `-c:e` = config channel (always `tcgrey` for this paper)

## How preprocessing works (where FreqST plugs in)

Each experiment config defines a function `_dataloader_shape_to_model_input_shape(inputs)`
that converts raw loaded frames into the 3-channel input the model expects:

- **RGB**: pass-through (3 color channels, no conversion)
- **GrayST**: loads T grayscale frames, reshapes every 3 consecutive frames into 3 channels
- **TC Reordering**: reorders RGB channels across frames using index permutation
- **FreqST (to be added)**: load W consecutive grayscale frames, apply 1D DCT along time axis, keep 3 coefficients as channels

The conversion function lives in the experiment config file. To add FreqST we need to:
1. Create a new experiment config (e.g. `tsn_resnet50-FreqST_8frame.py`)
2. Define `_dataloader_shape_to_model_input_shape()` that applies our DCT transform
3. Set `sampling_mode = 'FreqST'` and adjust frame sampling count if needed

Our DCT code already exists in `transforms.py` at the project root — we just need to wire
it into PyVideoAI's pipeline.

## What's done so far

- [x] Initialized the channel_sampling submodule
- [x] Explored the codebase structure and found all integration points
- [ ] Azure VM setup (in progress)
- [ ] Download Diving48 to VM
- [ ] Add FreqST config and preprocessing
- [ ] Train RGB, GrayST, FreqST and compare
