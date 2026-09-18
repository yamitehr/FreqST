# FreqST Real-Data Experiment Plan

## Background

The synthetic prototype is done (stages 1-4b). It proves FreqST works on fake data.
Now we need to show it works on **real video**.

**Dataset: Diving48** (~10 GB, 18,404 clips, 48 dive types, smooth broadcast camera tracking).
Chosen over CATER because CATER's camera "jumps" rather than moves smoothly, which
doesn't match what FreqST is designed to handle.

**Goal: 3 training runs, same backbone, different preprocessing:**

| Run | Preprocessing | What it shows |
|---|---|---|
| RGB | Single frame, no temporal info | Floor: how much does temporal info matter? |
| GrayST | 3 consecutive grayscale frames | Kim et al.'s baseline |
| **FreqST** | DCT over W-frame window | Our method |

Backbone: TSN-ResNet-50 (pretrained on ImageNet) for all three.

---

## Phase 1: Infrastructure Setup

- [V] **1.1** Initialize the channel_sampling submodule
  - Run: `git submodule update --init`
  - This pulls Kim et al.'s code into `third_party/channel_sampling/`
  - Verify: the folder is no longer empty

- [V] **1.2** Explore the channel_sampling codebase
  - Understand the folder structure
  - Find the main training script
  - Find where preprocessing (GrayST/RGB) is applied
  - Find configuration files (how do you choose GrayST vs RGB?)

- [V] **1.3** Set up the Azure VM environment
  - Confirm GPU is available and working (`nvidia-smi`)
  - Install/verify PyTorch with CUDA support
  - Install dependencies from channel_sampling's requirements
  - Clone/copy the repo to the VM

- [V - almost done, out of storage] **1.4** Download Diving48 dataset to Azure VM
  - Download videos (~10 GB)
  - Extract to per-frame JPEGs (using ffmpeg or provided scripts)
  - Download and set up train/test split annotation files
  - Verify: correct number of videos in train (16,067) and test (2,337)

- [ ] **1.5** Verify Diving48 loads correctly in PyVideoAI
  - Configure PyVideoAI to point to the Diving48 data directory
  - Run a quick test: load a batch, print shapes, confirm no errors
  - Visualize a few sample frames to make sure they look right

---

## Phase 2: Integration

- [ ] **2.1** Understand how GrayST preprocessing works in PyVideoAI
  - Read the dataloader code
  - Trace the path: video file -> frame loading -> preprocessing -> model input
  - Identify the exact function/class where GrayST converts frames to 3 channels
  - Note the input format (shape, dtype, value range) the model expects

- [ ] **2.2** Add FreqST as a new preprocessing mode
  - Copy/adapt the DCT transform from our `transforms.py`
  - Add a "freqst" option alongside "grayst" and "rgb" in PyVideoAI
  - FreqST needs W consecutive frames (e.g. W=6 or W=8) instead of 3
  - Make sure the output is the same shape/format as GrayST (3 channels, same H x W)

- [ ] **2.3** Sanity check the FreqST integration
  - Run the pipeline with FreqST on a few clips
  - Visualize the 3 output channels:
    - Channel 0 (DC) should look like an averaged frame
    - Channels 1-2 (AC) should show motion energy
  - Confirm output shape matches what the backbone expects
  - Run a few training iterations, confirm loss decreases

---

## Phase 3: Training

- [ ] **3.1** Train Run 1: RGB baseline
  - TSN-ResNet-50 with single-frame RGB input
  - Full training schedule on Diving48 train set
  - Record: top-1 accuracy on test set, training curves

- [ ] **3.2** Train Run 2: GrayST baseline
  - TSN-ResNet-50 with GrayST preprocessing
  - Same training schedule and hyperparameters as RGB
  - Record: top-1 accuracy on test set, training curves

- [ ] **3.3** Train Run 3: FreqST (ours)
  - TSN-ResNet-50 with FreqST preprocessing
  - Same training schedule and hyperparameters
  - Record: top-1 accuracy on test set, training curves

- [ ] **3.4** Compare results
  - Build comparison table (method vs accuracy)
  - Plot training curves side by side
  - Analyze: does FreqST beat GrayST? By how much?
  - If FreqST loses: diagnose (wrong window size? bug? DCT mismatch?)

---

## Phase 4: Writeup

- [ ] **4.1** Add real-data results to the report
  - Describe Diving48 and why we chose it over CATER
  - Present comparison table and training curves
  - Discuss what the results mean for FreqST's thesis

- [ ] **4.2** Prepare final presentation (if required)

---

## Open Questions (to resolve as we go)

- What window size W to use for FreqST on Diving48? (W=6 from pilot plan? W=8 from synthetic experiments?)
- What training schedule / hyperparameters does PyVideoAI default to for Diving48?
- Does the channel_sampling codebase already support Diving48, or do we need to add a dataset config?
