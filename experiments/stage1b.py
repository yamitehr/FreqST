"""Stage 1b -- qualitative check on real video (no training).

Reproduces the comparison style of Figure 3 in Kim et al. 2022 (rows =
method, columns = frames across time) on:
  (a) a real clip (vtest.avi: pedestrians, static camera), and
  (b) a synthetic controlled camera-pan over a REAL static background frame,
      with a synthetic blob composited on top (global + local motion).

Saves the paper-style composites plus FreqST DC/AC1/AC2 channel-grid
breakdowns for each clip.
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_data import generate_pan_over_real_background
from transforms import grayst, tc_reordering, freqst, apply_windowed

OUT_DIR = "results/stage1b"
VIDEO_PATH = "data/vtest.avi"
DISP_H, DISP_W = 180, 240   # display resolution (keeps vtest's 4:3 aspect)
WINDOW = 8
STRIDE = 1
ACTIVE_START = 511          # most-active 9-frame pedestrian window (measured)
N_INPUT = WINDOW + 8        # -> 9 output columns with stride 1


def load_gray_frames(path, start, count, size=(DISP_W, DISP_H), return_color=False):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, color = [], []
    for _ in range(count):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(g.astype(np.float32) / 255.0)
        # BGR (OpenCV) -> RGB for display
        color.append(fr[:, :, ::-1].astype(np.float32) / 255.0)
    cap.release()
    gray = np.stack(frames, axis=0)
    if return_color:
        return gray, np.stack(color, axis=0)
    return gray


def load_static_background(path, frame_idx=0, size=None):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if size is not None:
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return g


def channel_ranges(stack, symmetric_channels=()):
    """Per-channel (lo, hi) for display, computed across all columns so
    colours are comparable over time. `stack` is (T', 3, H, W)."""
    ranges = []
    for c in range(3):
        vals = stack[:, c]
        if c in symmetric_channels:
            m = max(abs(float(vals.min())), abs(float(vals.max())), 1e-6)
            ranges.append((-m, m))
        else:
            ranges.append((float(vals.min()), float(vals.max())))
    return ranges


def to_rgb(out3, ranges):
    """(3,H,W) -> (H,W,3) in [0,1], each channel scaled by its own range."""
    chans = []
    for c in range(3):
        lo, hi = ranges[c]
        x = (out3[c] - lo) / (hi - lo + 1e-8)
        chans.append(np.clip(x, 0, 1))
    return np.stack(chans, axis=-1)


def build_comparison_figure(frames, title, out_path, original_rgb=None):
    """Paper-style Figure-3 layout: 4 rows (Original, Time-Color Reordering,
    GrayST, FreqST), one column per sliding-window output position. Matches
    Kim et al. Fig 3 (rows=method, cols=time). If `original_rgb` (T,H,W,3) is
    given, the Original row shows true colour frames as in the paper."""
    gray_stack = apply_windowed(frames, grayst, WINDOW, STRIDE)
    tc_stack = apply_windowed(frames, tc_reordering, WINDOW, STRIDE)
    freq_stack = apply_windowed(frames, freqst, WINDOW, STRIDE)
    n_cols = gray_stack.shape[0]

    # Reference frame per column = last (most recent) frame of that window.
    ref_idx = [i * STRIDE + WINDOW - 1 for i in range(n_cols)]

    gray_r = channel_ranges(gray_stack)
    tc_r = channel_ranges(tc_stack)
    freq_r = channel_ranges(freq_stack, symmetric_channels=(1, 2))

    if original_rgb is not None:
        orig_kind = "rgb_raw"
        orig_data = [original_rgb[i] for i in ref_idx]
        orig_label = "Original RGB"
    else:
        orig_kind = "gray"
        orig_data = [frames[i] for i in ref_idx]
        orig_label = "Original\n(gray frame)"

    rows = [
        (orig_label, orig_kind, orig_data, None),
        ("Time-Color\nReordering", "rgb", tc_stack, tc_r),
        ("GrayST", "rgb", gray_stack, gray_r),
        ("FreqST\n(DC=R, AC1=G, AC2=B)", "rgb", freq_stack, freq_r),
    ]

    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.5 * n_cols, 1.5 * len(rows)))
    for r, (label, kind, data, ranges) in enumerate(rows):
        for col in range(n_cols):
            ax = axes[r, col]
            if kind == "gray":
                ax.imshow(data[col], cmap="gray", vmin=0, vmax=1)
            elif kind == "rgb_raw":
                ax.imshow(np.clip(data[col], 0, 1))
            else:
                ax.imshow(to_rgb(data[col], ranges))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={ref_idx[col]}", fontsize=8)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_channel_grid(frames, title, out_path, example_positions=None):
    """FreqST DC/AC1/AC2 as separate grayscale rows; columns = a few window
    positions."""
    freq_stack = apply_windowed(frames, freqst, WINDOW, STRIDE)
    n = freq_stack.shape[0]
    if example_positions is None:
        example_positions = sorted(set([0, n // 2, n - 1]))
    ref_idx = [p * STRIDE + WINDOW - 1 for p in example_positions]

    ch_names = ["ch0 (DC, avg appearance)", "ch1 (AC1, slow motion)", "ch2 (AC2, fast motion)"]
    ac_max = max(np.abs(freq_stack[:, 1]).max(), np.abs(freq_stack[:, 2]).max(), 1e-6)

    fig, axes = plt.subplots(3, len(example_positions),
                             figsize=(2.6 * len(example_positions), 7.2))
    if len(example_positions) == 1:
        axes = axes[:, None]
    for r in range(3):
        for c, p in enumerate(example_positions):
            ax = axes[r, c]
            img = freq_stack[p, r]
            if r == 0:
                im = ax.imshow(img, cmap="gray")
            else:
                im = ax.imshow(img, cmap="coolwarm", vmin=-ac_max, vmax=ac_max)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"window ending t={ref_idx[c]}", fontsize=9)
            if c == 0:
                ax.set_ylabel(ch_names[r], fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def ac_spatial_spread(frames):
    """Quantify how spatially spread-out the AC (motion) activation is.
    Returns the fraction of pixels needed to account for 80% of total AC
    energy -- small = concentrated on a compact moving region, large =
    smeared across the whole frame (as a global pan would cause)."""
    freq_stack = apply_windowed(frames, freqst, WINDOW, STRIDE)
    tc_stack = apply_windowed(frames, tc_reordering, WINDOW, STRIDE)

    def conc(stack, is_freq):
        fracs = []
        for out in stack:
            if is_freq:
                energy = out[1] ** 2 + out[2] ** 2
            else:
                # TC motion signal = deviation of the 3 stacked frames from
                # their mean (i.e. how much the raw pixels changed over time)
                mean = out.mean(axis=0)
                energy = ((out - mean) ** 2).sum(axis=0)
            e = np.sort(energy.ravel())[::-1]
            csum = np.cumsum(e)
            total = csum[-1] + 1e-12
            k = np.searchsorted(csum, 0.8 * total) + 1
            fracs.append(k / energy.size)
        return float(np.mean(fracs))

    return conc(freq_stack, True), conc(tc_stack, False)


def blob_energy_fraction(frames, gt):
    """Fraction of motion energy that lands ON the known local blob (vs. the
    panning background). Uses the blob's ground-truth position within each
    window. Higher = method isolates the local object; lower = the global
    pan's background edges dominate the motion signal."""
    freq_stack = apply_windowed(frames, freqst, WINDOW, STRIDE)
    tc_stack = apply_windowed(frames, tc_reordering, WINDOW, STRIDE)
    n = freq_stack.shape[0]
    _, h, w = freq_stack.shape[1:]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    blob_pos = gt["blob_pos_viewport"]
    radius = 14.0  # generous footprint around blob path within the window

    def frac(stack, is_freq):
        vals = []
        for i in range(n):
            mask = np.zeros((h, w), dtype=bool)
            for t in range(i * STRIDE, i * STRIDE + WINDOW):
                cy, cx = blob_pos[t]
                mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            out = stack[i]
            if is_freq:
                energy = out[1] ** 2 + out[2] ** 2
            else:
                mean = out.mean(axis=0)
                energy = ((out - mean) ** 2).sum(axis=0)
            vals.append(energy[mask].sum() / (energy.sum() + 1e-12))
        return float(np.mean(vals))

    return frac(freq_stack, True), frac(tc_stack, False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # (a) Real pedestrian clip.
    real_frames, real_color = load_gray_frames(VIDEO_PATH, ACTIVE_START, N_INPUT, return_color=True)
    build_comparison_figure(
        real_frames,
        "Stage 1b (a): vtest.avi -- real pedestrians, static camera",
        f"{OUT_DIR}/vtest_comparison.png",
        original_rgb=real_color,
    )
    build_channel_grid(
        real_frames,
        "FreqST channels on vtest.avi (real pedestrian motion)",
        f"{OUT_DIR}/vtest_channels_grid.png",
    )

    # (b) Controlled pan over a real static background, with composited blob.
    bg = load_static_background(VIDEO_PATH, frame_idx=0)  # full-res static frame
    PAN_SPEED = 5.0
    BLOB_SPEED = 3.0
    pan_frames, pan_gt = generate_pan_over_real_background(
        bg, n_frames=N_INPUT, h=DISP_H, w=DISP_W,
        pan_speed=PAN_SPEED, pan_direction=(0.0, 1.0),
        add_blob=True, blob_speed=BLOB_SPEED, blob_direction=(1.0, 0.0),
        blob_sigma=7.0, blob_intensity=0.7,
    )
    build_comparison_figure(
        pan_frames,
        f"Stage 1b (b): real bg + camera pan ({PAN_SPEED}px/f) + local blob ({BLOB_SPEED}px/f)",
        f"{OUT_DIR}/panmotion_comparison.png",
    )
    build_channel_grid(
        pan_frames,
        f"FreqST channels on pan-over-real-bg (pan={PAN_SPEED}, blob={BLOB_SPEED} px/f)",
        f"{OUT_DIR}/panmotion_channels_grid.png",
    )

    # Quantify spatial concentration of motion activation on the pan clip.
    freq_conc_real, tc_conc_real = ac_spatial_spread(real_frames)
    freq_conc_pan, tc_conc_pan = ac_spatial_spread(pan_frames)

    # Fraction of motion energy that lands ON the local blob (pan clip only).
    freq_blob, tc_blob = blob_energy_fraction(pan_frames, pan_gt)

    lines = [
        "Stage 1b quantitative side-notes\n",
        "\n[1] AC-energy spatial spread = fraction of pixels holding 80% of motion energy\n",
        "    (smaller = motion concentrated on a compact region; larger = smeared over frame)\n",
        f"    vtest (static cam):   FreqST={freq_conc_real:.3f}   TC Reordering={tc_conc_real:.3f}\n",
        f"    pan+blob clip:        FreqST={freq_conc_pan:.3f}   TC Reordering={tc_conc_pan:.3f}\n",
        "\n[2] Fraction of motion energy landing ON the local blob (pan clip only)\n",
        "    (higher = isolates the local object; lower = global pan's bg edges dominate)\n",
        f"    pan+blob clip:        FreqST={freq_blob:.3f}   TC Reordering={tc_blob:.3f}\n",
    ]
    with open(f"{OUT_DIR}/metrics.txt", "w") as f:
        f.writelines(lines)
    print("".join(lines))
    print(f"Saved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
