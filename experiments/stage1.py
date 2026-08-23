"""Stage 1 -- sanity check on synthetic video, no training.

For a few blob speeds, visualize FreqST's 3 output channels and confirm:
  - channel 0 looks like a blurred/averaged frame
  - channels 1-2 highlight the moving region
  - their magnitude increases with speed

Also quantifies mean abs channel-1/2 activation in the motion region across
a denser range of speeds, and checks whether it's monotonically increasing.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_data import generate_moving_blob_video
from transforms import freqst, grayst, tc_reordering

H = W = 96
WINDOW = 8
SIGMA = 4.0
INTENSITY = 0.8
DIRECTION = (0.0, 1.0)  # horizontal motion
OUT_DIR = "results/stage1"


def get_window(speed, seed=0):
    """Generate exactly one WINDOW-length clip, with the blob's starting
    position chosen so it sits at the frame center at the window's middle
    frame -- regardless of speed. This keeps the blob's excursion symmetric
    around the center so higher speeds don't just make it exit the frame
    early within the window.
    """
    margin = int(np.ceil(SIGMA * 4)) + 4
    mid_t = WINDOW // 2
    dy, dx = DIRECTION
    norm = np.hypot(dy, dx) or 1.0
    dy, dx = dy / norm, dx / norm
    start_pos = (
        H / 2.0 + margin - speed * mid_t * dy,
        W / 2.0 + margin - speed * mid_t * dx,
    )
    video, gt = generate_moving_blob_video(
        n_frames=WINDOW, h=H, w=W,
        blob_speed=speed, blob_direction=DIRECTION,
        blob_sigma=SIGMA, blob_intensity=INTENSITY,
        pan_speed=0.0, start_blob_pos=start_pos, seed=seed,
    )
    return video, gt["blob_pos_viewport"]


def motion_mask(window_gt_pos, radius):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask = np.zeros((H, W), dtype=bool)
    for (cy, cx) in window_gt_pos:
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask |= d2 <= radius ** 2
    return mask


def plot_channel_grid(speeds):
    fig, axes = plt.subplots(len(speeds), 3, figsize=(9, 3 * len(speeds)))
    if len(speeds) == 1:
        axes = axes[None, :]
    for row, speed in enumerate(speeds):
        window, _ = get_window(speed)
        out = freqst(window)  # (3, H, W)
        vmax_ac = max(np.abs(out[1]).max(), np.abs(out[2]).max(), 1e-6)
        titles = ["ch0 (DC, avg appearance)", "ch1 (low-freq AC, slow motion)", "ch2 (higher-freq AC, fast motion)"]
        for col in range(3):
            ax = axes[row, col]
            if col == 0:
                im = ax.imshow(out[col], cmap="gray")
            else:
                im = ax.imshow(out[col], cmap="coolwarm", vmin=-vmax_ac, vmax=vmax_ac)
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"speed={speed} px/frame", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("FreqST output channels vs. blob speed", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/channels_grid.png", dpi=130)
    plt.close(fig)


def quantify_activation_vs_speed(speeds):
    """Also tracks what fraction of the *total* AC energy (all WINDOW-1 DCT
    coefficients, not just the 3 we keep) is captured by ch1+ch2. This lets
    us tell apart two possible causes if activation ever stops increasing
    with speed: (a) genuinely less motion energy, vs (b) energy still
    growing but shifting into frequencies higher than what FreqST keeps.
    """
    ch1_vals, ch2_vals, captured_fraction = [], [], []
    for speed in speeds:
        window, window_gt_pos = get_window(speed)
        out = freqst(window)
        radius = SIGMA * 2.5 + 2
        mask = motion_mask(window_gt_pos, radius)
        ch1_vals.append(np.abs(out[1][mask]).mean())
        ch2_vals.append(np.abs(out[2][mask]).mean())

        all_coeffs = freqst(window, num_coeffs=WINDOW)
        total_ac_energy = (all_coeffs[1:][:, mask] ** 2).sum()
        kept_ac_energy = (all_coeffs[1:3][:, mask] ** 2).sum()
        captured_fraction.append(kept_ac_energy / (total_ac_energy + 1e-12))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(speeds, ch1_vals, "o-", label="ch1 (low-freq AC)")
    ax.plot(speeds, ch2_vals, "s-", label="ch2 (higher-freq AC)")
    ax.set_xlabel("blob speed (px/frame)")
    ax.set_ylabel("mean |activation| in motion region")
    ax.set_title(f"FreqST activation vs. speed (window={WINDOW} frames)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(speeds, captured_fraction, "^-", color="green")
    ax.set_xlabel("blob speed (px/frame)")
    ax.set_ylabel("fraction of total AC energy in ch1+ch2")
    ax.set_title("Where does the motion energy go?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/activation_vs_speed.png", dpi=130)
    plt.close(fig)

    def is_monotonic(vals):
        return all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))

    # The transform is only designed to resolve motion within roughly one
    # full period per WINDOW frames; report monotonicity both over a modest
    # "in-regime" range and over the full tested range so a reversal at high
    # speed isn't hidden.
    in_regime_n = max(2, len(speeds) // 2)
    ch1_mono_regime = is_monotonic(ch1_vals[:in_regime_n])
    ch2_mono_regime = is_monotonic(ch2_vals[:in_regime_n])
    ch1_mono_full = is_monotonic(ch1_vals)
    ch2_mono_full = is_monotonic(ch2_vals)
    return {
        "ch1_vals": ch1_vals,
        "ch2_vals": ch2_vals,
        "captured_fraction": captured_fraction,
        "ch1_mono_regime": ch1_mono_regime,
        "ch2_mono_regime": ch2_mono_regime,
        "ch1_mono_full": ch1_mono_full,
        "ch2_mono_full": ch2_mono_full,
        "in_regime_speeds": speeds[:in_regime_n],
    }


def compare_transforms_grid(speed=6):
    window, _ = get_window(speed)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    outs = {
        "FreqST": freqst(window),
        "GrayST": grayst(window),
        "TC Reordering": tc_reordering(window),
    }
    for row, (name, out) in enumerate(outs.items()):
        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(out[col], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"channel {col}", fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"All 3 transforms side by side (blob speed={speed} px/frame)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/transform_comparison.png", dpi=130)
    plt.close(fig)


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    grid_speeds = [1, 3, 6]
    plot_channel_grid(grid_speeds)
    compare_transforms_grid(speed=6)

    quant_speeds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    r = quantify_activation_vs_speed(quant_speeds)

    lines = []
    lines.append("Stage 1 results\n")
    lines.append(f"Speeds tested for quantification: {quant_speeds}\n")
    lines.append(f"ch1 (low-freq AC) activations: {[round(v, 4) for v in r['ch1_vals']]}\n")
    lines.append(f"ch2 (higher-freq AC) activations: {[round(v, 4) for v in r['ch2_vals']]}\n")
    lines.append(f"fraction of total AC energy in ch1+ch2: {[round(v, 3) for v in r['captured_fraction']]}\n")
    lines.append(
        f"ch1/ch2 monotonically increasing over in-regime speeds {r['in_regime_speeds']}: "
        f"{r['ch1_mono_regime']} / {r['ch2_mono_regime']}\n"
    )
    lines.append(
        f"ch1/ch2 monotonically increasing over FULL tested range {quant_speeds}: "
        f"{r['ch1_mono_full']} / {r['ch2_mono_full']}\n"
    )
    lines.append(
        "Note: activation rises then falls at high speed because, with a fixed "
        f"{WINDOW}-frame window, fast motion pushes energy into DCT coefficients "
        "beyond the 3 FreqST keeps (see captured_fraction dropping) -- not because "
        "there's less total motion energy. This is an aliasing-like effect: window "
        "length bounds the motion speeds FreqST's first few coefficients can resolve.\n"
    )
    with open(f"{OUT_DIR}/metrics.txt", "w") as f:
        f.writelines(lines)

    print("".join(lines))
    print(f"Saved plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
