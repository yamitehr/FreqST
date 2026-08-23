"""Stage 3 -- learnability check: does FreqST help a CNN classify actions?

Two synthetic 4-class (left/right/up/down) blob-direction datasets:
  A: static camera.
  B: same task + random global background pan (nuisance, no label info).

Trains ONE tiny CNN architecture with identical hyperparameters, varying only
the input preprocessing: RGB (single-frame appearance control), GrayST,
TC Reordering, FreqST. 4 methods x 2 datasets = 8 runs, plus a FreqST
window-size sweep (W=4/8/12) on Dataset A.

Runs TWO difficulty regimes:
  easy: bright, low-noise blob -> checks the sanity/ceiling case.
  hard: dim, small, noisy blob, smaller model + less data -> forces the
        methods below ceiling so they can actually be ranked.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_data import generate_labeled_clip, DIRECTION_NAMES
from transforms import grayst, tc_reordering, chunk_avg, freqst
from models import TinyCNN

OUT_DIR = "results/stage3"

# ---- fixed across all runs ----
H = W_IMG = 56
CLIP_LEN = 12          # generated frames; main runs use the last WINDOW of them
WINDOW = 8
N_VAL, N_TEST = 400, 400
SPEED_RANGE = (1.0, 3.0)      # mostly below Stage-1 energy-leakage breakpoint
BATCH = 64
LR = 1e-3
SEED = 0
METHODS = ["rgb", "grayst", "tc_reordering", "freqst"]
METHOD_LABELS = {
    "rgb": "RGB (1-frame)",
    "grayst": "GrayST",
    "tc_reordering": "TC Reordering",
    "freqst": "FreqST",
}

# ---- difficulty regimes (only these differ) ----
REGIMES = {
    "easy": dict(blob_intensity=0.70, blob_sigma=3.0, noise_std=0.03,
                 bg_contrast=(0.15, 0.35), pan_range=(2.0, 4.0),
                 base_ch=16, n_train=2000, epochs=16),
    "hard": dict(blob_intensity=0.30, blob_sigma=2.0, noise_std=0.06,
                 bg_contrast=(0.25, 0.55), pan_range=(3.0, 6.0),
                 base_ch=8, n_train=1000, epochs=22),
}
torch.set_num_threads(max(1, os.cpu_count() // 2))


# ---------------- dataset ----------------
def build_dataset(n, with_pan, seed, cfg):
    rng = np.random.default_rng(seed)
    clips = np.zeros((n, CLIP_LEN, H, W_IMG), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        d = int(rng.integers(4))
        speed = float(rng.uniform(*SPEED_RANGE))
        bg_scale = float(rng.choice([8.0, 14.0, 20.0]))
        bg_contrast = float(rng.uniform(*cfg["bg_contrast"]))
        if with_pan:
            pan_speed = float(rng.uniform(*cfg["pan_range"]))
            ang = float(rng.uniform(0, 2 * np.pi))
            pan_dir = (np.sin(ang), np.cos(ang))
        else:
            pan_speed, pan_dir = 0.0, (0.0, 1.0)
        clip, lab = generate_labeled_clip(
            d, n_frames=CLIP_LEN, h=H, w=W_IMG,
            blob_speed=speed, blob_sigma=cfg["blob_sigma"],
            blob_intensity=cfg["blob_intensity"],
            bg_scale=bg_scale, bg_contrast=bg_contrast, noise_std=cfg["noise_std"],
            pan_speed=pan_speed, pan_direction=pan_dir, rng=rng,
        )
        clips[i] = clip
        labels[i] = lab
    return clips, labels


def make_inputs(clips, method, window):
    """(N, CLIP_LEN, H, W) clips -> (N, 3, H, W) inputs for `method`.

    RGB baseline = single frame at the trajectory's temporal center
    (clip midpoint), replicated to 3 channels. Because the blob path is
    centered on the clip, that frame's blob position is ~independent of
    direction -> a genuine appearance-only control (expected ~chance)."""
    n = clips.shape[0]
    mid_idx = clips.shape[1] // 2
    out = np.zeros((n, 3, H, W_IMG), dtype=np.float32)
    for i in range(n):
        if method == "rgb":
            mid = clips[i, mid_idx]
            out[i] = np.stack([mid, mid, mid], axis=0)
            continue
        if method == "grayst_wide":
            # 3 grayscale frames spread across the W-frame window (frames
            # 0, W/2, W-1) instead of 3 consecutive. Same channel/frame count
            # as GrayST, wider temporal span. NB: in a grayscale pipeline this
            # is identical to TC Reordering (both stack frames [0, W/2, W-1]).
            win = clips[i, -window:]
            out[i] = win[[0, window // 2, window - 1]]
            continue
        if method == "tc_wide":
            # 3 frames spread across the FULL clip (frames 0, L/2, L-1) --
            # the widest possible span (12 frames here, wider than FreqST's
            # 8-frame window). Still 3 frames / 3 channels.
            full = clips[i]
            L = clips.shape[1]
            out[i] = full[[0, L // 2, L - 1]]
            continue
        win = clips[i, -window:]
        if method == "tavg":
            # Control: plain temporal average over the SAME W-frame window,
            # replicated to 3 channels. Isolates "integrate W frames" from
            # "frequency-decompose W frames" (= FreqST's DC channel alone).
            avg = win.mean(axis=0)
            out[i] = np.stack([avg, avg, avg], axis=0)
            continue
        if method == "grayst":
            out[i] = grayst(win)
        elif method == "tc_reordering":
            out[i] = tc_reordering(win)
        elif method == "chunk_avg":
            out[i] = chunk_avg(win)
        elif method == "freqst":
            out[i] = freqst(win)
        else:
            raise ValueError(method)
    return out


def normalize(train, *others):
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return [(x - mean) / std for x in (train, *others)]


# ---------------- training ----------------
def train_one(Xtr, ytr, Xva, yva, Xte, yte, base_ch, epochs, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyCNN(num_classes=4, in_channels=3, base_channels=base_ch)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr)
    Xva = torch.tensor(Xva); yva = torch.tensor(yva)
    Xte = torch.tensor(Xte); yte = torch.tensor(yte)
    n = Xtr.shape[0]

    hist = {"train_loss": [], "val_acc": [], "test_acc": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for j in range(0, n, BATCH):
            idx = perm[j:j + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            va = (model(Xva).argmax(1) == yva).float().mean().item()
            te = (model(Xte).argmax(1) == yte).float().mean().item()
        hist["train_loss"].append(total / n)
        hist["val_acc"].append(va)
        hist["test_acc"].append(te)   # per-epoch test acc (for final-vs-best reporting only; NOT used for selection)

    # Reported accuracy = FINAL epoch's model on test (no best-epoch/early-stop selection).
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1).numpy()
    test_acc = float((preds == yte.numpy()).mean())
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(yte.numpy(), preds):
        cm[t, p] += 1
    return hist, test_acc, cm


def run_suite(clips_tr, ytr, clips_va, yva, clips_te, yte, cfg, window=WINDOW):
    results = {}
    for m in METHODS:
        Xtr = make_inputs(clips_tr, m, window)
        Xva = make_inputs(clips_va, m, window)
        Xte = make_inputs(clips_te, m, window)
        Xtr, Xva, Xte = normalize(Xtr, Xva, Xte)
        t0 = time.time()
        hist, acc, cm = train_one(Xtr, ytr, Xva, yva, Xte, yte,
                                  cfg["base_ch"], cfg["epochs"])
        results[m] = {"hist": hist, "test_acc": acc, "cm": cm}
        print(f"    {m:16s} test_acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)
    return results


# ---------------- plots ----------------
def plot_curves(resA, resB, regime, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for col, (res, name) in enumerate([(resA, "A (static camera)"), (resB, "B (camera pan)")]):
        for m in METHODS:
            h = res[m]["hist"]
            axes[0, col].plot(h["train_loss"], label=METHOD_LABELS[m])
            axes[1, col].plot(h["val_acc"], label=METHOD_LABELS[m])
        axes[0, col].set_title(f"Dataset {name}: training loss")
        axes[1, col].set_title(f"Dataset {name}: validation accuracy")
        axes[1, col].axhline(0.25, ls="--", c="gray", lw=1)
        axes[0, col].set_xlabel("epoch"); axes[0, col].set_ylabel("loss")
        axes[1, col].set_xlabel("epoch"); axes[1, col].set_ylabel("acc")
        for r in range(2):
            axes[r, col].grid(alpha=0.3); axes[r, col].legend(fontsize=8)
    fig.suptitle(f"Stage 3 ({regime} regime): training curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusions(resA, resB, regime, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    for row, (res, name) in enumerate([(resA, "A static"), (resB, "B pan")]):
        for col, m in enumerate(METHODS):
            ax = axes[row, col]
            cm = res[m]["cm"]
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"{name} | {METHOD_LABELS[m]}\nacc={res[m]['test_acc']:.3f}", fontsize=8)
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(DIRECTION_NAMES, fontsize=7, rotation=45)
            ax.set_yticklabels(DIRECTION_NAMES, fontsize=7)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if cmn[i, j] < 0.6 else "white")
    fig.suptitle(f"Stage 3 ({regime} regime): confusion matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------- main ----------------
def run_regime(regime, cfg):
    print(f"\n==== REGIME: {regime} ({cfg}) ====", flush=True)
    rdir = f"{OUT_DIR}/{regime}"
    os.makedirs(rdir, exist_ok=True)

    print("Building Dataset A (static camera) ...", flush=True)
    A_tr, Aytr = build_dataset(cfg["n_train"], False, 1, cfg)
    A_va, Ayva = build_dataset(N_VAL, False, 2, cfg)
    A_te, Ayte = build_dataset(N_TEST, False, 3, cfg)
    print("Building Dataset B (camera pan) ...", flush=True)
    B_tr, Bytr = build_dataset(cfg["n_train"], True, 4, cfg)
    B_va, Byva = build_dataset(N_VAL, True, 5, cfg)
    B_te, Byte = build_dataset(N_TEST, True, 6, cfg)

    print("Training on Dataset A (static camera):", flush=True)
    resA = run_suite(A_tr, Aytr, A_va, Ayva, A_te, Ayte, cfg)
    print("Training on Dataset B (camera pan):", flush=True)
    resB = run_suite(B_tr, Bytr, B_va, Byva, B_te, Byte, cfg)

    plot_curves(resA, resB, regime, f"{rdir}/curves.png")
    plot_confusions(resA, resB, regime, f"{rdir}/confusion.png")

    print("FreqST window-size sweep on Dataset A:", flush=True)
    window_accs = {}
    for win in [4, 8, 12]:
        Xtr = make_inputs(A_tr, "freqst", win)
        Xva = make_inputs(A_va, "freqst", win)
        Xte = make_inputs(A_te, "freqst", win)
        Xtr, Xva, Xte = normalize(Xtr, Xva, Xte)
        _, acc, _ = train_one(Xtr, Aytr, Xva, Ayva, Xte, Ayte, cfg["base_ch"], cfg["epochs"])
        window_accs[win] = acc
        print(f"    W={win:2d}  test_acc={acc:.3f}", flush=True)

    return resA, resB, window_accs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    all_lines = ["Stage 3 results\n"]

    for regime, cfg in REGIMES.items():
        resA, resB, window_accs = run_regime(regime, cfg)
        all_lines.append(f"\n===== {regime.upper()} REGIME =====\n")
        all_lines.append(
            f"blob_intensity={cfg['blob_intensity']}, sigma={cfg['blob_sigma']}, "
            f"noise={cfg['noise_std']}, base_ch={cfg['base_ch']}, n_train={cfg['n_train']}, "
            f"epochs={cfg['epochs']}, pan~U{cfg['pan_range']}\n")
        all_lines.append("Test accuracy (4-class direction, chance=0.25):\n")
        all_lines.append(f"{'method':<16}{'A(static)':>11}{'B(pan)':>10}{'drop A->B':>11}\n")
        for m in METHODS:
            a, b = resA[m]["test_acc"], resB[m]["test_acc"]
            all_lines.append(f"{METHOD_LABELS[m]:<16}{a:>11.3f}{b:>10.3f}{a-b:>11.3f}\n")
        all_lines.append("FreqST window-size sensitivity (Dataset A): "
                         + ", ".join(f"W={w}:{window_accs[w]:.3f}" for w in [4, 8, 12]) + "\n")

    all_lines.append(f"\nTotal runtime: {time.time()-t_start:.0f}s\n")
    with open(f"{OUT_DIR}/metrics.txt", "w") as f:
        f.writelines(all_lines)
    print("\n" + "".join(all_lines), flush=True)
    print(f"Saved to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
