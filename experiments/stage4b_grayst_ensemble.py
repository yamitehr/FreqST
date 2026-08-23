"""Stage 4b -- GrayST as a 3-glimpse ensemble: no concatenation, no averaging
of pixels, no discarding.

Splits the 9-frame window into 3 non-overlapping GrayST triplets:
  glimpse 1 = frames (0,1,2)  glimpse 2 = frames (3,4,5)  glimpse 3 = frames (6,7,8)
Each glimpse is a genuine, unmodified GrayST image (3 raw frames stacked as
channels -- identical operation to transforms.grayst(), just applied to a
different 3-frame slice of the window). All 9 raw frames are used, none
discarded, none blurred by averaging.

Per the user's design:
  - TRAINING: every clip contributes 3 training examples (one per glimpse),
    all carrying the clip's single ground-truth label -- 3x the effective
    training set size.
  - TEST: the 3 glimpses of a test clip are each scored by the (single,
    shared) CNN, their softmax outputs are averaged, and the argmax of that
    average is the ONE final prediction for the clip -- directly comparable
    to every other method's one-prediction-per-clip accuracy number.

Same underlying clips as stage3.py / stage4_window9.py (identical
build_dataset seeds), hard regime, single seed.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.stage3 as s3
from models import TinyCNN

OUT_DIR = "results/stage4b_grayst_ensemble"
WINDOW_T = 9
N_GLIMPSES = 3
GLIMPSE_LEN = WINDOW_T // N_GLIMPSES  # 3 frames per glimpse
H, W_IMG = s3.H, s3.W_IMG
BATCH, LR, SEED = s3.BATCH, s3.LR, s3.SEED

# Reference numbers from results/stage4_window9/metrics.txt (same clips/regime/seed)
REFERENCE = {
    "GrayST (original, last 3 of 9)": (1.000, 0.280),
    "Chunk-avg (3 groups averaged, 9 used)": (1.000, 0.640),
    "FreqST (9f DCT)": (1.000, 0.875),
}


def build_glimpses(clips, window=WINDOW_T):
    """(N, CLIP_LEN, H, W) -> (N, N_GLIMPSES, GLIMPSE_LEN, H, W), each glimpse
    a genuine GrayST image (3 consecutive raw frames, unmodified)."""
    win = clips[:, -window:]
    imgs = np.stack([win[:, i * GLIMPSE_LEN:(i + 1) * GLIMPSE_LEN] for i in range(N_GLIMPSES)], axis=1)
    return imgs.astype(np.float32)


def normalize_glimpses(train_imgs, *others):
    flat = train_imgs.reshape(-1, GLIMPSE_LEN, H, W_IMG)
    mean = flat.mean(axis=(0, 2, 3)).reshape(1, 1, GLIMPSE_LEN, 1, 1)
    std = flat.std(axis=(0, 2, 3)).reshape(1, 1, GLIMPSE_LEN, 1, 1) + 1e-6
    norm = lambda x: (x - mean) / std
    return [norm(train_imgs)] + [norm(o) for o in others]


def train_ensemble(Xtr_g, ytr, Xva_g, yva, Xte_g, yte, base_ch, epochs, seed=SEED):
    """Xtr_g/Xva_g/Xte_g: (N, N_GLIMPSES, GLIMPSE_LEN, H, W). Train on all
    N*N_GLIMPSES (image,label) pairs; eval by averaging the N_GLIMPSES
    softmax outputs per clip into one prediction."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyCNN(num_classes=4, in_channels=GLIMPSE_LEN, base_channels=base_ch)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    Xtr = torch.tensor(Xtr_g.reshape(-1, GLIMPSE_LEN, H, W_IMG))
    ytr_rep = torch.tensor(np.repeat(ytr, N_GLIMPSES))
    n = Xtr.shape[0]

    Xva = torch.tensor(Xva_g); yva_t = torch.tensor(yva)
    Xte = torch.tensor(Xte_g); yte_t = torch.tensor(yte)

    def ensemble_predict(X_glimpses):
        n_clips = X_glimpses.shape[0]
        flat = X_glimpses.reshape(n_clips * N_GLIMPSES, GLIMPSE_LEN, H, W_IMG)
        with torch.no_grad():
            probs = torch.softmax(model(flat), dim=1).reshape(n_clips, N_GLIMPSES, -1).mean(dim=1)
        return probs.argmax(1)

    hist = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for j in range(0, n, BATCH):
            idx = perm[j:j + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr_rep[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        va_acc = (ensemble_predict(Xva) == yva_t).float().mean().item()
        hist["train_loss"].append(total / n)
        hist["val_acc"].append(va_acc)

    model.eval()
    te_pred = ensemble_predict(Xte)
    test_acc = float((te_pred == yte_t).float().mean().item())
    return hist, test_acc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = s3.REGIMES["hard"]

    print("Building Dataset A/B (hard regime, same seeds as stage3) ...", flush=True)
    A_tr, Aytr = s3.build_dataset(cfg["n_train"], False, 1, cfg)
    A_va, Ayva = s3.build_dataset(s3.N_VAL, False, 2, cfg)
    A_te, Ayte = s3.build_dataset(s3.N_TEST, False, 3, cfg)
    B_tr, Bytr = s3.build_dataset(cfg["n_train"], True, 4, cfg)
    B_va, Byva = s3.build_dataset(s3.N_VAL, True, 5, cfg)
    B_te, Byte = s3.build_dataset(s3.N_TEST, True, 6, cfg)

    results = {}
    for name, (tr, ytr, va, yva, te, yte) in [
        ("A", (A_tr, Aytr, A_va, Ayva, A_te, Ayte)),
        ("B", (B_tr, Bytr, B_va, Byva, B_te, Byte)),
    ]:
        Xtr = build_glimpses(tr)
        Xva = build_glimpses(va)
        Xte = build_glimpses(te)
        Xtr, Xva, Xte = normalize_glimpses(Xtr, Xva, Xte)
        hist, acc = train_ensemble(Xtr, ytr, Xva, yva, Xte, yte, cfg["base_ch"], cfg["epochs"])
        results[name] = acc
        print(f"  Dataset {name}: test_acc={acc:.3f}", flush=True)

    all_methods = dict(REFERENCE)
    all_methods["GrayST 3-glimpse ensemble (this script)"] = (results["A"], results["B"])

    lines = [
        "Stage 4b -- GrayST as a 3-glimpse ensemble (train x3 augmented, test softmax-averaged)\n",
        f"window={WINDOW_T} frames -> 3 non-overlapping GrayST triplets: (0,1,2)/(3,4,5)/(6,7,8)\n",
        "Train: each clip -> 3 (image,label) pairs. Test: average the 3 softmax outputs per\n",
        "clip into ONE prediction, compared to the clip's true label (same protocol as every\n",
        "other method's accuracy number). Hard regime, single seed, same clips as stage3/stage4.\n\n",
        f"{'method':<42}{'A(static)':>11}{'B(pan)':>10}\n",
    ]
    for name, (a, b) in all_methods.items():
        lines.append(f"{name:<42}{a:>11.3f}{b:>10.3f}\n")

    with open(f"{OUT_DIR}/metrics.txt", "w") as f:
        f.writelines(lines)
    print("\n" + "".join(lines), flush=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(all_methods.keys())
    x = np.arange(len(names))
    ax.bar(x - 0.2, [all_methods[m][0] for m in names], 0.4, label="A: static camera")
    ax.bar(x + 0.2, [all_methods[m][1] for m in names], 0.4, label="B: + camera pan")
    ax.axhline(0.25, ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Stage 4b: GrayST 3-glimpse ensemble vs. references (hard regime, 1 seed)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ensemble_comparison.png", dpi=130)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
