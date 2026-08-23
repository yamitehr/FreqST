"""Stage 3 verification + follow-up controls.

Checks, in the HARD regime across 3 seeds:
  - Is FreqST's camera-pan robustness stable across seeds? (original check)
  - Is it just "integrate more frames"? -> tavg (8-frame average) control.
  - Is it just "narrow time span"? -> grayst_wide (3 frames spread over the
    8-frame window) and tc_wide (3 frames spread over the full 12-frame clip)
    controls. Both keep 3 frames / 3 channels but widen the sampling span.
  - Epoch selection: reports FINAL-epoch accuracy (the rule used everywhere)
    AND best-over-epochs accuracy, side by side, for every method -- so we
    can see whether the noisy FreqST-on-pan curve inflates the headline.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.stage3 as s3

OUT_DIR = "results/stage3"
SEEDS = [10, 20, 30]
# ordered by increasing temporal span / integration
METHODS = ["rgb", "grayst", "grayst_wide", "tc_reordering", "tc_wide", "tavg", "freqst"]
LABELS = {
    "rgb": "RGB (1 frame)",
    "grayst": "GrayST (3f, consecutive)",
    "grayst_wide": "GrayST-wide (3f, span 8)",
    "tc_reordering": "TC Reordering (3f, span 8)",
    "tc_wide": "TC-wide (3f, span 12)",
    "tavg": "Temporal avg (8f)",
    "freqst": "FreqST (8f DCT)",
}


def run_seed(seed, cfg):
    """Returns {method: dict(A_final, B_final, B_best)} for one seed."""
    A_tr, Aytr = s3.build_dataset(cfg["n_train"], False, 100 + seed, cfg)
    A_va, Ayva = s3.build_dataset(s3.N_VAL, False, 200 + seed, cfg)
    A_te, Ayte = s3.build_dataset(s3.N_TEST, False, 300 + seed, cfg)
    B_tr, Bytr = s3.build_dataset(cfg["n_train"], True, 400 + seed, cfg)
    B_va, Byva = s3.build_dataset(s3.N_VAL, True, 500 + seed, cfg)
    B_te, Byte = s3.build_dataset(s3.N_TEST, True, 600 + seed, cfg)

    out = {}
    for m in METHODS:
        # Dataset A
        Xtr = s3.make_inputs(A_tr, m, s3.WINDOW)
        Xva = s3.make_inputs(A_va, m, s3.WINDOW)
        Xte = s3.make_inputs(A_te, m, s3.WINDOW)
        Xtr, Xva, Xte = s3.normalize(Xtr, Xva, Xte)
        _, a_final, _ = s3.train_one(Xtr, Aytr, Xva, Ayva, Xte, Ayte,
                                     cfg["base_ch"], cfg["epochs"], seed=seed)
        # Dataset B
        Xtr = s3.make_inputs(B_tr, m, s3.WINDOW)
        Xva = s3.make_inputs(B_va, m, s3.WINDOW)
        Xte = s3.make_inputs(B_te, m, s3.WINDOW)
        Xtr, Xva, Xte = s3.normalize(Xtr, Xva, Xte)
        histB, b_final, _ = s3.train_one(Xtr, Bytr, Xva, Byva, Xte, Byte,
                                         cfg["base_ch"], cfg["epochs"], seed=seed)
        b_best = float(max(histB["test_acc"]))
        out[m] = dict(A_final=a_final, B_final=b_final, B_best=b_best)
        print(f"  seed={seed} {m:26s} A={a_final:.3f} B_final={b_final:.3f} B_best={b_best:.3f}",
              flush=True)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = s3.REGIMES["hard"]
    per_seed = []
    for sd in SEEDS:
        print(f"=== seed {sd} ===", flush=True)
        per_seed.append(run_seed(sd, cfg))

    def agg(key):
        return {m: np.array([ps[m][key] for ps in per_seed]) for m in METHODS}
    A = agg("A_final"); Bf = agg("B_final"); Bb = agg("B_best")

    def ms(a):
        return a.mean(), (a.max() - a.min()) / 2

    lines = [f"Stage 3 verification + wide-span controls -- HARD regime, seeds {SEEDS}\n",
             "Reported accuracy = FINAL-epoch model on test (same rule for every method;\n",
             "no best-epoch/early-stopping selection). B_best = best test acc over epochs,\n",
             "shown only to check whether the noisy pan curve inflates the final number.\n",
             "chance = 0.25\n\n",
             f"{'method':<28}{'A static':>14}{'B pan (final)':>16}{'B pan (best)':>15}{'drop A->B':>11}\n"]
    for m in METHODS:
        am, ah = ms(A[m]); bfm, bfh = ms(Bf[m]); bbm, _ = ms(Bb[m])
        lines.append(f"{LABELS[m]:<28}{am:>7.3f}+/-{ah:<4.3f}{bfm:>9.3f}+/-{bfh:<4.3f}"
                     f"{bbm:>13.3f}{am-bfm:>11.3f}\n")

    lines.append("\nFreqST on Dataset B, per seed -- final vs best epoch:\n")
    for i, sd in enumerate(SEEDS):
        lines.append(f"  seed {sd}: final={per_seed[i]['freqst']['B_final']:.3f}  "
                     f"best={per_seed[i]['freqst']['B_best']:.3f}\n")

    with open(f"{OUT_DIR}/verify_metrics.txt", "w") as f:
        f.writelines(lines)
    print("\n" + "".join(lines), flush=True)

    # bar chart (A vs B_final)
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - 0.2, [A[m].mean() for m in METHODS], 0.4, label="A: static camera",
           yerr=[ms(A[m])[1] for m in METHODS], capsize=4)
    ax.bar(x + 0.2, [Bf[m].mean() for m in METHODS], 0.4, label="B: + camera pan (nuisance)",
           yerr=[ms(Bf[m])[1] for m in METHODS], capsize=4)
    ax.axhline(0.25, ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy (final epoch)"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Stage 3 (hard regime): camera-pan robustness vs. temporal span\n"
                 f"3-frame methods at increasing span, then 8-frame integration (mean over {len(SEEDS)} seeds)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/verify_pan_robustness.png", dpi=130)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/verify_pan_robustness.png", flush=True)


if __name__ == "__main__":
    main()
