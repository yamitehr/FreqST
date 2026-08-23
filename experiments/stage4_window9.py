"""Stage 4 -- same-window fairness check.

Stage 3 already feeds every method the same WINDOW=8 span, but GrayST and
TC Reordering only ever read 3 of those 8 raw frames (see transforms.py) --
so "same window" and "same amount of frame content consumed" are not the
same thing. This script asks the literal question: if every method is
handed the exact same T=9 window, what does each one do with it?

  rgb            : ignores window entirely (always the clip's mid frame) --
                   appearance-only control, included for reference.
  grayst         : last 3 consecutive frames of the 9      (frames 6,7,8)
                   -- frames 0-5 are read by NOTHING, discarded.
  tc_reordering  : 3 frames spread across the 9            (frames 0,4,8)
                   -- frames 1,2,3,5,6,7 are read by NOTHING, discarded.
  chunk_avg      : split the 9 into 3 contiguous groups of 3, average each
                   group into one channel (frames 0-2 / 3-5 / 6-8) -- every
                   frame contributes to exactly one channel, none discarded,
                   but still no frequency decomposition (like GrayST/TC, just
                   averaged instead of a single raw snapshot per channel).
  tavg           : average of all 9 frames -> 3 identical channels
  freqst         : DCT of all 9 frames -> 3 coefficient channels (DC+2 AC)

Reuses the exact same underlying clips as Stage 3's hard regime (identical
seeds), just re-sliced to a 9-frame window instead of 8 -- so any
difference vs results/stage3/metrics.txt's hard-regime numbers is coming
only from the window-size change, not from different data.

Single seed, hard regime, both Dataset A (static) and Dataset B (pan).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiments.stage3 as s3

OUT_DIR = "results/stage4_window9"
WINDOW_T = 9
METHODS = ["rgb", "grayst", "tc_reordering", "chunk_avg", "tavg", "freqst"]
LABELS = {
    "rgb": "RGB (1 frame, no window)",
    "grayst": "GrayST (3f of 9, consecutive)",
    "tc_reordering": "TC Reordering (3f of 9, spread)",
    "chunk_avg": "Chunk-avg (3 groups of 9, none dropped)",
    "tavg": "Temporal avg (9f)",
    "freqst": "FreqST (9f DCT)",
}
FRAMES_USED = {
    "rgb": "1 (mid frame; window ignored)",
    "grayst": "3 of 9 (6 discarded)",
    "tc_reordering": "3 of 9 (6 discarded)",
    "chunk_avg": "9 of 9 (0 discarded)",
    "tavg": "9 of 9 (0 discarded)",
    "freqst": "9 of 9 (0 discarded)",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = s3.REGIMES["hard"]

    print(f"Building Dataset A/B (hard regime, same seeds as stage3) ...", flush=True)
    A_tr, Aytr = s3.build_dataset(cfg["n_train"], False, 1, cfg)
    A_va, Ayva = s3.build_dataset(s3.N_VAL, False, 2, cfg)
    A_te, Ayte = s3.build_dataset(s3.N_TEST, False, 3, cfg)
    B_tr, Bytr = s3.build_dataset(cfg["n_train"], True, 4, cfg)
    B_va, Byva = s3.build_dataset(s3.N_VAL, True, 5, cfg)
    B_te, Byte = s3.build_dataset(s3.N_TEST, True, 6, cfg)

    results = {}
    for m in METHODS:
        Atr_x = s3.make_inputs(A_tr, m, WINDOW_T)
        Ava_x = s3.make_inputs(A_va, m, WINDOW_T)
        Ate_x = s3.make_inputs(A_te, m, WINDOW_T)
        Atr_x, Ava_x, Ate_x = s3.normalize(Atr_x, Ava_x, Ate_x)
        _, a_acc, _ = s3.train_one(Atr_x, Aytr, Ava_x, Ayva, Ate_x, Ayte,
                                    cfg["base_ch"], cfg["epochs"])

        Btr_x = s3.make_inputs(B_tr, m, WINDOW_T)
        Bva_x = s3.make_inputs(B_va, m, WINDOW_T)
        Bte_x = s3.make_inputs(B_te, m, WINDOW_T)
        Btr_x, Bva_x, Bte_x = s3.normalize(Btr_x, Bva_x, Bte_x)
        _, b_acc, _ = s3.train_one(Btr_x, Bytr, Bva_x, Byva, Bte_x, Byte,
                                    cfg["base_ch"], cfg["epochs"])

        results[m] = (a_acc, b_acc)
        print(f"  {m:16s} A={a_acc:.3f}  B={b_acc:.3f}", flush=True)

    lines = [f"Stage 4 -- same-window (T={WINDOW_T}) fairness check, hard regime, single seed\n",
             f"Same underlying clips as stage3.py's hard regime (same build_dataset seeds),\n",
             f"re-sliced to a {WINDOW_T}-frame window for every method. chance=0.25\n\n",
             f"{'method':<40}{'frames used':<24}{'A(static)':>11}{'B(pan)':>10}{'drop A->B':>11}\n"]
    for m in METHODS:
        a, b = results[m]
        lines.append(f"{LABELS[m]:<40}{FRAMES_USED[m]:<24}{a:>11.3f}{b:>10.3f}{a-b:>11.3f}\n")

    with open(f"{OUT_DIR}/metrics.txt", "w") as f:
        f.writelines(lines)
    print("\n" + "".join(lines), flush=True)

    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, [results[m][0] for m in METHODS], 0.4, label="A: static camera")
    ax.bar(x + 0.2, [results[m][1] for m in METHODS], 0.4, label="B: + camera pan")
    ax.axhline(0.25, ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Stage 4: same {WINDOW_T}-frame window, all methods (hard regime, 1 seed)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/window9_comparison.png", dpi=130)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
