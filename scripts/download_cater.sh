#!/usr/bin/env bash
#
# Download the CATER dataset (task 2, compositional actions, 301 classes).
# Fetches BOTH camera variants:
#   - max2action                = static camera
#   - max2action_cameramotion   = moving camera (the pilot's cameramotion split)
#
# Train / val splits ship inside lists.zip as lists/actions_order_uniq/{train,val}.txt.
# CATER has no separate "test" split — the paper reports val mAP as its test metric,
# matching Kim et al.'s protocol.
#
# Usage:
#   scripts/download_cater.sh [DATA_ROOT]
#     DATA_ROOT defaults to $HOME/data/cater
#
# The script is idempotent: re-running skips any variant/component already extracted
# (marked by a .done sentinel). Partial curl downloads resume via -C -.
#
# Post-conditions (on success):
#   $DATA_ROOT/
#     downloads/                          <- zip cache (safe to delete after extract)
#     max2action/{videos,scenes,lists}/    <- created by each zip's own top-level prefix
#     max2action_cameramotion/{videos,scenes,lists}/
#
# Note on extract layout: each zip ships with its own top-level directory
# (videos/, scenes/, or lists/), so we extract into $DATA_ROOT/<variant>/ and let
# the zip's structure create the component subdir. Per-component sentinels go in
# $DATA_ROOT/<variant>/.<component>_done to keep the script idempotent.
#
# Next step after this script: extract per-clip JPEG frames from each variant's
# videos/*.avi (Kim et al.'s pipeline expects 300 JPEGs/clip), then run
#   python third_party/channel_sampling/tools/datasets/generate_cater_task1_2_splits.py \
#     $DATA_ROOT/max2action_cameramotion
# to produce the CSV splits PyVideoAI's dataloader reads.
#
# Source: https://github.com/rohitgirdhar/CATER (generate/README.md direct links).
# License: CC-BY 4.0 (per the CATER GitHub repo).

set -euo pipefail

DATA_ROOT="${1:-$HOME/data/cater}"
DOWNLOAD_DIR="$DATA_ROOT/downloads"
mkdir -p "$DOWNLOAD_DIR"

# Layout: (variant, component, url)
# Direct URLs from https://github.com/rohitgirdhar/CATER/blob/master/generate/README.md
entries=(
  "max2action|videos|https://cmu.box.com/shared/static/jgbch9enrcfvxtwkrqsdbitwvuwnopl0.zip"
  "max2action|scenes|https://cmu.box.com/shared/static/922x4qs3feynstjj42muecrlch1o7pmv.zip"
  "max2action|lists|https://cmu.box.com/shared/static/7svgta3kqat1jhe9kp0zuptt3vrvarzw.zip"
  "max2action_cameramotion|videos|https://cmu.box.com/shared/static/yvhx9p5haip5abzh9i2fofssjpq34zwz.zip"
  "max2action_cameramotion|scenes|https://cmu.box.com/shared/static/zfau8j1e6n7ylobf0g1d2wjdgdu86j2e.zip"
  "max2action_cameramotion|lists|https://cmu.box.com/shared/static/i9kexj33if00t338esnw93uzm5f6sfar.zip"
)

download_and_extract() {
  local variant="$1"
  local component="$2"
  local url="$3"

  local zip_path="$DOWNLOAD_DIR/${variant}_${component}.zip"
  local variant_dir="$DATA_ROOT/$variant"
  local sentinel="$variant_dir/.${component}_done"
  local expected_dir="$variant_dir/$component"

  if [ -f "$sentinel" ]; then
    echo "[skip]     $variant/$component (already extracted)"
    return 0
  fi

  echo "[download] $variant/$component"
  echo "           <- $url"
  curl -L -C - --fail --retry 3 --retry-delay 5 -o "$zip_path" "$url"

  echo "[extract]  $variant/$component -> $variant_dir/  (zip's own $component/ prefix creates the subdir)"
  mkdir -p "$variant_dir"
  unzip -q -o "$zip_path" -d "$variant_dir"

  if [ ! -d "$expected_dir" ]; then
    echo "[ERROR]    Expected $expected_dir after extract, not found."
    echo "           Zip layout may have changed upstream; inspect $zip_path manually."
    return 1
  fi

  touch "$sentinel"
  echo
}

echo "CATER download starting."
echo "  Destination: $DATA_ROOT"
echo "  Cache:       $DOWNLOAD_DIR"
echo

for entry in "${entries[@]}"; do
  IFS='|' read -r variant component url <<< "$entry"
  download_and_extract "$variant" "$component" "$url"
done

echo "=================================================================="
echo "All variants downloaded and extracted."
echo
echo "Disk usage:"
du -sh "$DATA_ROOT"/* 2>/dev/null | sort -k2
echo
echo "Verify train/val split files (task 2 = actions_order_uniq, task 1 = actions_present):"
for variant in max2action max2action_cameramotion; do
  for task_dir in actions_order_uniq actions_present; do
    for split in train val; do
      path="$DATA_ROOT/$variant/lists/$task_dir/${split}.txt"
      if [ -f "$path" ]; then
        n_lines=$(wc -l < "$path")
        printf "  %-28s %-20s %-6s -> %s lines\n" "$variant" "$task_dir" "$split" "$n_lines"
      else
        printf "  %-28s %-20s %-6s -> MISSING (%s)\n" "$variant" "$task_dir" "$split" "$path"
      fi
    done
  done
done
echo
echo "Next: extract per-clip JPEGs from videos/*.avi for each variant, then run"
echo "  python third_party/channel_sampling/tools/datasets/generate_cater_task1_2_splits.py \\"
echo "    $DATA_ROOT/max2action_cameramotion"
echo "(and the same for max2action for the static-camera split later)."
