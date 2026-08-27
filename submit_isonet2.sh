#!/bin/bash
#SBATCH -J isonet2
#SBATCH --partition=emgpu
#SBATCH --qos=emgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=sgp[01-04]
#SBATCH --mem=100G
#SBATCH --time=00-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o isonet2_%j.out
#SBATCH -e isonet2_%j.err
#SBATCH -D ./

set -eo pipefail
shopt -s nullglob

# =============================================================================
# USER SETTINGS
# =============================================================================

RUN_NAME="dataset"
WARP_TS="/path/to/warp_tiltseries_dataset"
MASK_SOURCE_DIR="/path/to/masks"    # expected: <prefix>.mrc or <prefix>_Vol_bmask.mrc

AC=0.07
CUBE_SIZE=96
VOLTAGE=300
EPOCHS=100
MW_WEIGHT=200
OUTPUT_NAME="CTFnetwork_box96"
WORK_ROOT="isonet2_work"

ISONET_MODULE="IsoNet/2.0.0"

# =============================================================================
# END USER SETTINGS
# =============================================================================

die() {
    echo "ERROR: $*" >&2
    exit 1
}

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$(pwd -P)" "$path"
    fi
}

WARP_TS="$(abs_path "$WARP_TS")"
MASK_SOURCE_DIR="$(abs_path "$MASK_SOURCE_DIR")"
WORK_ROOT="$(abs_path "$WORK_ROOT")"

RECON="$WARP_TS/reconstruction_miss"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$WORK_ROOT/${RUN_NAME}_${OUTPUT_NAME}_${RUN_ID}"

[[ -d "$WARP_TS" ]] || die "Warp tilt-series directory not found: $WARP_TS"
[[ -d "$RECON/odd" ]] || die "Odd half-map directory not found: $RECON/odd"
[[ -d "$RECON/even" ]] || die "Even half-map directory not found: $RECON/even"
[[ -d "$MASK_SOURCE_DIR" ]] || die "Mask directory not found: $MASK_SOURCE_DIR"
[[ ! -e "$RUN_DIR" ]] || die "Output directory already exists: $RUN_DIR"

mkdir -p "$RUN_DIR"/{odd,even,mask}
cd "$RUN_DIR"

ml purge
ml "$ISONET_MODULE"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"

find_mask_file() {
    local prefix="$1"
    local candidates=(
        "$MASK_SOURCE_DIR/${prefix}.mrc"
        "$MASK_SOURCE_DIR/${prefix}_Vol_bmask.mrc"
    )
    local candidate

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

mapfile -t ODD_FILES < <(
    find "$RECON/odd" -maxdepth 1 -type f -name "*Apx.mrc" | sort
)

((${#ODD_FILES[@]} > 0)) || die "No odd half maps matching *Apx.mrc in $RECON/odd"

DEFOCUS_A_LIST=()
PREFIXES=()
TILT_MIN_LIST=()
TILT_MAX_LIST=()

for odd_source in "${ODD_FILES[@]}"; do
    base="$(basename "$odd_source" .mrc)"
    prefix="${base%_*Apx}"

    even_source="$RECON/even/$(basename "$odd_source")"
    [[ -f "$even_source" ]] || die "Even half map not found: $even_source"

    mask_source="$(find_mask_file "$prefix")" || die "Could not locate mask for $prefix"
    xml_file="$WARP_TS/${prefix}.xml"

    [[ -f "$xml_file" ]] || die "Warp XML not found: $xml_file"

    ln -s "$(realpath "$odd_source")" "odd/${prefix}_ODD_Vol.mrc"
    ln -s "$(realpath "$even_source")" "even/${prefix}_EVN_Vol.mrc"
    ln -s "$(realpath "$mask_source")" "mask/${prefix}_Vol_bmask.mrc"

    defocus_um="$(
        sed -n 's/.*Name="Defocus" Value="\([^"]*\)".*/\1/p' "$xml_file" |
        head -n 1
    )"

    [[ -n "$defocus_um" ]] || die "Could not read Defocus from $xml_file"

    defocus_angstrom="$(
        awk -v defocus="$defocus_um" 'BEGIN {printf "%.2f", defocus * 10000.0}'
    )"

    read -r tomo_tilt_min tomo_tilt_max < <(
        python - "$xml_file" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()

angles = []
for node in root.findall(".//Angles"):
    angles.extend(float(value) for value in (node.text or "").split())

use_tilt = []
for node in root.findall(".//UseTilt"):
    use_tilt.extend(
        value.lower() in {"true", "1"}
        for value in (node.text or "").split()
    )

if use_tilt:
    if len(use_tilt) != len(angles):
        raise SystemExit(
            f"UseTilt/Angles length mismatch in {sys.argv[1]}: "
            f"{len(use_tilt)} versus {len(angles)}"
        )
    angles = [angle for angle, use in zip(angles, use_tilt) if use]

if not angles:
    raise SystemExit(f"No enabled tilt angles found in {sys.argv[1]}")

# Warp stores tilt angles in the opposite convention from IMOD/PyTom.
angles = [-angle for angle in angles]
print(f"{min(angles):.2f} {max(angles):.2f}")
PY
    )

    DEFOCUS_A_LIST+=("$defocus_angstrom")
    PREFIXES+=("$prefix")
    TILT_MIN_LIST+=("$tomo_tilt_min")
    TILT_MAX_LIST+=("$tomo_tilt_max")
done

defocus_csv="$(IFS=,; echo "${DEFOCUS_A_LIST[*]}")"

echo "IsoNet2 prefixes:"
for i in "${!PREFIXES[@]}"; do
    printf '  %s: tilt_min=%s tilt_max=%s\n' \
        "${PREFIXES[$i]}" "${TILT_MIN_LIST[$i]}" "${TILT_MAX_LIST[$i]}"
done
echo "defocus_A: [$defocus_csv]"
echo "AC:        $AC"
echo "cube_size: $CUBE_SIZE"
echo "GPU IDs:   $GPU_IDS"
echo "run dir:   $RUN_DIR"

STAR_FILE="tomograms_${OUTPUT_NAME}.star"
NETWORK_DIR="isonet2_out_${OUTPUT_NAME}"

isonet.py prepare_star \
    --even even/ \
    --odd odd/ \
    --mask_folder mask/ \
    --tilt_min "${TILT_MIN_LIST[0]}" \
    --tilt_max "${TILT_MAX_LIST[0]}" \
    --defocus "[$defocus_csv]" \
    --ac "$AC" \
    --voltage "$VOLTAGE" \
    --star_name "$STAR_FILE"

TILT_RANGE_FILE="tilt_ranges.tsv"
for i in "${!PREFIXES[@]}"; do
    printf '%s\t%s\t%s\n' \
        "${PREFIXES[$i]}" "${TILT_MIN_LIST[$i]}" "${TILT_MAX_LIST[$i]}"
done > "$TILT_RANGE_FILE"

python - "$STAR_FILE" "$TILT_RANGE_FILE" <<'PY'
import sys
from pathlib import Path

import starfile

star_path = sys.argv[1]
ranges = {}

with open(sys.argv[2], encoding="utf-8") as handle:
    for line in handle:
        prefix, tilt_min, tilt_max = line.rstrip("\n").split("\t")
        ranges[prefix] = (float(tilt_min), float(tilt_max))

table = starfile.read(star_path)

for index, row in table.iterrows():
    name = Path(
        str(row["rlnTomoReconstructedTomogramHalf1"])
    ).name

    suffix = "_EVN_Vol.mrc"

    if not name.endswith(suffix):
        raise SystemExit(
            f"Unexpected even-half filename in {star_path}: {name}"
        )

    prefix = name[:-len(suffix)]

    if prefix not in ranges:
        raise SystemExit(f"No XML tilt range found for {prefix}")

    table.at[index, "rlnTiltMin"] = ranges[prefix][0]
    table.at[index, "rlnTiltMax"] = ranges[prefix][1]

starfile.write(table, star_path)
PY

time isonet.py refine \
    "$STAR_FILE" \
    -o "$NETWORK_DIR" \
    --method isonet2-n2n \
    --cube_size "$CUBE_SIZE" \
    --epochs "$EPOCHS" \
    --mw_weight "$MW_WEIGHT" \
    --CTF_mode network \
    --clip_first_peak_mode 1 \
    --isCTFflipped True \
    --gpuID "$GPU_IDS"

MODEL="$NETWORK_DIR/network_isonet2-n2n_unet-medium_${CUBE_SIZE}_full.pt"
[[ -f "$MODEL" ]] || die "Expected trained model not found: $MODEL"

time isonet.py predict \
    "$STAR_FILE" \
    "$MODEL" \
    --gpuID "$GPU_IDS"

echo "IsoNet2 output written to: $RUN_DIR"
