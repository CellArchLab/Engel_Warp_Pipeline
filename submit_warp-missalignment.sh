#!/bin/bash
#SBATCH -o warp_miss_%j.out
#SBATCH -e warp_miss_%j.err
#SBATCH -D ./
#SBATCH -J warp_miss
#SBATCH --partition=emgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --qos=emgpu
#SBATCH --mail-type=none

set -eo pipefail
shopt -s nullglob
# =============================================================================
# USER SETTINGS
# =============================================================================
RUN_NAME="dataset"
DATA_FOLDER="/path/to/raw_frames_and_mdocs"
EXTENSION="tiff"                    # eer | tiff | tif | mrc
ANGPIX="1.912"                      # unbinned acquisition pixel size (A/px)
PRE_BIN="1"                         # frame-series pre-binning during import
GAIN_PATH=""                        # optional; leave empty for no gain correction
EXPOSURE="2.22"                     # dose per tilt (e-/A^2)
TILT_AXIS="-84.25"                  # leave empty to use the mdoc value
EER_NGROUPS="16"
GAIN_FLIP_Y="false"
AT_BINNING="8"                      # QC/alignment reconstruction binning
MA_BINNING="4"                      # post-MissAlignment reconstruction binning
ALIGNZ="2200"
MIN_INTENSITY="0.90"
MIN_FOV="0.00"
USE_ETOMO="true"
USE_ARETOMO3="false"
PATCH_SIZE="2000"
DELETE_INTERMEDIATE="false"
AT_PATCHES="0,0"
AUTOZERO="true"
AUTOLEVEL="false"
TILT_OFFSET="-10"
AUTOLEVEL_PATCH_SIZE="1500"
PERDEVICE="1"
DEFOCUS_HAND="set_auto"             # check | set_auto | set_flip | set_noflip
M_GRID="1x1x3"
C_GRID="2x2x1"
C_RANGE_MAX="5"
C_RANGE_MIN="40"
C_DEFOCUS_MAX="7.5"
C_DEFOCUS_MIN="0.5"
TOMO_DIMENSIONS="4096x4096x2048"
AXIS_ITER="0"
TS_RANGE_LOW="40"
TS_RANGE_HIGH="9"
TS_DEFOCUS_MAX="7.5"
TS_DEFOCUS_MIN="0.5"
OUT_AVERAGE_HALVES="true"
HALFMAP_FRAMES="true"

MISS_TRAIN_DEV="0,1"
MISS_RECON_DEV="2,2,2,3,3,3"
MISS_DATALOADERS="5"
MISS_PREPARE_STACKS="10.0"
MISS_NCCL_P2P_DISABLE="false"
WARP_MODULE="WarpM"
IMOD_MODULE="IMOD"
ARETOMO2_MODULE="AreTomo2"
ARETOMO3_MODULE="AreTomo3"
MISS_MODULE="MissAlignment"
# =============================================================================
# END USER SETTINGS
# =============================================================================

die() {
    echo "ERROR: $*" >&2
    exit 1
}

backup_if_exists() {
    local target="$1"
    if [[ -e "$target" || -L "$target" ]]; then
        local backup="${target}.backup_$(date +%Y%m%d_%H%M%S)"
        echo "Preserving existing path: $target -> $backup"
        mv -- "$target" "$backup"
    fi
}

[[ -d "$DATA_FOLDER" ]] || die "DATA_FOLDER not found: $DATA_FOLDER"
[[ "$PRE_BIN" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "PRE_BIN must be numeric"

if [[ "$USE_ARETOMO3" == "true" ]]; then
    ARETOMO_MODULE="$ARETOMO3_MODULE"
else
    ARETOMO_MODULE="$ARETOMO2_MODULE"
fi

ml purge
ml "$WARP_MODULE"
ml "$IMOD_MODULE"
ml "$ARETOMO_MODULE"
export no_proxy=localhost,127.0.0.1

GAIN_ARGS=()

if [[ -n "$GAIN_PATH" ]]; then
    [[ -f "$GAIN_PATH" ]] || die "GAIN_PATH not found: $GAIN_PATH"

    case "${GAIN_PATH,,}" in
        *.mrc)
            ;;
        *)
            GAIN_MRC="${GAIN_PATH%.*}.mrc"
            if [[ ! -f "$GAIN_MRC" ]]; then
                echo "Converting gain to MRC: $GAIN_PATH -> $GAIN_MRC"
                tif2mrc "$GAIN_PATH" "$GAIN_MRC"
            fi
            GAIN_PATH="$GAIN_MRC"
            ;;
    esac

    GAIN_ARGS+=(--gain_path "$GAIN_PATH")
    [[ "$GAIN_FLIP_Y" == "true" ]] && GAIN_ARGS+=(--gain_flip_y)
else
    echo "No gain reference supplied; Warp will process without gain correction."
fi

FS_DIR="warp_frameseries_${RUN_NAME}"
TS_DIR="warp_tiltseries_${RUN_NAME}"
TOMOSTAR_DIR="tomostar_${RUN_NAME}"
FS_SETTINGS="${FS_DIR}.settings"
TS_SETTINGS="${TS_DIR}.settings"

AT_RECON_ANGPIX=$(awk -v a="$ANGPIX" -v b="$AT_BINNING" 'BEGIN {printf "%g", a*b}')
MA_RECON_ANGPIX=$(awk -v a="$ANGPIX" -v b="$MA_BINNING" 'BEGIN {printf "%g", a*b}')

TOMO_X="${TOMO_DIMENSIONS%%x*}"
_rest="${TOMO_DIMENSIONS#*x}"
TOMO_Y="${_rest%%x*}"
TOMO_Z="${_rest#*x}"

EER_ARGS=()
[[ "${EXTENSION,,}" == "eer" ]] && EER_ARGS+=(--eer_ngroups "$EER_NGROUPS")

IMPORT_GEOMETRY_ARGS=()
if [[ -n "$TILT_AXIS" ]]; then
    IMPORT_GEOMETRY_ARGS+=(--override_axis "$TILT_AXIS")
    ARETOMO_AXIS_ARGS=(--axis "$TILT_AXIS")
else
    ARETOMO_AXIS_ARGS=()
    echo "WARNING: TILT_AXIS is empty; Warp will use the mdoc value."
fi
if [[ "$AUTOZERO" == "true" ]]; then
    IMPORT_GEOMETRY_ARGS+=(--auto_zero)
elif [[ -n "$TILT_OFFSET" ]]; then
    IMPORT_GEOMETRY_ARGS+=(--tilt_offset "$TILT_OFFSET")
fi

MOTION_HALF_ARGS=()
[[ "$OUT_AVERAGE_HALVES" == "true" ]] && MOTION_HALF_ARGS+=(--out_average_halves)

RECON_HALF_ARGS=()
[[ "$HALFMAP_FRAMES" == "true" ]] && RECON_HALF_ARGS+=(--halfmap_frames)

DELETE_ARGS=()
[[ "$DELETE_INTERMEDIATE" == "true" ]] && DELETE_ARGS+=(--delete_intermediate)

echo "Job ID: ${SLURM_JOB_ID:-none} | Node: ${SLURMD_NODENAME:-local} | PWD: $PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v WarpTools
nvidia-smi -L
mkdir -p "$FS_DIR" "$TOMOSTAR_DIR" "$TS_DIR"

# 1. Frame-series settings
WarpTools create_settings \
    --folder_data "$DATA_FOLDER" \
    --folder_processing "$FS_DIR" \
    --output "$FS_SETTINGS" \
    --extension "*.${EXTENSION}" \
    --angpix "$ANGPIX" \
    --bin "$PRE_BIN" \
    "${GAIN_ARGS[@]}" \
    "${EER_ARGS[@]}" \
    --exposure "$EXPOSURE"

# 2. Motion correction and frame CTF
time WarpTools fs_motion_and_ctf \
    --settings "$FS_SETTINGS" \
    --m_grid "$M_GRID" \
    --c_grid "$C_GRID" \
    --c_range_max "$C_RANGE_MAX" \
    --c_range_min "$C_RANGE_MIN" \
    --c_defocus_max "$C_DEFOCUS_MAX" \
    --c_defocus_min "$C_DEFOCUS_MIN" \
    --c_use_sum \
    --out_averages \
    "${MOTION_HALF_ARGS[@]}" \
    --perdevice "$PERDEVICE"

WarpTools filter_quality --settings "$FS_SETTINGS" --histograms

# 3. Import mdocs
time WarpTools ts_import \
    --mdocs "$DATA_FOLDER" \
    --frameseries "$FS_DIR" \
    --tilt_exposure "$EXPOSURE" \
    --min_intensity "$MIN_INTENSITY" \
    "${IMPORT_GEOMETRY_ARGS[@]}" \
    --output "$TOMOSTAR_DIR"

# 4. Tilt-series settings
WarpTools create_settings \
    --output "$TS_SETTINGS" \
    --folder_processing "$TS_DIR" \
    --folder_data "$TOMOSTAR_DIR" \
    --extension "*.tomostar" \
    --angpix "$ANGPIX" \
    "${GAIN_ARGS[@]}" \
    --exposure "$EXPOSURE" \
    "${EER_ARGS[@]}" \
    --tomo_dimensions "$TOMO_DIMENSIONS"

# 5. Initial alignment
if [[ "$USE_ETOMO" == "true" ]]; then
    REC_SUFFIX="etomo"
    time WarpTools ts_etomo_patches \
        --settings "$TS_SETTINGS" \
        --angpix "$AT_RECON_ANGPIX" \
        --patch_size "$PATCH_SIZE" \
        --min_fov "$MIN_FOV" \
        "${DELETE_ARGS[@]}" \
        --perdevice "$PERDEVICE"
else
    REC_SUFFIX="aretomo"
    if [[ "$USE_ARETOMO3" == "true" ]]; then
        ALIGNZ_ARGS=()
        [[ "$ALIGNZ" != "0" ]] && ALIGNZ_ARGS=(--alignz "$ALIGNZ")
        time WarpTools ts_aretomo3 \
            --settings "$TS_SETTINGS" \
            --angpix "$AT_RECON_ANGPIX" \
            "${ALIGNZ_ARGS[@]}" \
            --axis_iter "$AXIS_ITER" \
            "${ARETOMO_AXIS_ARGS[@]}" \
            --perdevice "$PERDEVICE" \
            "${DELETE_ARGS[@]}" \
            --min_fov "$MIN_FOV" \
            --patches "$AT_PATCHES"
    else
        time WarpTools ts_aretomo \
            --settings "$TS_SETTINGS" \
            --angpix "$AT_RECON_ANGPIX" \
            --alignz "$ALIGNZ" \
            --axis_iter "$AXIS_ITER" \
            "${ARETOMO_AXIS_ARGS[@]}" \
            --perdevice "$PERDEVICE" \
            "${DELETE_ARGS[@]}" \
            --min_fov "$MIN_FOV" \
            --patches "$AT_PATCHES"
    fi
fi

if [[ "$AUTOLEVEL" == "true" ]]; then
    time WarpTools ts_autolevel \
        --settings "$TS_SETTINGS" \
        --angpix "$AT_RECON_ANGPIX" \
        --patch_size "$AUTOLEVEL_PATCH_SIZE"
fi

# 6. Defocus handedness and tilt-series CTF
WarpTools ts_defocus_hand --settings "$TS_SETTINGS" "--${DEFOCUS_HAND}"

time WarpTools ts_ctf \
    --settings "$TS_SETTINGS" \
    --range_low "$TS_RANGE_LOW" \
    --range_high "$TS_RANGE_HIGH" \
    --defocus_max "$TS_DEFOCUS_MAX" \
    --defocus_min "$TS_DEFOCUS_MIN"

TOMOSTARS=("$TOMOSTAR_DIR"/*.tomostar)
((${#TOMOSTARS[@]} > 0)) || die "No tomostar files found in $TOMOSTAR_DIR"
for tomostar in "${TOMOSTARS[@]}"; do
    awk 'NF > 2 {print -1*$2}' "$tomostar" > "${tomostar%.*}.tlt"
done

# 7. QC reconstruction
backup_if_exists "$TS_DIR/reconstruction"
backup_if_exists "$TS_DIR/reconstruction_${REC_SUFFIX}"

time WarpTools ts_reconstruct \
    --settings "$TS_SETTINGS" \
    --angpix "$AT_RECON_ANGPIX" \
    --dont_invert

[[ -d "$TS_DIR/reconstruction" ]] || die "Warp QC reconstruction was not created"
mv -- "$TS_DIR/reconstruction" "$TS_DIR/reconstruction_${REC_SUFFIX}"

# 8. Miss-Alignment helper files
TS_ABS="$(cd "$TS_DIR" && pwd -P)"

cat > "$TS_DIR/update_warp_xml.py" <<PY
import torch
from pathlib import Path
from warpylib import TiltSeries

original_stack_shape = (${TOMO_X}, ${TOMO_Y})
volume_shape = (${TOMO_X}, ${TOMO_Y}, ${TOMO_Z})
original_pixel_size = ${ANGPIX}

for xml_path in Path(".").glob("*.xml"):
    ts = TiltSeries(xml_path)
    ts.image_dimensions_physical = torch.tensor(
        [
            original_stack_shape[0] * original_pixel_size,
            original_stack_shape[1] * original_pixel_size,
        ],
        dtype=torch.float32,
    )
    ts.volume_dimensions_physical = torch.tensor(
        [
            volume_shape[0] * original_pixel_size,
            volume_shape[1] * original_pixel_size,
            volume_shape[2] * original_pixel_size,
        ],
        dtype=torch.float32,
    )
    ts.save_meta(xml_path)
PY

cat > "$TS_DIR/config_template.yml" <<YAML
general:
  training_directory: ${TS_ABS}/
  apply_ctf: False
  iteration_settings:
    - {downsample: 3, alignment: anchoring}
    - {downsample: 2, alignment: anchoring}
    - {downsample: 1, alignment: global}
    - {downsample: 1, alignment: global}
    - {downsample: 1, alignment: [3, 3]}
    - {downsample: 1, alignment: [3, 3]}
    - {downsample: 1, alignment: [3, 3]}
    - {downsample: 1, alignment: [3, 3]}
  seed: 45132
model_training:
  model_architecture: default
  model_checkpoint: null
  loss_margin: 0.5
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  max_epochs_per_iteration: 30
  warmup_steps: 500
  multistep_lr_scheduler:
    milestones: [5, 15]
    gamma: 0.5
data_loading:
  batch_size: 32
  patch_size: 96
  steps_per_epoch: 1000
shift_generation:
  trajectory_probability: 0.5
  trajectory_max_shift: 10.0
  jitter_probability: 0.5
  jitter_max_std: 2.0
  outlier_probability: 0.5
  outlier_max_shift: 20.0
  fracture_probability: 0.5
  fracture_max_shift: 20.0
tilt_series_alignment:
  patch_size: 96
  patch_overlap: 0.1
  batch_size: 32
YAML

NCCL_LINE=""
[[ "$MISS_NCCL_P2P_DISABLE" == "true" ]] && NCCL_LINE="export NCCL_P2P_DISABLE=1"

cat > "$TS_DIR/submit_missalignment.sh" <<MISS
#!/bin/bash
#SBATCH --job-name=miss_${RUN_NAME}
#SBATCH --output=miss_%j.out
#SBATCH --error=miss_%j.err
#SBATCH --partition=emgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --qos=emgpu

set -Eeuo pipefail

ml purge
ml "${MISS_MODULE}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
${NCCL_LINE}
unset SLURM_NTASKS_PER_NODE

python update_warp_xml.py

miss-alignment train \
    --config-file config_template.yml \
    --training-devices "${MISS_TRAIN_DEV}" \
    --reconstruction-devices "${MISS_RECON_DEV}" \
    --dataloaders-per-trainer "${MISS_DATALOADERS}" \
    --start-at-iteration 0 \
    --prepare-stacks "${MISS_PREPARE_STACKS}"
MISS

chmod +x "$TS_DIR/submit_missalignment.sh"

# 9. Post-MissAlignment CTF and reconstruction
cat > "submit_post_miss_${RUN_NAME}.sh" <<POST
#!/bin/bash
#SBATCH -o ${RUN_NAME}_post_miss_%j.out
#SBATCH -e ${RUN_NAME}_post_miss_%j.err
#SBATCH -D ./
#SBATCH -J ${RUN_NAME}_miss_recon
#SBATCH --partition=emgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --qos=emgpu

set -Eeuo pipefail

backup_if_exists() {
    local target="\$1"
    if [[ -e "\$target" || -L "\$target" ]]; then
        local backup="\${target}.backup_\$(date +%Y%m%d_%H%M%S)"
        echo "Preserving existing path: \$target -> \$backup"
        mv -- "\$target" "\$backup"
    fi
}

ml purge
ml "${WARP_MODULE}"
ml "${IMOD_MODULE}"
ml "${ARETOMO_MODULE}"
export no_proxy=localhost,127.0.0.1

backup_if_exists "${TS_DIR}/reconstruction"
backup_if_exists "${TS_DIR}/reconstruction_miss"

WarpTools ts_ctf \
    --settings "${TS_SETTINGS}" \
    --range_low "${TS_RANGE_LOW}" \
    --range_high "${TS_RANGE_HIGH}" \
    --defocus_max "${TS_DEFOCUS_MAX}" \
    --defocus_min "${TS_DEFOCUS_MIN}"

WarpTools filter_quality --settings "${TS_SETTINGS}" --histograms

WarpTools ts_reconstruct \
    --settings "${TS_SETTINGS}" \
    --angpix "${MA_RECON_ANGPIX}" \
    ${RECON_HALF_ARGS[*]} \
    --dont_invert

[[ -d "${TS_DIR}/reconstruction" ]] || {
    echo "ERROR: post-MissAlignment reconstruction was not created" >&2
    exit 1
}

mv -- "${TS_DIR}/reconstruction" "${TS_DIR}/reconstruction_miss"
ln -sfn "reconstruction_miss" "${TS_DIR}/reconstruction"

echo "Post-MissAlignment output: ${TS_DIR}/reconstruction_miss"
echo "Compatibility symlink:    ${TS_DIR}/reconstruction -> reconstruction_miss"
POST

chmod +x "submit_post_miss_${RUN_NAME}.sh"

# 10. Submit dependency chain
MISS_JID=$(
    cd "$TS_DIR"
    sbatch --parsable submit_missalignment.sh
)
echo "Submitted Miss-Alignment job: $MISS_JID"

POST_JID=$(
    sbatch --parsable \
        --dependency="afterok:${MISS_JID}" \
        "submit_post_miss_${RUN_NAME}.sh"
)
echo "Submitted post-MissAlignment reconstruction: $POST_JID"

echo "Warp processing complete; Miss-Alignment chain submitted."
echo "QC reconstruction:        $TS_DIR/reconstruction_${REC_SUFFIX}"
echo "Final reconstruction:     $TS_DIR/reconstruction_miss"
