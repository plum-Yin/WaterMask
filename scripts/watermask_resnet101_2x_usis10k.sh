#!/usr/bin/env bash


SEED="${SEED:-1884018747}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29136}"

# Prevent OpenMP/BLAS oversubscription in each distributed worker.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATERMASK_DIR="$(dirname "${SCRIPT_DIR}")"
CONFIG="project/WaterMask/configs/watermask_resnet101_2x_usis10k.py"
WORK_DIR="${WORK_DIR:-work_dirs/watermask_resnet101_2x_usis10k}"
DATA_ROOT="${DATA_ROOT:-/data/zyin418/Underwater/ViT-UWA/datasets/USIS10K}"
ANN_DIR="${DATA_ROOT}/multi_class_annotations"

for required_file in \
    "${ANN_DIR}/multi_class_train_annotations.json" \
    "${ANN_DIR}/multi_class_test_annotations.json"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required dataset annotation not found: ${required_file}" >&2
        exit 1
    fi
done

source "${CONDA_SH:-/data/zyin418/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-usis}"
cd "${WATERMASK_DIR}"

CFG_OPTIONS=(
    "randomness.seed=${SEED}"
    "randomness.deterministic=False"
    "train_dataloader.dataset.data_root=${DATA_ROOT}/"
    "val_dataloader.dataset.data_root=${DATA_ROOT}/"
    "test_dataloader.dataset.data_root=${DATA_ROOT}/"
    "val_evaluator.ann_file=${ANN_DIR}/multi_class_test_annotations.json"
    "test_evaluator.ann_file=${ANN_DIR}/multi_class_test_annotations.json"
)

# Run a one-sample train/validation cycle when checking the environment.
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    CFG_OPTIONS+=(
        "train_cfg.max_epochs=1"
        "train_dataloader.dataset.indices=[0]"
        "train_dataloader.num_workers=0"
        "train_dataloader.persistent_workers=False"
        "val_dataloader.dataset.indices=[0]"
        "val_dataloader.num_workers=0"
        "val_dataloader.persistent_workers=False"
        "default_hooks.logger.interval=1"
    )
fi

python -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    tools/train.py "${CONFIG}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR}" \
    --resume \
    "$@" \
    --cfg-options "${CFG_OPTIONS[@]}"
