#!/usr/bin/env bash

#SBATCH --job-name=wm101usis
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=truncate

SEED="${SEED:-1884018747}"
GPUS="${GPUS:-1}"
CONDA_ENV="${CONDA_ENV:-watermask}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /data/zyin418/Underwater/WaterMask
source /data/zyin418/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

python -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="29235" \
    tools/train.py \
    configs/_our_/water_r101_fpn_3x_usis10k.py \
    --launcher pytorch \
    --seed "${SEED}" \
    --deterministic \
    --work-dir work_dirs/water_r101_fpn_3x_usis10k \
    --auto-resume \
    "$@"
