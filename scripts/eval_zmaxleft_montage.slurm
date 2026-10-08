#!/bin/bash
#SBATCH --job-name=zl-montage
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:30:00
#SBATCH --output=slurm_outputs/zl-montage-%j.out

# Same allocation as eval_zmaxleft_video.slurm; add CPU montage assembly.
set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
: "${PKL:?Set PKL to the checkpoint}"
: "${CONFIG:?Set CONFIG to the matching eval config}"
: "${OUT_DIR:?Set OUT_DIR to a fresh render directory}"
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp
uv run --no-sync python scripts/eval_zmaxleft_video.py \
    --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet \
    --state-pkl "$PKL" --config "$CONFIG" --out-dir "$OUT_DIR"
uv run --no-sync python scripts/montage_zmaxleft.py \
    --input-dir "$OUT_DIR" --output-prefix "$OUT_DIR/montage" \
    --title "C (v7f), checkpoint 800: six holdout episodes"
echo RENDERS_COMPLETE
