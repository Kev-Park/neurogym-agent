#!/bin/bash
# Brief end-to-end image/rendering check for the security-minimum dependencies.
# Run after syncing the updated lockfile and environment.
#SBATCH --job-name=security-smoke
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:25:00
#SBATCH --output=slurm_outputs/security-smoke-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project worktree}"
export PYTHONUNBUFFERED=1

echo "=== node: $(hostname) ==="
uv --version
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

uv run --no-sync python - <<'PY'
from importlib.metadata import version
import subprocess

uv_version = subprocess.check_output(["uv", "--version"], text=True).split()[1]
print(f"uv={uv_version} (minimum 0.13.4)", flush=True)
assert tuple(map(int, uv_version.split("."))) >= (0, 13, 4)

requirements = {"pillow": "13.0.0", "certifi": "2026.10.19"}
for package, minimum in requirements.items():
    installed = version(package)
    print(f"{package}={installed} (minimum {minimum})", flush=True)
    assert tuple(map(int, installed.split("."))) >= tuple(map(int, minimum.split(".")))
PY

uv run --no-sync python native/probe_smoke_both.py \
  --config configs/native_select.yaml --steps 8 --backends chrome,simulator
