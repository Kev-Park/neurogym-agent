"""Three matched short continuations: normal cuts, complete episodes, short cuts."""
import json
import shutil
import subprocess
from pathlib import Path


root = Path('/scratch/kp0374/wt/neurogym-agent-zmax-left')
manifest = root / 'boundary_credit_pipeline.json'
assert not manifest.exists(), 'Inspect existing boundary experiment instead of duplicating it'
source = Path('/scratch/kp0374/checkpoints/zmaxleft-click-credit')
jobs = {}
for arm in ['default', 'complete', 'short']:
    run = 'zmaxleft-boundary-' + arm
    dest = Path('/scratch/kp0374/checkpoints') / run
    dest.mkdir(exist_ok=True)
    assert not (dest / 'meta.json').exists()
    shutil.copy2(source / 'ckpt_001011.pkl', dest)
    meta = json.loads((source / 'meta.json').read_text())
    assert meta['iteration'] == 1011
    meta.pop('wandb_id', None); meta['run_name'] = run
    (dest / 'meta.json').write_text(json.dumps(meta))
    args = ['sbatch', '--parsable', '--job-name=boundary-' + arm, '--time=00:40:00',
            '--exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl16-6',
            '--export=ALL,TRAIN_CONFIG=configs/native_boundary_' + arm + '.yaml,CKPT_EVERY=1,WANDB_MODE=offline',
            'native/r_train_zmaxleft_compact.slurm', run, '1014']
    jobs[arm] = subprocess.check_output(args, text=True).strip().split(';')[0]
args = ['sbatch', '--parsable', '--job-name=boundary-report', '--partition=highpri', '--account=pni',
        '--qos=normal', '--cpus-per-task=4', '--mem=16G', '--time=00:10:00',
        '--dependency=afterok:' + ':'.join(jobs.values()), '--output=slurm_outputs/boundary-report-%j.out',
        '--wrap=cd ' + str(root) + ' && uv run --no-sync python scripts/analyze_boundary_credit.py']
jobs['report'] = subprocess.check_output(args, text=True).strip().split(';')[0]
manifest.write_text(json.dumps({'jobs': jobs, 'source_iteration': 1011, 'target_iteration': 1014}, indent=2))
print(json.dumps(jobs))
