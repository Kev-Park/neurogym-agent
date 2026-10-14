"""Launch an isolated diagnostic continuation and linked counterfactual probes."""
import json
import subprocess
from pathlib import Path


ROOT = Path('/scratch/kp0374/wt/neurogym-agent-zmax-left')
manifest = ROOT / 'click_credit_followup.json'
assert not manifest.exists(), 'Follow-up already launched; inspect manifest instead'
meta = json.loads(Path('/scratch/kp0374/checkpoints/zmaxleft-click-credit/meta.json').read_text())
assert meta['iteration'] == 1001, 'First complete diagnostic update is required'


def submit(*args):
    return subprocess.check_output(['sbatch', '--parsable', *args], text=True).strip().split(';')[0]


jobs = {}
jobs['train'] = submit('--job-name=click-credit-full', '--time=00:55:00',
    '--export=ALL,TRAIN_CONFIG=configs/native_zmaxleft_credit.yaml,CKPT_EVERY=1,WANDB_MODE=offline',
    'native/r_train_zmaxleft_compact.slurm', 'zmaxleft-click-credit', '1011')
directory = 'click_credit_v7j_control_full'
jobs['report'] = submit('--job-name=click-credit-report', '--partition=highpri', '--account=pni',
    '--qos=normal', '--cpus-per-task=4', '--mem=16G', '--time=00:10:00',
    '--output=slurm_outputs/click-credit-report-%j.out', '--dependency=afterok:' + jobs['train'],
    '--wrap=cd ' + str(ROOT) + ' && uv run --no-sync python scripts/analyze_click_credit.py ' + directory +
    ' --output click_credit_report.json && uv run --no-sync python scripts/select_click_credit_cases.py ' +
    directory + ' --output click_credit_cases.json')
for i in range(4):
    jobs['cf' + str(i)] = submit('--job-name=click-credit-cf' + str(i), '--time=00:40:00',
        '--dependency=afterok:' + jobs['report'], 'scripts/probe_click_counterfactual.slurm',
        '--offset', str(i * 2), '--limit', '2', '--horizon', '100', '--repeats', '2',
        '--state-pkl', '/scratch/kp0374/checkpoints/zmaxleft-click-credit/ckpt_001001.pkl',
        '--train-config', 'configs/native_zmaxleft_v7j_control.yaml', '--only-3d-misses',
        '--cases-file', 'click_credit_cases.json', '--output', 'cf_credit_part' + str(i) + '.json')
manifest.write_text(json.dumps({'jobs': jobs, 'source_iteration': 1001, 'target_iteration': 1011,
                               'counterfactual_states': 8, 'repeats': 2, 'continuation_actions': 100}, indent=2))
print(json.dumps(jobs))
