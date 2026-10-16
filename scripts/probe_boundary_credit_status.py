"""Reconcile time-limited boundary probes and emit a connection sentinel."""
import argparse
import fcntl
import json
import subprocess
from pathlib import Path


ap = argparse.ArgumentParser()
ap.add_argument('--jobs', nargs='*')  # Shared Windows monitor interface; manifest is authoritative.
ap.add_argument('--manifest', default='boundary_credit_pipeline.json')
args = ap.parse_args()
root = Path('/scratch/kp0374/wt/neurogym-agent-zmax-left')
assert Path(args.manifest).name == args.manifest
path = root / args.manifest


def submit(*args):
    return subprocess.check_output(['sbatch', '--parsable', *args], text=True, timeout=20).strip().split(';')[0]


with path.with_suffix('.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest = json.loads(path.read_text())
    jobs = manifest['jobs']
    arms = manifest.get('arms', ['default', 'complete', 'short'])
    target = manifest.get('target_iteration', 1014)
    run_prefix = manifest.get('run_prefix', 'zmaxleft-boundary-')
    config_prefix = manifest.get('config_prefix', 'configs/native_boundary_')
    report_command = manifest.get('report_command', 'uv run --no-sync python scripts/analyze_boundary_credit.py')
    raw = subprocess.check_output(['sacct', '-j', ','.join(jobs.values()), '-n', '-P',
                                  '-o', 'JobIDRaw,State,ExitCode'], text=True, timeout=20)
    states = {}
    for line in raw.splitlines():
        parts = line.split('|')
        if parts[0] in jobs.values():
            states[parts[0]] = {'state': parts[1].split()[0], 'exit_code': parts[2]}
    iterations, events = {}, []
    changed = False
    for arm in arms:
        run = run_prefix + arm
        meta = json.loads((Path('/scratch/kp0374/checkpoints') / run / 'meta.json').read_text())
        iterations[arm] = meta['iteration']
        state = states.get(jobs[arm], {}).get('state')
        retries = manifest.setdefault('retries', {}).get(arm, 0)
        if state in ['TIMEOUT', 'PREEMPTED'] and meta['iteration'] < target and retries < 3:
            old = jobs[arm]
            jobs[arm] = submit('--job-name=boundary-' + arm, '--time=00:30:00',
                '--exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl16-6',
                '--export=ALL,TRAIN_CONFIG=' + config_prefix + arm + '.yaml,CKPT_EVERY=1,WANDB_MODE=offline',
                'native/r_train_zmaxleft_compact.slurm', run, str(target))
            manifest['retries'][arm] = retries + 1
            manifest.setdefault('history', []).append({'arm': arm, 'job': old, 'state': state})
            states[jobs[arm]] = {'state': 'PENDING', 'exit_code': '0:0'}
            events.append('RESUMED ' + arm + ' at ' + str(meta['iteration']))
            changed = True
    if changed:
        subprocess.run(['scancel', jobs['report']], check=False, timeout=20)
        jobs['report'] = submit('--job-name=boundary-report', '--partition=highpri', '--account=pni',
            '--qos=normal', '--cpus-per-task=4', '--mem=16G', '--time=00:10:00',
            '--dependency=afterok:' + ':'.join(jobs[a] for a in arms),
            '--output=slurm_outputs/boundary-report-%j.out',
            '--wrap=cd ' + str(root) + ' && ' + report_command)
        states[jobs['report']] = {'state': 'PENDING', 'exit_code': '0:0'}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2)); temporary.replace(path)
    waiting = bool(manifest.get('awaiting_training_barrier'))
    if waiting:
        ready = True
        for name in manifest['awaiting_training_barrier']:
            other_path = root / name
            if not other_path.exists():
                continue
            other = json.loads(other_path.read_text())
            for arm in other['arms']:
                directory = Path('/scratch/kp0374/checkpoints') / (other['run_prefix'] + arm)
                ready &= json.loads((directory / 'meta.json').read_text())['iteration'] >= other['target_iteration']
                ready &= (directory / ('ckpt_' + str(other['target_iteration']).zfill(6) + '.pkl')).exists()
        if ready:
            jobs['report'] = submit('--job-name=entropy-training-barrier-report', '--partition=highpri',
                '--account=pni', '--qos=normal', '--cpus-per-task=4', '--mem=16G', '--time=00:15:00',
                '--output=slurm_outputs/entropy-training-barrier-%j.out',
                '--wrap=cd ' + str(root) + ' && ' + report_command)
            states[jobs['report']] = {'state': 'PENDING', 'exit_code': '0:0'}
            manifest.pop('awaiting_training_barrier'); waiting = False
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(manifest, indent=2)); temporary.replace(path)
    current = {jid: states.get(jid, {'state': 'UNKNOWN'}) for jid in jobs.values()}
    live = {'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'UNKNOWN'}
    failed = [jid for jid, s in current.items() if s['state'] not in live | {'COMPLETED'}]
    report = root / manifest.get('report_file', 'boundary_credit_report.json')
    print('CREDIT_STATUS_JSON ' + json.dumps({'jobs': current, 'iterations': iterations,
        'failed': failed, 'events': events, 'report_done': report.exists(),
        'waiting_for_training_barrier': waiting,
        'terminal': not waiting and all(s['state'] not in live for s in current.values())}))
