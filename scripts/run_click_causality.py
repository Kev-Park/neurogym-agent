"""Bounded, event-driven SLURM pipeline; callbacks survive VPN/client loss.

Run once with --start, then after-any callbacks reconcile states and submit
ready stages. No model calls or credential changes. All artifacts stay in the
experiment checkout; checkpoint directories are separate from v7h.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/scratch/kp0374/wt/neurogym-agent-zmax-left')
MANIFEST = ROOT / 'entropy_v7j_pipeline.json'
RUN_PREFIX = 'zmaxleft-v7j'
OUTPUT_STEM = 'entropy_v7j'
COMPACT = False
ARMS = ['control', 'zero', 'free', 'both', 'noop']
EXCLUDE = 'sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl16-6'


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def valid(key, stage):
    """A successful process exit alone does not establish complete coverage."""
    try:
        kind = stage['kind']
        if kind == 'cf':
            return len(json.loads((ROOT / f'cf_click_part{key[-1]}.json').read_text())['completed_indices']) == 6
        arm = key.split('.')[0]
        run = f'{RUN_PREFIX}-{arm}'
        if kind == 'train':
            directory = Path('/scratch/kp0374/checkpoints') / run
            return (directory / 'ckpt_001000.pkl').exists() and json.loads((directory / 'meta.json').read_text())['iteration'] >= 1000
        if kind == 'eval':
            offset = key.split('.eval')[1]
            d = json.loads((ROOT / f'eval_{run}_part{offset}.json').read_text())
            return len(d['per_state']) == 100 and all(r['steps'] == 500 and not r['wedged'] for r in d['per_state'])
        if kind == 'audit':
            offset = key.split('.audit')[1]
            d = json.loads((ROOT / f'dbl_{run}_{offset}.json').read_text())
            return len(d['completed_indices']) == 10
        if kind == 'video':
            d = json.loads((ROOT / f'eval_videos_{run}/manifest.json').read_text())
            return len(d) == 6 and all(r['steps'] == 500 for r in d)
        if key.endswith('.merge'):
            return json.loads((ROOT / f'eval_{run}_full.json').read_text())['summary']['n'] == 200
        if key.endswith('.stats'):
            return json.loads((ROOT / f'dbl_{run}_stats.json').read_text())['episodes'] == 40
        return (ROOT / f'{OUTPUT_STEM}_report.json').exists()
    except (OSError, ValueError, KeyError):
        return False


def stages():
    result = {}
    cf = []
    for i, key in enumerate(cf):
        result[key] = dict(needs=[], kind='cf', args=['scripts/probe_click_counterfactual.slurm',
                           '--offset', str(i * 6), '--limit', '6', '--output', f'cf_click_part{i}.json'])
    for arm in ARMS:
        run = f'{RUN_PREFIX}-{arm}'
        pkl = f'/scratch/kp0374/checkpoints/{run}/ckpt_001000.pkl'
        train = arm + '.train'
        result[train] = dict(needs=cf, kind='train', args=[
            '--nodes=1', '--time=10:00:00' if COMPACT else '--time=04:00:00', f'--exclude={EXCLUDE}',
            f'--export=ALL,TRAIN_CONFIG=configs/native_zmaxleft_v7j_{arm}.yaml,CKPT_EVERY=1' if COMPACT else
            f'--export=ALL,TRAIN_CONFIG=configs/native_zmaxleft_v7j_{arm}.yaml,CKPT_EVERY=2',
            'native/r_train_zmaxleft_compact.slurm' if COMPACT else 'native/r_train_zmaxleft_mn.slurm', run, '1000'])
        # Wait for all training arms before highpri evaluation: our own evals
        # should not preempt the remaining experimental training arms.
        needs = [a + '.train' for a in ARMS]
        evals, audits = [], []
        for offset in [0, 100]:
            key = f'{arm}.eval{offset}'
            evals.append(key)
            output = f'eval_{run}_part{offset}.json'
            policy = (f'--state-pkl {pkl} --config configs/native_zmaxleft_v7h_eval.yaml '
                      f'--stochastic --offset {offset} --limit 100')
            result[key] = dict(needs=needs, kind='eval', args=[
                '--partition=highpri', '--time=02:30:00', '--mem=32G',
                f'--export=ALL,OUTPUT={output},POLICY_ARGS={policy}', 'scripts/eval_zmaxleft.slurm'])
        result[arm + '.video'] = dict(needs=needs, kind='video', args=[
            '--mem=32G', f'--export=ALL,PKL={pkl},VIDEO_ARGS=--config configs/native_zmaxleft_v7h_eval.yaml --out-dir eval_videos_{run}',
            'scripts/eval_zmaxleft_video.slurm'])
        for offset in [0, 10, 20, 30]:
            key = f'{arm}.audit{offset}'
            audits.append(key)
            result[key] = dict(needs=needs, kind='audit', args=[
                'scripts/probe_double_clicks.slurm', '--offset', str(offset), '--limit', '10',
                '--state-pkl', pkl, '--output', f'dbl_{run}_{offset}.json'])
        merge = (f'uv run --no-sync python scripts/merge_zmaxleft_evals.py eval_{run}_full.json '
                 f'eval_{run}_part0.json eval_{run}_part100.json')
        result[arm + '.merge'] = dict(needs=evals, kind='cpu', wrap=merge)
        inputs = ' '.join(f'dbl_{run}_{i}.json' for i in [0, 10, 20, 30])
        result[arm + '.stats'] = dict(needs=audits, kind='cpu', wrap=
            f'uv run --no-sync python scripts/analyze_double_clicks.py {inputs} --output dbl_{run}_stats.json')
    result['report'] = dict(needs=list(result), kind='cpu', wrap=
        'uv run --no-sync python scripts/report_click_causality.py' + (' --compact' if COMPACT else ''))
    return result


def submit(key, stage):
    extra = stage.get('extra_exclude', '')
    args = ['sbatch', '--parsable', f'--job-name=ent-{key}', '--mail-type=FAIL',
            '--mail-user=code@thiskevin.com']
    if stage['kind'] == 'cpu':
        args += ['--partition=highpri', '--account=pni', '--qos=normal', '--cpus-per-task=1',
                 '--mem=4G', '--time=00:05:00', '--output=slurm_outputs/entropy-%j.out',
                 '--wrap', f'cd {ROOT} && ' + stage['wrap']]
    else:
        stage_args = list(stage['args'])
        args += ['--exclude=' + EXCLUDE]
        if extra:
            stage_args = [a for a in stage_args if not a.startswith('--exclude=')]
            args += ['--exclude=' + EXCLUDE + ',' + extra]
        args += stage_args
    if key == 'report':
        args[3] = '--mail-type=END,FAIL'
    jid = command(*args).split(';')[0]
    # After-any permits retries on failure and does not depend on the VPN.
    callback = command('sbatch', '--parsable', '--partition=highpri', '--account=pni',
        '--qos=normal', '--cpus-per-task=1', '--mem=1G', '--time=00:05:00',
        f'--dependency=afterany:{jid}', '--job-name=entropy-tick',
        '--output=slurm_outputs/entropy-tick-%j.out', '--wrap',
        f'cd {ROOT} && uv run --no-sync python scripts/run_click_causality.py --tick' + (' --compact' if COMPACT else ''))
    stage.update(job=jid, callback=callback, status='submitted', attempts=stage.get('attempts', 0) + 1)
    print('SUBMITTED', key, jid, flush=True)


def main():
    global MANIFEST, RUN_PREFIX, OUTPUT_STEM, COMPACT
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', action='store_true')
    ap.add_argument('--tick', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--compact', action='store_true')
    args = ap.parse_args()
    if args.compact:
        COMPACT = True
        RUN_PREFIX = 'zmaxleft-v7j-compact'
        OUTPUT_STEM = 'entropy_v7j_compact'
        MANIFEST = ROOT / f'{OUTPUT_STEM}_pipeline.json'
    if args.dry_run:
        print(json.dumps(stages(), indent=2))
        return
    if args.status:
        data = json.loads(MANIFEST.read_text())
        active = {k: v['job'] for k, v in data['stages'].items() if v.get('status') == 'submitted'}
        progress = {}
        for arm in ARMS:
            p = Path('/scratch/kp0374/checkpoints') / f'{RUN_PREFIX}-{arm}' / 'meta.json'
            progress[arm] = json.loads(p.read_text()).get('iteration') if p.exists() else None
        print('ENTROPY_STATUS_JSON ' + json.dumps({
            'updated': data['updated'], 'active': active, 'progress': progress,
            'failed': [k for k, v in data['stages'].items() if v.get('status') == 'failed'],
            'report_done': data['stages']['report'].get('status') == 'done'}))
        if active:
            print(command('sacct', '-n', '-X', '-P', '-j', ','.join(active.values()), '-o', 'JobID,State,Elapsed'))
        return
    with open(ROOT / 'entropy_v7j_pipeline.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.start:
            if MANIFEST.exists():
                raise RuntimeError('Pipeline already exists; use --tick')
            for arm in ARMS:
                dest = Path('/scratch/kp0374/checkpoints') / f'{RUN_PREFIX}-{arm}'
                dest.mkdir(exist_ok=False)
                shutil.copy2('/scratch/kp0374/checkpoints/zmaxleft-v7i-grid/ckpt_000900.pkl', dest)
                # The reward curriculum reads this before the first iteration.
                # An absent progress file would briefly restore novelty rewards.
                source_meta = json.loads(Path('/scratch/kp0374/checkpoints/zmaxleft-v7i-grid/meta.json').read_text())
                (dest / 'meta.json').write_text(json.dumps({
                    'iteration': 900, 'total_steps': source_meta['total_steps'],
                    'run_name': f'{RUN_PREFIX}-{arm}', 'degraded_exits': []}))
            data = {'created': time.time(), 'source': command('git', 'rev-parse', 'HEAD'), 'stages': stages()}
        else:
            data = json.loads(MANIFEST.read_text())
        jobs = [s['job'] for s in data['stages'].values() if s.get('status') == 'submitted']
        states = {}
        if jobs:
            for line in command('sacct', '-n', '-X', '-P', '-j', ','.join(jobs), '-o', 'JobID,State,NodeList').splitlines():
                parts = line.split('|')
                if len(parts) >= 3:
                    states[parts[0]] = parts[1:3]
        for key, stage in data['stages'].items():
            if stage.get('status') != 'submitted':
                continue
            state, node = states.get(stage['job'], ['', ''])
            if state == 'COMPLETED' and valid(key, stage):
                stage['status'] = 'done'
                continue
            if state == 'COMPLETED' or (state.split()[0:1] and state.split()[0] in ['FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL']):
                if stage['attempts'] < 3:
                    stage['status'] = 'retry'
                    # Do not keep retrying a node with a CUDA-init fault.
                    logs = ''.join(p.read_text(errors='replace')[-1000000:]
                                   for p in (ROOT / 'slurm_outputs').glob(f'*{stage["job"]}*.out'))
                    if node.startswith('sarekl') and '[' not in node and ('CUDA unknown error' in logs or 'eglInitialize failed' in logs):
                        stage['extra_exclude'] = node
                else:
                    stage['status'] = 'failed'
                    print('NEEDS_ATTENTION', key, state, flush=True)
        for key, stage in data['stages'].items():
            if stage.get('status') in ['submitted', 'done', 'failed']:
                continue
            if all(data['stages'][p].get('status') == 'done' for p in stage['needs']):
                submit(key, stage)
        data['updated'] = time.time()
        tmp = MANIFEST.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(MANIFEST)
        print('PIPELINE_STATUS', {k: v.get('status', 'waiting') for k, v in data['stages'].items()}, flush=True)


if __name__ == '__main__':
    main()
