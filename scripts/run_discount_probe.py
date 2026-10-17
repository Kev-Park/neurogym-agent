"""Matched gamma continuation with holdout action traces and existing renders."""
import argparse
import fcntl
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path('/scratch/kp0374/wt/neurogym-agent-zmax-left')
MANIFEST = ROOT / 'discount_pipeline.json'
ARMS = ['control', 'lower']
PREFIX = 'zmaxleft-discount-'
TARGET = 1071
EXCLUDE = 'sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl16-6'


def submit(*args):
    return subprocess.check_output(['sbatch', '--parsable', *args], text=True,
                                   timeout=20).strip().split(';')[0]


def cpu_job(stage, dependencies):
    return submit('--job-name=discount-' + stage, '--partition=highpri',
        '--account=pni', '--qos=normal', '--cpus-per-task=2', '--mem=4G',
        '--time=00:10:00', '--dependency=afterok:' + ':'.join(dependencies),
        '--output=slurm_outputs/discount-' + stage + '-%j.out',
        '--wrap=cd ' + str(ROOT) + ' && uv run --no-sync python scripts/run_discount_probe.py --' + stage)


def start():
    assert not MANIFEST.exists(), 'Inspect the existing experiment before starting another'
    source = Path('/scratch/kp0374/checkpoints/zmaxleft-verb-entropy-fixed-on')
    meta = json.loads((source / 'meta.json').read_text())
    assert meta['iteration'] == 1021
    jobs = {}
    for arm in ARMS:
        run = PREFIX + arm
        dest = Path('/scratch/kp0374/checkpoints') / run
        dest.mkdir(exist_ok=True)
        assert not (dest / 'meta.json').exists()
        shutil.copy2(source / 'ckpt_001021.pkl', dest)
        copied = dict(meta); copied.pop('wandb_id', None); copied['run_name'] = run
        (dest / 'meta.json').write_text(json.dumps(copied))
        jobs[arm] = submit('--job-name=discount-' + arm, '--exclude=' + EXCLUDE,
            '--export=ALL,TRAIN_CONFIG=configs/native_discount_' + arm + '.yaml,CKPT_EVERY=1,WANDB_MODE=offline',
            'native/r_train_zmaxleft_compact.slurm', run, str(TARGET))
    jobs['report'] = cpu_job('finish-training', list(jobs.values()))
    MANIFEST.write_text(json.dumps({'jobs': jobs, 'arms': ARMS,
        'source_iteration': 1021, 'target_iteration': TARGET, 'max_retries': 12,
        'run_prefix': PREFIX, 'config_prefix': 'configs/native_discount_',
        'report_file': 'discount_report.json',
        'report_command': 'uv run --no-sync python scripts/run_discount_probe.py --finish-training'}, indent=2))
    print(json.dumps(jobs))


def finish_training():
    with MANIFEST.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(MANIFEST.read_text())
        assert not any(k.startswith('audit_') for k in manifest['jobs'])
        evaluations = []
        for arm in ARMS:
            run = PREFIX + arm
            directory = Path('/scratch/kp0374/checkpoints') / run
            assert json.loads((directory / 'meta.json').read_text())['iteration'] >= TARGET
            checkpoint = str(directory / f'ckpt_{TARGET:06d}.pkl')
            assert Path(checkpoint).exists()
            for offset in [0, 20]:
                jid = submit('--job-name=discount-audit-' + arm, '--time=00:25:00',
                    'scripts/probe_double_clicks.slurm', '--offset', str(offset), '--limit', '20',
                    '--state-pkl', checkpoint, '--config', 'configs/native_zmaxleft_v7h_eval.yaml',
                    '--output', f'discount_{arm}_audit{offset}.json',
                    '--trace-output', f'discount_{arm}_trace{offset}.json')
                manifest['jobs'][f'audit_{arm}_{offset}'] = jid
                evaluations.append(jid)
            jid = submit('--job-name=discount-render-' + arm, '--time=00:30:00', '--mem=32G',
                '--export=ALL,PKL=' + checkpoint + ',VIDEO_ARGS=--config configs/native_zmaxleft_v7h_eval.yaml --out-dir eval_videos_' + run,
                'scripts/eval_zmaxleft_video.slurm')
            manifest['jobs']['render_' + arm] = jid
            evaluations.append(jid)
        manifest['jobs']['training_report'] = manifest['jobs']['report']
        manifest['jobs']['report'] = cpu_job('report', evaluations)
        MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['jobs']))


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(7)
    draws = values[rng.integers(len(values), size=(4000, len(values)))].mean(1)
    return {'episodes': len(values), 'mean': float(values.mean()),
            'episode_bootstrap_95ci': np.quantile(draws, [.025, .975]).tolist()}


def report():
    per_arm = {}
    for arm in ARMS:
        episodes = {}
        for offset in [0, 20]:
            audit = json.loads(Path(f'discount_{arm}_audit{offset}.json').read_text())
            trace = json.loads(Path(f'discount_{arm}_trace{offset}.json').read_text())
            assert len(audit['completed_indices']) == 20
            for episode in audit['episode_results']:
                idx = episode['idx']
                assert idx not in episodes and episode['steps'] == 500
                rows = [r for r in trace['records'] if r['idx'] == idx]
                clicks = [r for r in audit['records'] if r['idx'] == idx]
                assert len(rows) == 500
                episodes[idx] = {
                    'peak_dz': episode['peak_dz'],
                    'double_clicks': len(clicks),
                    'misses_3d': sum(r['pane'] == '3d' and not r['hit'] for r in clicks),
                    'hits_2d': sum(r['pane'] == '2d' and r['hit'] for r in clicks),
                    'neutral_zoom': sum(r['action'][0] in [2, 4] and r['action'][-1] == 4 for r in rows),
                    'neutral_rotation': sum(r['action'][0] == 1 and r['action'][3:6] == [4, 4, 4] for r in rows),
                }
        assert len(episodes) == 40
        videos = json.loads(Path('eval_videos_' + PREFIX + arm + '/manifest.json').read_text())
        assert len(videos) == 6 and all(r['steps'] == 500 for r in videos)
        per_arm[arm] = episodes
    assert set(per_arm['control']) == set(per_arm['lower'])
    metrics = list(next(iter(per_arm['control'].values())))
    output = {'design': 'Gamma .99 versus .98; common corrected-schedule checkpoint 1021; 50 updates; other training settings identical.',
        'arms': {arm: {metric: bootstrap([e[metric] for e in episodes.values()]) for metric in metrics}
                 for arm, episodes in per_arm.items()},
        'paired_lower_minus_control': {metric: bootstrap([
            per_arm['lower'][idx][metric] - per_arm['control'][idx][metric]
            for idx in sorted(per_arm['control'])]) for metric in metrics},
        'per_episode': per_arm,
        'limitations': ['One training seed per arm; episode intervals do not measure training-seed uncertainty.',
            'The lower-gamma critic is warm-started from a gamma-.99 critic and must adapt.',
            'Neutral parameter bins identify state-preserving commands; they are not necessarily avoidable waiting.',
            'Evaluation uses common task rewards and 500-step budgets, not discounted training returns.']}
    Path('discount_report.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--finish-training', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.start:
        start()
    elif args.finish_training:
        finish_training()
    elif args.report:
        report()
    else:
        parser.error('Choose a stage')
