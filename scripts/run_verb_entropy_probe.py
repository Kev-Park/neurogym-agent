"""Matched verb-entropy test with zero double-grid bonus and common holdout starts."""
import argparse
import fcntl
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path('/scratch/kp0374/wt/neurogym-agent-zmax-left')
MANIFEST = ROOT / 'verb_entropy_pipeline.json'


def submit(*args):
    return subprocess.check_output(['sbatch', '--parsable', *args], text=True).strip().split(';')[0]


def cpu_job(command, dependencies, name):
    return submit('--job-name=' + name, '--partition=highpri', '--account=pni', '--qos=normal',
        '--cpus-per-task=4', '--mem=16G', '--time=00:15:00', '--dependency=afterok:' + ':'.join(dependencies),
        '--output=slurm_outputs/' + name + '-%j.out', '--wrap=cd ' + str(ROOT) + ' && ' + command)


def start():
    assert not MANIFEST.exists(), 'Existing probe must be inspected, not duplicated'
    source = Path('/scratch/kp0374/checkpoints/zmaxleft-click-credit')
    original = json.loads((source / 'meta.json').read_text())
    assert original['iteration'] == 1011
    jobs = {}
    for arm in ['on', 'off']:
        run = 'zmaxleft-verb-entropy-' + arm
        dest = Path('/scratch/kp0374/checkpoints') / run
        dest.mkdir(exist_ok=True)
        assert not (dest / 'meta.json').exists()
        shutil.copy2(source / 'ckpt_001011.pkl', dest)
        meta = dict(original); meta.pop('wandb_id', None); meta['run_name'] = run
        (dest / 'meta.json').write_text(json.dumps(meta))
        jobs[arm] = submit('--job-name=verb-entropy-' + arm, '--time=00:40:00',
            '--exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl16-6',
            '--export=ALL,TRAIN_CONFIG=configs/native_verb_entropy_' + arm + '.yaml,CKPT_EVERY=1,WANDB_MODE=offline',
            'native/r_train_zmaxleft_compact.slurm', run, '1021')
    command = 'uv run --no-sync python scripts/run_verb_entropy_probe.py --finish-training'
    jobs['report'] = cpu_job(command, list(jobs.values()), 'verb-entropy-training-report')
    MANIFEST.write_text(json.dumps({'jobs': jobs, 'arms': ['on', 'off'], 'target_iteration': 1021,
        'source_iteration': 1011, 'run_prefix': 'zmaxleft-verb-entropy-',
        'config_prefix': 'configs/native_verb_entropy_', 'report_file': 'verb_entropy_report.json',
        'report_command': command}, indent=2))
    print(json.dumps(jobs))


def finish_training():
    for arm in ['on', 'off']:
        meta = json.loads(Path('/scratch/kp0374/checkpoints/zmaxleft-verb-entropy-' + arm + '/meta.json').read_text())
        assert meta['iteration'] >= 1021
        subprocess.run(['uv', 'run', '--no-sync', 'python', 'scripts/analyze_click_credit.py',
            'verb_entropy_' + arm + '/gradients', '--output', 'verb_entropy_' + arm + '_gradient_report.json'],
            check=True, stdout=subprocess.DEVNULL)
        for p in Path('verb_entropy_' + arm + '/gradients').glob('learner-*.jsonl'):
            for line in p.open():
                d = json.loads(line)
                assert d['verb_entropy_scale'] == (1. if arm == 'on' else 0.)
                assert d['double_grid_entropy_scale'] == 0.
    with MANIFEST.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(MANIFEST.read_text())
        assert not any(k.startswith('audit_') for k in manifest['jobs']), 'Audits already submitted'
        manifest['jobs']['training_report'] = manifest['jobs']['report']
        audits = []
        for arm in ['on', 'off']:
            for offset in [0, 20]:
                jid = submit('--job-name=verb-audit-' + arm + '-' + str(offset), '--time=00:25:00',
                    'scripts/probe_double_clicks.slurm', '--offset', str(offset), '--limit', '20',
                    '--state-pkl', '/scratch/kp0374/checkpoints/zmaxleft-verb-entropy-' + arm + '/ckpt_001021.pkl',
                    '--config', 'configs/native_zmaxleft_v7h_eval.yaml',
                    '--output', f'verb_entropy_{arm}_audit{offset}.json',
                    '--trace-output', f'verb_entropy_{arm}_trace{offset}.json')
                manifest['jobs'][f'audit_{arm}_{offset}'] = jid; audits.append(jid)
        manifest['jobs']['report'] = cpu_job('uv run --no-sync python scripts/run_verb_entropy_probe.py --report',
                                             audits, 'verb-entropy-final-report')
        MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['jobs']))


def bootstrap(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(7)
    means = a[rng.integers(len(a), size=(4000, len(a)))].mean(1)
    return {'n': len(a), 'mean': float(a.mean()), 'episode_bootstrap_95ci': np.quantile(means, [.025, .975]).tolist()}


def report():
    arms, episodes = {}, {}
    for arm in ['on', 'off']:
        records, results, covered = [], [], []
        for offset in [0, 20]:
            d = json.load(open(f'verb_entropy_{arm}_audit{offset}.json'))
            assert len(d['completed_indices']) == 20
            records.extend(d['records']); results.extend(d['episode_results']); covered.extend(d['completed_indices'])
        assert len(covered) == len(set(covered)) == 40
        assert all(r['steps'] == 500 for r in results)
        per = {r['idx']: {'peak_dz': r['peak_dz']} for r in results}
        for idx in covered:
            rr = [r for r in records if r['idx'] == idx]
            per[idx].update(double_clicks=len(rr), misses=sum(not r['hit'] for r in rr),
                misses_3d=sum(r['pane'] == '3d' and not r['hit'] for r in rr),
                hits_2d=sum(r['pane'] == '2d' and r['hit'] for r in rr), hits=sum(r['hit'] for r in rr))
        episodes[arm] = per
        arms[arm] = {key: bootstrap([r[key] for r in per.values()]) for key in next(iter(per.values()))}
        arms[arm]['3d_miss_action_fraction'] = sum(r['misses_3d'] for r in per.values()) / 20000
        arms[arm]['gradient_report'] = json.load(open('verb_entropy_' + arm + '_gradient_report.json'))
    assert set(episodes['on']) == set(episodes['off'])
    paired = {key: bootstrap([episodes['off'][idx][key] - episodes['on'][idx][key]
                              for idx in sorted(episodes['on'])]) for key in next(iter(episodes['on'].values()))}
    output = {'design': 'Shared checkpoint 1011; ten updates; double-grid entropy zero in both arms; verb entropy scale 1 versus 0; rewards unchanged.',
        'arms': arms, 'paired_off_minus_on': paired, 'limitations': [
            'One training continuation per arm; episode intervals do not cover training-seed uncertainty.',
            'Verb entropy removal affects all verb choices, not only double clicks.',
            'Rotation/zoom conditional-entropy branch weights remain attached; their effects are included in the gradient trace.',
            'Ten updates test short-term sensitivity, not asymptotic convergence.']}
    Path('verb_entropy_report.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


ap = argparse.ArgumentParser()
ap.add_argument('--start', action='store_true')
ap.add_argument('--finish-training', action='store_true')
ap.add_argument('--report', action='store_true')
args = ap.parse_args()
if args.start: start()
elif args.finish_training: finish_training()
elif args.report: report()
else: ap.error('Choose one stage')
