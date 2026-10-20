"""Three-gamma, three-seed continuation study from one shared checkpoint."""
import argparse
import fcntl
import json
import subprocess
from pathlib import Path

ROOT = Path('/scratch/kp0374/wt/neurogym-agent-zmax-left')
CHECKPOINTS = Path('/scratch/kp0374/checkpoints')
MANIFEST = ROOT / 'discount_sweep_pipeline.json'
SOURCE = CHECKPOINTS / 'zmaxleft-verb-entropy-fixed-on'
SOURCE_ITER = 1021
TARGET = 1071
SEEDS = [3101, 3102, 3103]
GAMMAS = {'g099': ('configs/native_discount_control.yaml', 0.99),
          'g0985': ('configs/native_discount_middle.yaml', 0.985),
          'g097': ('configs/native_discount_aggressive.yaml', 0.97)}
EXCLUDE = 'sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl16-6'


def submit(*args):
    return subprocess.check_output(['sbatch', '--parsable', *args], text=True,
                                   timeout=20).strip().split(';')[0]


def start():
    assert not MANIFEST.exists(), 'Existing sweep manifest found; inspect it before starting.'
    meta = json.loads((SOURCE / 'meta.json').read_text())
    assert meta['iteration'] == SOURCE_ITER
    arms = []
    jobs = {}
    retries = {}
    for seed in SEEDS:
        for gamma_key, (config, gamma) in GAMMAS.items():
            key = f's{seed}-{gamma_key}'
            run = f'zmaxleft-discount-sweep-{key}'
            dest = CHECKPOINTS / run
            dest.mkdir(exist_ok=True)
            assert not (dest / 'meta.json').exists()
            source_checkpoint = SOURCE / f'ckpt_{SOURCE_ITER:06d}.pkl'
            (dest / source_checkpoint.name).write_bytes(source_checkpoint.read_bytes())
            copied = dict(meta)
            copied.pop('wandb_id', None)
            copied['run_name'] = run
            (dest / 'meta.json').write_text(json.dumps(copied))
            arm = {'key': key, 'seed': seed, 'gamma': gamma,
                   'config': config, 'run_name': run}
            arms.append(arm)
            jobs[key] = submit('--job-name=ds-' + key, '--exclude=' + EXCLUDE,
                '--export=ALL,TRAIN_CONFIG=' + config + ',TRAIN_SEED=' + str(seed) +
                    ',CKPT_EVERY=1,WANDB_MODE=offline',
                'native/r_train_zmaxleft_compact.slurm', run, str(TARGET))
            retries[key] = 0
    manifest = {'phase': 'training', 'arms': arms, 'jobs': jobs,
        'retries': retries, 'history': [], 'source_iteration': SOURCE_ITER,
        'target_iteration': TARGET, 'max_retries': 12,
        'report_file': 'discount_sweep_report.json'}
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(jobs))


def _eval_job(arm, offset):
    key = arm['key']
    checkpoint = CHECKPOINTS / arm['run_name'] / f"ckpt_{TARGET:06d}.pkl"
    return submit('--job-name=ds-eval-' + key, '--time=00:25:00',
        'scripts/probe_double_clicks.slurm', '--offset', str(offset), '--limit', '20',
        '--state-pkl', str(checkpoint), '--config', 'configs/native_zmaxleft_v7h_eval.yaml',
        '--output', f'discount_sweep_{key}_audit{offset}.json',
        '--trace-output', f'discount_sweep_{key}_trace{offset}.json')


def advance(manifest):
    jobs = manifest['jobs']
    arms = manifest['arms']
    if manifest['phase'] == 'training':
        all_ready = all(
            json.loads((CHECKPOINTS / arm['run_name'] / 'meta.json').read_text())['iteration'] >= TARGET
            and (CHECKPOINTS / arm['run_name'] / f'ckpt_{TARGET:06d}.pkl').exists()
            for arm in arms)
        if all_ready:
            for arm in arms:
                for offset in (0, 20):
                    name = f"eval_{arm['key']}_{offset}"
                    jobs[name] = _eval_job(arm, offset)
            manifest['phase'] = 'evaluation'
    elif manifest['phase'] == 'evaluation':
        eval_names = [f"eval_{a['key']}_{offset}" for a in arms for offset in (0, 20)]
        raw = subprocess.check_output(['sacct', '-j', ','.join(jobs[n] for n in eval_names),
            '-n', '-P', '-o', 'JobIDRaw,State'], text=True, timeout=20)
        states = {p[0]: p[1].split()[0] for p in (line.split('|') for line in raw.splitlines()) if len(p) > 1}
        if all(states.get(jobs[n]) == 'COMPLETED' for n in eval_names):
            jobs['report'] = submit('--job-name=ds-report', '--partition=highpri',
                '--account=pni', '--qos=normal', '--cpus-per-task=2', '--mem=4G',
                '--time=00:10:00', '--dependency=afterok:' + ':'.join(jobs[n] for n in eval_names),
                '--output=slurm_outputs/discount-sweep-report-%j.out',
                '--wrap=cd ' + str(ROOT) + ' && uv run --no-sync python scripts/run_discount_sweep.py --report')
            manifest['phase'] = 'report'
    return manifest


def status():
    with MANIFEST.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(MANIFEST.read_text())
        changed = False
        if manifest['phase'] == 'training':
            keys = [a['key'] for a in manifest['arms']]
            raw = subprocess.check_output(['sacct', '-j', ','.join(manifest['jobs'][k] for k in keys),
                '-n', '-P', '-o', 'JobIDRaw,State,ExitCode'], text=True, timeout=20)
            states = {}
            for line in raw.splitlines():
                p = line.split('|')
                if len(p) > 2:
                    states[p[0]] = {'state': p[1].split()[0], 'exit_code': p[2]}
            for arm in manifest['arms']:
                key = arm['key']
                ckpt_dir = CHECKPOINTS / arm['run_name']
                meta = json.loads((ckpt_dir / 'meta.json').read_text())
                state = states.get(manifest['jobs'][key], {}).get('state')
                if state in {'TIMEOUT', 'PREEMPTED', 'NODE_FAIL'} and meta['iteration'] < TARGET and manifest['retries'][key] < manifest['max_retries']:
                    old = manifest['jobs'][key]
                    manifest['jobs'][key] = submit('--job-name=ds-' + key, '--exclude=' + EXCLUDE,
                        '--export=ALL,TRAIN_CONFIG=' + arm['config'] + ',TRAIN_SEED=' + str(arm['seed']) +
                            ',CKPT_EVERY=1,WANDB_MODE=offline',
                        'native/r_train_zmaxleft_compact.slurm', arm['run_name'], str(TARGET))
                    manifest['retries'][key] += 1
                    manifest['history'].append({'arm': key, 'job': old, 'state': state,
                                                'iteration': meta['iteration']})
                    changed = True
            old_phase = manifest['phase']
            manifest = advance(manifest)
            changed |= manifest['phase'] != old_phase
        elif manifest['phase'] == 'evaluation':
            manifest = advance(manifest)
            changed = True
        if changed:
            tmp = MANIFEST.with_suffix('.tmp')
            tmp.write_text(json.dumps(manifest, indent=2))
            tmp.replace(MANIFEST)
        live = {'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED'}
        raw = subprocess.check_output(['sacct', '-j', ','.join(manifest['jobs'].values()),
            '-n', '-P', '-o', 'JobIDRaw,State'], text=True, timeout=20)
        states = {p[0]: p[1].split()[0] for p in (line.split('|') for line in raw.splitlines()) if len(p) > 1}
        failed = [jid for jid in manifest['jobs'].values()
                  if jid in states and states[jid] not in live | {'COMPLETED'}]
        report_state = states.get(manifest['jobs'].get('report'))
        terminal = manifest['phase'] == 'report' and report_state is not None and report_state not in live
        print('CREDIT_STATUS_JSON ' + json.dumps({'phase': manifest['phase'], 'jobs': states,
            'failed': failed, 'terminal': terminal,
            'report_done': (ROOT / manifest['report_file']).exists()}))


def bootstrap(values, seed=7):
    import numpy as np

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {'n': len(values), 'mean': float(values.mean()),
            'bootstrap_95ci': np.quantile(draws, [.025, .975]).tolist()}


def report():
    import numpy as np

    metrics = ('peak_dz', 'double_clicks', 'misses_3d', 'hits_2d', 'neutral_zoom', 'neutral_rotation')
    rows = {}
    for seed in SEEDS:
        rows[seed] = {}
        for gamma_key in GAMMAS:
            key = f's{seed}-{gamma_key}'
            episodes = {}
            for offset in (0, 20):
                audit = json.loads(Path(f'discount_sweep_{key}_audit{offset}.json').read_text())
                trace = json.loads(Path(f'discount_sweep_{key}_trace{offset}.json').read_text())
                assert len(audit['completed_indices']) == 20
                for ep in audit['episode_results']:
                    idx = ep['idx']
                    assert idx not in episodes and ep['steps'] == 500
                    action_rows = [r for r in trace['records'] if r['idx'] == idx]
                    clicks = [r for r in audit['records'] if r['idx'] == idx]
                    assert len(action_rows) == 500
                    episodes[idx] = {
                        'peak_dz': ep['peak_dz'],
                        'double_clicks': len(clicks),
                        'misses_3d': sum(r['pane'] == '3d' and not r['hit'] for r in clicks),
                        'hits_2d': sum(r['pane'] == '2d' and r['hit'] for r in clicks),
                        'neutral_zoom': sum(r['action'][0] in [2, 4] and r['action'][-1] == 4 for r in action_rows),
                        'neutral_rotation': sum(r['action'][0] == 1 and r['action'][3:6] == [4, 4, 4] for r in action_rows),
                    }
            assert len(episodes) == 40
            rows[seed][gamma_key] = episodes
    output = {'design': 'Three gamma values (.99, .985, .97), three explicitly seeded continuations per value; common checkpoint 1021; 50 updates; 40 paired 500-step holdout episodes per continuation.',
        'source_checkpoint': str(SOURCE / f'ckpt_{SOURCE_ITER:06d}.pkl'),
        'training_seeds': SEEDS, 'gammas': {k: v[1] for k, v in GAMMAS.items()},
        'per_seed': {}, 'across_seed_paired_differences_vs_099': {},
        'limitations': ['All continuations share one trained starting checkpoint; seeds vary only post-checkpoint training randomness.',
            'Only three training seeds; seed-level intervals are imprecise.',
            'The critic is initialized from a gamma-.99 checkpoint and adapts after each gamma change.',
            'Evaluation uses common undiscounted task metrics and paired starts.']}
    for seed in SEEDS:
        output['per_seed'][str(seed)] = {}
        for gamma_key, (_, gamma) in GAMMAS.items():
            episodes = rows[seed][gamma_key]
            output['per_seed'][str(seed)][gamma_key] = {
                'gamma': gamma,
                'metrics': {m: float(np.mean([e[m] for e in episodes.values()])) for m in metrics}}
    for gamma_key in ('g0985', 'g097'):
        diff_by_seed = {m: [] for m in metrics}
        for seed in SEEDS:
            base = rows[seed]['g099']; target = rows[seed][gamma_key]
            assert set(base) == set(target)
            for metric in metrics:
                diff_by_seed[metric].append(float(np.mean([target[i][metric] - base[i][metric] for i in base])))
        output['across_seed_paired_differences_vs_099'][gamma_key] = {
            m: {'per_seed': vals, **bootstrap(vals)} for m, vals in diff_by_seed.items()}
    (ROOT / 'discount_sweep_report.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--jobs', nargs='*')
    parser.add_argument('--manifest', default='discount_sweep_pipeline.json')
    args = parser.parse_args()
    if Path(args.manifest).name != args.manifest:
        parser.error('--manifest must be a filename in the repository root')
    MANIFEST = ROOT / args.manifest
    if args.start:
        start()
    elif args.status or args.jobs is not None:
        status()
    elif args.report:
        report()
    else:
        parser.error('Choose --start, --status, or --report')
