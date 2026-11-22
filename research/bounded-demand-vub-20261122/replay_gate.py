"""One complete stored-prior demand-bounded cut gate; no automatic expansion."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE
INPUT = PRIOR = None


def dump(state):
    (ROOT / 'state.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')


def bounded():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))


def main():
    global ROOT, INPUT, PRIOR
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance_path', required=True)
    p.add_argument('--prior_path', required=True)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()
    INPUT, PRIOR = Path(args.instance_path).resolve(), Path(args.prior_path).resolve()
    ROOT = Path(args.output_dir).resolve()
    ROOT.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT))
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'case': 'large4', 'cold_run': False,
             'method': 'bounded_customer_support_cut', 'revision': 'accept_loaded_native_warning',
             'limits': {'cpus': 2, 'address_space_bytes': 4 * 1024**3, 'whole_seconds': 60,
                        'max_distinct_sets_including_initial': 8}}
    dump(state)
    for file in ('check_cut.py', 'check_master.py'):
        with (ROOT / (file + '.log')).open('w') as log:
            proc = subprocess.run([sys.executable, '-B', str(SOURCE / file)], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=15, preexec_fn=bounded)
        if proc.returncode:
            state.update(status='STOP_FIXTURE_FAILED', fixture=file, fixture_exit=proc.returncode)
            dump(state)
            return
    state['fixture_exit'] = 0
    dump(state)
    start = time.monotonic()
    cmd = [sys.executable, '-B', str(SOURCE / 'joint.py'), '--instance_path', str(INPUT),
           '--prior_path', str(PRIOR), '--output_dir', str(ROOT / 'gate'), '--seed', '1']
    with (ROOT / 'algorithm.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                preexec_fn=bounded, start_new_session=True)
        state['worker_pid'] = proc.pid
        dump(state)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
            state['failure'] = 'WHOLE_60S_TIMEOUT'
    wall = time.monotonic() - start
    state.update(algorithm_exit=code, algorithm_whole_seconds=wall)
    if code or not (ROOT / 'gate' / 'metrics.json').exists():
        state['status'] = 'STOP_NO_COMPLETE_GATE'
        dump(state)
        return
    import common
    import feasibility_check
    data = common.load(INPUT)
    results = []
    for path in sorted((ROOT / 'gate').glob('candidate-*.json')) + [ROOT / 'gate' / 'solution.json']:
        sol = json.loads(path.read_text())
        independent = common.verify(data, sol)
        official = feasibility_check.check_feasibility(data, sol)
        assert official['feasible']
        results.append({'file': path.name, 'independent': independent, 'official': official})
    metrics = json.loads((ROOT / 'gate' / 'metrics.json').read_text())
    actual = [t for t in metrics['trials'] if t.get('oracle', {}).get('native_result_available')
              and t['oracle'].get('termination') == 'RETURNED_VERIFIED_FIXED_S'
              and t['oracle'].get('fixed_s_model_status') == 'HighsModelStatus.kOptimal']
    improves = metrics['best_objective'] <= metrics['start_objective'] * (1 - .005)
    full_wall = time.monotonic() - start
    state.update(status='DONE', double_checks=results, metrics=metrics,
                 gate='PASS' if improves and actual and full_wall <= 60 and not metrics.get('failure') else 'STOP',
                 complete_new_set_oracles=len(actual), whole_with_checkers_seconds=full_wall)
    dump(state)


if __name__ == '__main__':
    main()
