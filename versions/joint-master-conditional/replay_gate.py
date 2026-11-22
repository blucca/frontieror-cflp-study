"""One stored-prior large4 gate, bounded worker and complete original checkers."""
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
ROOT = SOURCE  # Set to the requested output directory in main.
INPUT = PRIOR = None


def dump(value):
    (ROOT / 'state.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


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
    a = p.parse_args()
    ROOT, INPUT, PRIOR = Path(a.output_dir).resolve(), Path(a.instance_path).resolve(), Path(a.prior_path).resolve()
    ROOT.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT))
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'case': 'large4', 'cold_run': False,
             'limits': {'cpus': 2, 'address_space_bytes': 4 * 1024**3, 'whole_seconds': 60}}
    dump(state)
    for file in ('check_cut.py', 'check_master.py'):
        with (ROOT / (file + '.log')).open('w') as log:
            check = subprocess.run([sys.executable, '-B', str(SOURCE / file)], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=15, preexec_fn=bounded)
        if check.returncode:
            state.update(status='STOP_FIXTURE_FAILED', fixture=file, fixture_exit=check.returncode)
            dump(state)
            return
    state['fixture_exit'] = 0
    dump(state)
    start = time.monotonic()
    command = [sys.executable, '-B', str(SOURCE / 'joint.py'), '--instance_path', str(INPUT),
               '--prior_path', str(PRIOR), '--output_dir', str(ROOT / 'gate'), '--seed', '1']
    with (ROOT / 'algorithm.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                preexec_fn=bounded, start_new_session=True)
        state['worker_pid'] = proc.pid
        dump(state)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
            state['failure'] = 'WHOLE_60S_TIMEOUT'
    state.update(algorithm_exit=code, algorithm_whole_seconds=time.monotonic() - start)
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
    actual = [trial for trial in metrics['trials'] if trial.get('termination') == 'RETURNED_FULL_JOINT_WITNESS'
              and trial.get('oracle', {}).get('native_result_available')
              and trial['oracle'].get('termination') == 'RETURNED_VERIFIED_FIXED_S'
              and trial['oracle'].get('fixed_s_model_status') == 'HighsModelStatus.kOptimal']
    improves = any(trial['improvement_fraction'] >= .005 for trial in actual)
    state.update(status='DONE', double_checks=results, metrics=metrics,
                 gate='PASS' if improves and metrics['extra_seconds'] <= 20.0 and metrics['termination'] == 'JOINT_GATE_COMPLETE' else 'STOP',
                 complete_joint_oracles=len(actual), whole_with_checkers_seconds=time.monotonic() - start)
    dump(state)


if __name__ == '__main__':
    main()
