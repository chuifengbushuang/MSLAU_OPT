"""Wait for recorded existing jobs, then run the authorized official-split study."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        tail=(Path('/proc')/str(pid)/'stat').read_text().rpartition(')')[2].split()
        return dict(pid=int(pid),start_ticks=tail[19],state=tail[0])
    except (FileNotFoundError,ProcessLookupError):
        return None


def same_process(expected):
    current=process_identity(expected['pid'])
    return bool(current and current['start_ticks']==expected['start_ticks'] and current['state'] not in ('Z','X'))


def dependency_state(dependency):
    failure=dependency.get('failure_file')
    if failure and Path(failure).exists():
        raise RuntimeError(f"Prior task failed: {dependency['name']}; inspect {failure}")
    alive=[p['pid'] for p in dependency.get('processes',[]) if same_process(p)]
    try:
        state=json.loads(Path(dependency['state_file']).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return dict(name=dependency['name'],ready=False,status='state_not_ready',alive=alive)
    complete=all(state.get(k)==v for k,v in dependency['expected'].items())
    if not complete and not alive:
        raise RuntimeError(f"Prior task disappeared without completion: {dependency['name']}")
    return dict(name=dependency['name'],ready=complete and not alive,
                status=state.get('status'),alive=alive)


def verify_signature(plan):
    import hashlib
    for name,expected in plan['source_sha256'].items():
        actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Queued study source changed before launch: {name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    verify_signature(plan)
    state=[dependency_state(d) for d in plan['dependencies']]
    print(json.dumps(dict(command=plan['command'],dependencies=state,run=args.run),ensure_ascii=False),flush=True)
    if not args.run:return
    directory=args.plan.resolve().parent
    lock=(directory/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/'dispatch.json').exists():
        raise RuntimeError('This queue was already dispatched; inspect instead of restarting')
    previous=None
    try:
        while True:
            state=[dependency_state(d) for d in plan['dependencies']]
            progress=dict(status='waiting',queue_pid=os.getpid(),dependencies=state,
                command=plan['command'],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(directory/'queue_progress.json',progress)
            key=json.dumps(state,sort_keys=True)
            if key != previous:
                print('WAITING',key,flush=True);previous=key
            if all(d['ready'] for d in state):break
            time.sleep(45)
        verify_signature(plan)
        if Path(plan['target_directory']).exists():
            raise RuntimeError('Target run already exists; refusing duplicate study')
        with (directory/'study.console.log').open('x') as stream:
            child=subprocess.Popen(plan['command'],cwd=plan['cwd'],stdout=stream,stderr=subprocess.STDOUT)
            write(directory/'dispatch.json',dict(child_pid=child.pid,command=plan['command'],
                dispatched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            write(directory/'queue_progress.json',dict(status='running',child_pid=child.pid,queue_pid=os.getpid(),
                output_directory=plan['target_directory']))
            def stop(signum,frame):
                if child.poll() is None:child.send_signal(signum)
                raise SystemExit(128+signum)
            handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
            try:code=child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait()
                for sig,handler in handlers.items():signal.signal(sig,handler)
        if code:raise RuntimeError(f'New study failed ({code}); inspect study.console.log')
        write(directory/'queue_progress.json',dict(status='completed',
            output_directory=plan['target_directory']))
    except BaseException as exc:
        write(directory/'queue_progress.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':main()
