"""Pool workers end when the process that started them dies without shutting the pool down (a native crash)."""
import os
import subprocess
import sys
import textwrap
import time


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_workers_end_with_their_parent(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(textwrap.dedent("""
        import concurrent.futures as cf, multiprocessing, os, sys
        from sealed_lab.procguard import watch_parent

        if __name__ == "__main__":
            ex = cf.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=watch_parent)
            pids = sorted({f.result() for f in [ex.submit(os.getpid) for _ in range(6)]})
            print(" ".join(map(str, pids)), flush=True)
            os._exit(3)                       # die like a native crash: no pool shutdown
    """), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
    pids = [int(p) for p in out.stdout.split()]
    assert out.returncode == 3 and pids, out.stderr
    deadline = time.time() + 20
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.5)
    assert not any(_alive(p) for p in pids)
