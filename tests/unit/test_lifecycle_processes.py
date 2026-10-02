"""Exercise real backend processes and process-group ownership."""

from contextlib import suppress
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from threading import Thread
import time

from easyllama.lifecycle import Backend
import pytest

pytestmark = pytest.mark.unit


def wait_for_file(path: Path) -> str:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and (payload := path.read_text()):
            return payload
        time.sleep(0.01)
    raise AssertionError("backend did not become ready")


def fast_shutdown(monkeypatch):
    original = subprocess.Popen.wait

    def wait(proc, timeout=None):
        return original(proc, min(timeout, 0.2) if timeout is not None else None)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)


def test_repeated_wake_keeps_healthy_backend(monkeypatch):
    backend = Backend([sys.executable, "-c", "import time; time.sleep(60)"])
    original = backend.changed.wait_for
    monkeypatch.setattr(
        backend.changed, "wait_for", lambda predicate, timeout=None: original(predicate, 0.05)
    )
    try:
        pid = backend.wake()
        assert backend.wake() == pid
        assert backend.proc.poll() is None
    finally:
        backend.sleep()


def test_managed_llama_is_owned_and_releases_port_on_forced_sleep(tmp_path, monkeypatch):
    fast_shutdown(monkeypatch)
    ready = tmp_path / "ready"
    child = (
        "import os,signal,socket,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
        f"open({str(ready)!r},'w').write(str(os.getpid())+' '+str(s.getsockname()[1])); "
        "time.sleep(60)"
    )
    launcher = (
        "import sys; from easyllama.servers.llamacpp import LlamaCppServer; "
        "from easyllama.servers.base import Spec; "
        f"LlamaCppServer().run(Spec(cmd=[sys.executable,'-c',{child!r}]))"
    )
    backend = Backend([sys.executable, "-c", launcher])
    child_pid = None
    try:
        backend.wake()
        child_pid, port = map(int, wait_for_file(ready).split())
        assert os.getpgid(child_pid) == backend.proc.pid
        backend.sleep()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        assert backend.status()["state"] == "sleeping"
    finally:
        backend.sleep()
        if child_pid:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


def test_wake_waits_for_concurrent_sleep(tmp_path, monkeypatch):
    fast_shutdown(monkeypatch)
    ready = tmp_path / "ready"
    command = [
        sys.executable,
        "-c",
        (
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"open({str(ready)!r},'w').write('ready'); time.sleep(60)"
        ),
    ]
    backend = Backend(command)
    sleeper = None
    try:
        old_pid = backend.wake()
        wait_for_file(ready)
        sleeper = Thread(target=backend.sleep)
        sleeper.start()
        with backend.changed:
            assert backend.changed.wait_for(lambda: backend.stopping, timeout=2)
        new_pid = backend.wake()
        assert new_pid != old_pid
        sleeper.join(timeout=2)
        assert not sleeper.is_alive()
        assert backend.proc.pid == new_pid
        assert backend.proc.poll() is None
    finally:
        if sleeper:
            sleeper.join(timeout=2)
        backend.sleep()


def test_shutdown_error_preserves_owned_process(monkeypatch):
    backend = Backend([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pid = backend.wake()
        with monkeypatch.context() as patch:

            def fail_signal(*args):
                raise PermissionError("injected signal failure")

            patch.setattr(os, "killpg", fail_signal)
            with pytest.raises(PermissionError):
                backend.sleep()
        assert not backend.stopping
        assert backend.wake() == pid
    finally:
        backend.sleep()
