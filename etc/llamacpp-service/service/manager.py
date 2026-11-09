#!/usr/bin/env python3
"""按 Neovim 实例登记管理共享模型进程。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.request


# 与 Neovim 客户端共用的模型标识。
MODEL_ALIAS = "local-llm"


def process_start(pid):
    """读取存活进程的启动时间，用于识别 PID 复用。"""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] not in ("Z", "X") else None
    except (OSError, ValueError, IndexError):
        return None


def live_clients(directory):
    count = 0
    for lease in directory.glob("*.json"):
        try:
            data = json.loads(lease.read_text())
            start = process_start(data["pid"])
            if start is not None and start == data["start"]:
                count += 1
                continue
        except (OSError, ValueError, KeyError, TypeError):
            pass
        lease.unlink(missing_ok=True)
    return count


class Manager:
    def __init__(self, config, runtime):
        self.config, self.runtime = config, runtime
        self.child = None
        self.started = 0
        self.retry_at = 0
        self.idle_since = None
        self.last_status = None
        self.stop_event = threading.Event()
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def status(self, phase, detail=""):
        data = {
            "phase": phase, "detail": detail,
            "manager_pid": os.getpid(), "manager_start": process_start(os.getpid()),
            "server_pid": self.child.pid if self.child else None,
            "server_start": process_start(self.child.pid) if self.child else None,
            "alias": MODEL_ALIAS, "port": self.config["port"],
        }
        if data != self.last_status:
            tmp = self.runtime / "status.tmp"
            tmp.write_text(json.dumps(data))
            tmp.replace(self.runtime / "status.json")
            self.last_status = data
            print(f"{phase}: {detail}", flush=True)

    def launch(self):
        cfg = self.config
        server = shutil.which("llama-server")
        if server is None:
            raise FileNotFoundError("llama-server not found in the systemd service PATH")
        # 拒绝使用已被占用的端口。
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", cfg["port"]))
        command = [server, *cfg.get("args", []),
                   "--model", str(Path(cfg["model"]).expanduser()),
                   "--alias", MODEL_ALIAS, "--host", "127.0.0.1", "--port", str(cfg["port"])]
        self.child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        self.started = time.monotonic()
        self.status("loading")

    def stop(self):
        if self.child is not None:
            if self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            self.child = None

    def healthy(self):
        try:
            url = f"http://127.0.0.1:{self.config['port']}"
            with self.http.open(url + "/health", timeout=0.5) as response:
                if json.load(response).get("status") != "ok":
                    return False
            with self.http.open(url + "/v1/models", timeout=0.5) as response:
                return any(m.get("id") == MODEL_ALIAS for m in json.load(response).get("data", []))
        except (OSError, ValueError, TypeError):
            return False

    def tick(self):
        now = time.monotonic()
        clients = live_clients(self.runtime / "clients")
        if not clients:
            if self.idle_since is None:
                self.idle_since = now
            if now - self.idle_since >= self.config["idle_seconds"]:
                self.stop()
                self.status("idle")
            return
        self.idle_since = None
        if self.child is not None and self.child.poll() is not None:
            code = self.child.returncode
            self.child = None
            self.retry_at = now + 5
            self.status("error", f"llama-server exited with code {code}; see journalctl --user -u llamacpp-service")
        if self.child is None and now >= self.retry_at:
            try:
                self.launch()
            except OSError as error:
                self.retry_at = now + 5
                self.status("error", str(error))
        if self.child is not None:
            if self.healthy() and self.child.poll() is None:
                self.status("ready")
            elif now - self.started > self.config["startup_timeout"]:
                self.stop()
                self.retry_at = time.monotonic() + 5
                self.status("error", "Health check timed out")
            else:
                self.status("loading")

    def run(self):
        self.status("idle")
        try:
            while not self.stop_event.is_set():
                self.tick()
                self.stop_event.wait(0.5)
        finally:
            self.stop()
            self.status("idle")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime_path = os.environ.get("RUNTIME_DIRECTORY")
    runtime = Path(runtime_path) if runtime_path else Path(os.environ["XDG_RUNTIME_DIR"]) / "llamacpp-service"
    (runtime / "clients").mkdir(parents=True, exist_ok=True, mode=0o700)
    # 防止重复启动管理进程。
    with (runtime / "manager.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        manager = Manager(config, runtime)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: manager.stop_event.set())
        manager.run()


if __name__ == "__main__":
    main()
