#!/usr/bin/env python3
# fs_proxy.py -- DEV-BOX-SIDE half of the shared-filesystem OpenAI bridge.
#
# Serves an OpenAI-compatible HTTP endpoint locally and forwards every request to
# the pod over the shared NFS dir (paired with fs_bridge.py in the pod). This lets
# eval_ckpts.py / curl reach the remote gpt-oss server WITHOUT any network route to
# the pod -- only the shared /share5 NFS is required.
#
#   client HTTP --> [this shim] --> req/<id>.json (NFS) --> [pod fs_bridge] --> vLLM
#   vLLM --> [pod fs_bridge] --> resp/<id>.json (NFS) --> [this shim] --> client HTTP
#
# Stdlib only (ThreadingHTTPServer): one thread per connection, each polls its own
# response file. The client pre-creates the response placeholder so it polls file
# CONTENT (fast NFS revalidation) rather than existence.

from __future__ import annotations
import argparse
import json
import os
import secrets
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIR = None
TIMEOUT = 1800.0
POLL = 0.4
HB_MAX = 30.0


def atomic_write(path: str, obj) -> None:
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, f".{os.path.basename(path)}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def bridge_alive() -> bool:
    try:
        with open(os.path.join(DIR, "bridge.alive")) as f:
            hb = json.load(f)
        return (time.time() - float(hb.get("ts", 0))) < HB_MAX
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, status: int, obj) -> None:
        b = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _bridge(self, method: str) -> None:
        # liveness ping (GET /) -- handy for "wait for shim up" loops
        if method == "GET" and self.path.rstrip("/") in ("", "/"):
            self._send(200, {"ok": True, "bridge_alive": bridge_alive(), "dir": DIR})
            return

        req_dir = os.path.join(DIR, "req")
        resp_dir = os.path.join(DIR, "resp")
        os.makedirs(req_dir, exist_ok=True)
        os.makedirs(resp_dir, exist_ok=True)
        if not bridge_alive():
            self._send(503, {"error": {"message": "fs bridge heartbeat stale/missing; is the pod serving?",
                                       "type": "fs_bridge_down"}})
            return

        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        body = None
        if raw:
            try:
                body = json.loads(raw)
            except Exception:
                body = None

        rid = uuid.uuid4().hex
        resp_path = os.path.join(resp_dir, f"{rid}.json")
        # pre-create placeholder -> poll CONTENT (acreg ~3s), not existence (acdir ~30s)
        atomic_write(resp_path, {"id": rid, "status": 0, "pending": True})
        atomic_write(os.path.join(req_dir, f"{rid}.json"),
                     {"id": rid, "method": method, "path": self.path, "body": body, "ts": time.time()})

        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            try:
                with open(resp_path) as f:
                    ro = json.load(f)
            except Exception:
                time.sleep(POLL)
                continue
            if ro.get("pending") or int(ro.get("status", 0)) == 0:
                time.sleep(POLL)
                continue
            try:
                os.remove(resp_path)
            except OSError:
                pass
            if ro.get("error"):
                self._send(502, {"error": {"message": ro["error"], "type": "fs_bridge_upstream"}})
            else:
                self._send(int(ro["status"]), ro.get("body") if ro.get("body") is not None else {})
            return

        try:
            os.remove(resp_path)
        except OSError:
            pass
        self._send(504, {"error": {"message": "fs bridge timeout waiting for the pod",
                                   "type": "fs_bridge_timeout"}})

    def do_POST(self):
        self._bridge("POST")

    def do_GET(self):
        self._bridge("GET")


def main():
    global DIR, TIMEOUT, POLL
    ap = argparse.ArgumentParser(description="dev-box OpenAI shim over the file bridge")
    ap.add_argument("--dir", required=True, help="shared NFS bridge dir (matches fs_bridge --dir)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=38900)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--poll", type=float, default=0.4)
    a = ap.parse_args()
    DIR, TIMEOUT, POLL = a.dir, a.timeout, a.poll
    os.makedirs(os.path.join(DIR, "req"), exist_ok=True)
    os.makedirs(os.path.join(DIR, "resp"), exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[shim] OpenAI shim http://{a.host}:{a.port}/v1 -> files {DIR}  (bridge_alive={bridge_alive()})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
