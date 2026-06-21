#!/usr/bin/env python3
# fs_bridge.py -- POD-SIDE half of the shared-filesystem OpenAI bridge.
#
# Runs INSIDE the cluster pod (started by launch_gpt_oss_serve.sh), where it can
# reach the local vLLM server at http://127.0.0.1:<port>. The dev box, which has
# NO network route to the pod (only the /share5 NFS is shared), talks to the model
# purely by exchanging files in a shared directory via fs_proxy.py.
#
# Protocol (all files under --dir, on the shared NFS):
#   req/<id>.json    client -> bridge : {"id","method","path","body","headers","ts"}
#   resp/<id>.json   bridge -> client : {"id","status","body","error","ts"}
#                    The client PRE-CREATES this as {"status":0,"pending":true} and
#                    the bridge OVERWRITES it, so the client polls file CONTENT
#                    (fast NFS revalidation, acreg ~3s) instead of file EXISTENCE
#                    (slow negative-lookup caching, acdir ~30s).
#   bridge.alive     heartbeat {"ts","pid","inflight"} refreshed every --heartbeat s.
# All writes are atomic: write ".<name>.<rnd>.tmp" in the SAME dir, then os.replace
# (atomic rename within one NFS directory).
#
# The bridge claims each request by renaming req/<id>.json -> req/.<id>.busy before
# forwarding, so a re-scan never double-processes, and a crash leaves a .busy file
# that the age-based sweeper reclaims.

from __future__ import annotations
import argparse
import asyncio
import json
import os
import secrets
import time

import httpx


def atomic_write(path: str, obj) -> None:
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, f".{os.path.basename(path)}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


async def handle(client, resp_dir, target, busy_path, rid, sem, timeout):
    try:
        with open(busy_path) as f:
            reqo = json.load(f)
    except Exception:
        try:
            os.remove(busy_path)
        except OSError:
            pass
        return
    method = (reqo.get("method") or "POST").upper()
    path = reqo.get("path") or "/v1/chat/completions"
    body = reqo.get("body")
    headers = reqo.get("headers") or {}
    url = target.rstrip("/") + path
    async with sem:
        try:
            r = await client.request(
                method, url,
                json=body if body is not None else None,
                headers=headers, timeout=timeout,
            )
            try:
                rbody = r.json()
            except Exception:
                rbody = {"raw": r.text}
            resp = {"id": rid, "status": r.status_code, "body": rbody, "error": None, "ts": time.time()}
        except Exception as e:  # noqa: BLE001
            resp = {"id": rid, "status": 0, "body": None, "error": f"{type(e).__name__}: {e}", "ts": time.time()}
    try:
        atomic_write(os.path.join(resp_dir, f"{rid}.json"), resp)
    finally:
        try:
            os.remove(busy_path)
        except OSError:
            pass


async def main_async(a):
    req_dir = os.path.join(a.dir, "req")
    resp_dir = os.path.join(a.dir, "resp")
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)
    # clear stale tmp/busy from a previous crashed run
    for d in (req_dir, resp_dir):
        for n in os.listdir(d):
            if n.endswith(".tmp") or n.endswith(".busy"):
                try:
                    os.remove(os.path.join(d, n))
                except OSError:
                    pass

    sem = asyncio.Semaphore(a.concurrency)
    inflight: set[str] = set()
    alive_path = os.path.join(a.dir, "bridge.alive")
    last_hb = 0.0
    last_clean = 0.0
    print(f"[bridge] up: dir={a.dir} target={a.target} concurrency={a.concurrency} poll={a.poll}s", flush=True)

    limits = httpx.Limits(max_connections=a.concurrency + 16, max_keepalive_connections=a.concurrency + 16)
    async with httpx.AsyncClient(limits=limits) as client:
        while True:
            now = time.time()
            if now - last_hb >= a.heartbeat:
                try:
                    atomic_write(alive_path, {"ts": now, "pid": os.getpid(), "inflight": len(inflight)})
                except Exception:
                    pass
                last_hb = now

            try:
                entries = os.listdir(req_dir)
            except FileNotFoundError:
                entries = []
            for n in entries:
                if not n.endswith(".json") or n.startswith("."):
                    continue
                rid = n[:-5]
                if rid in inflight:
                    continue
                busy = os.path.join(req_dir, f".{rid}.busy")
                try:
                    os.replace(os.path.join(req_dir, n), busy)   # atomic claim
                except FileNotFoundError:
                    continue
                inflight.add(rid)
                t = asyncio.create_task(handle(client, resp_dir, a.target, busy, rid, sem, a.timeout))
                t.add_done_callback(lambda _t, _rid=rid: inflight.discard(_rid))

            if now - last_clean >= 30:
                for d, match in ((resp_dir, ".json"), (req_dir, ".busy")):
                    for n in os.listdir(d):
                        if not n.endswith(match):
                            continue
                        p = os.path.join(d, n)
                        try:
                            if now - os.path.getmtime(p) > a.ttl:
                                os.remove(p)
                        except OSError:
                            pass
                last_clean = now

            await asyncio.sleep(a.poll)


def main():
    p = argparse.ArgumentParser(description="pod-side OpenAI-over-files bridge")
    p.add_argument("--dir", required=True, help="shared NFS bridge dir (has req/ resp/)")
    p.add_argument("--target", default="http://127.0.0.1:30000", help="local vLLM base")
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--poll", type=float, default=0.5, help="inbox scan interval (s)")
    p.add_argument("--heartbeat", type=float, default=5.0)
    p.add_argument("--timeout", type=float, default=1800.0, help="per-request upstream timeout")
    p.add_argument("--ttl", type=float, default=3600.0, help="age (s) after which orphan files are swept")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
