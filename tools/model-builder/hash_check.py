#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""Reproduce the pack's pinned SHARP hashes with a frozen dxr-model-builder, run exactly the way
the runtime's model store runs it (conversion-provider-discovery spec section 5.6.2 / 5.6.3):
a staging dir <root>/staging/<nonce>/{source,work,out}, the checkpoint at source/checkpoint (no
extension), argv only, working directory work/, stdin the null device, stderr to a log, its own
process group, and the minimal environment (POSIX: PATH=/usr/bin:/bin:/usr/sbin:/sbin,
LANG/LC_ALL=C.UTF-8, HOME/TMPDIR/TMP/TEMP=work/). Then it hashes every regular file in out/ and
matches (sha256, size) against packs/dxr-open-sharp.models.json, as the store's ingest does.

    python -I tools/model-builder/hash_check.py <dxr-model-builder> --variant int4 [--variant ...]
        --root <dir> (--checkpoint <file> | --download) [--one-run]

One build per variant by default, deleting its staging dir before the next, so at most one
model's files are on disk at a time; --one-run builds every --variant in one run (one export).
The checkpoint is linked (or copied) into source/ and never deleted from where it came from;
--download fetches it from the pack's official URL into <root>/checkpoint once. Exit 0 = every
requested variant's files match the pins. Prints no weight and keeps no output.
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACK = os.path.join(REPO, "packs", "dxr-open-sharp.models.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_pins():
    """variant -> (model name, {(sha256, size)}), and the checkpoint source."""
    pack = json.load(open(PACK, encoding="utf-8"))
    pins, source = {}, None
    for m in pack["models"]:
        lb = m.get("local_build") or {}
        if "variant" not in lb:
            continue
        files = {(f["sha256"], f["size"]) for v in m["variants"] for f in v["files"]}
        pins[lb["variant"]] = (m["name"], files)
        for s in lb.get("sources", []):
            if s["role"] == "checkpoint":
                source = s
    return pins, source


def min_env(work):
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    for k in ("HOME", "TMPDIR", "TMP", "TEMP"):
        env[k] = work
    return env


def download(source, dest):
    if os.path.isfile(dest) and os.path.getsize(dest) == source["size"]:
        return dest
    print(f"downloading {source['url']} ({source['size'] / 1e9:.2f} GB)", flush=True)
    h, done = hashlib.sha256(), 0
    req = urllib.request.Request(source["url"], headers={"User-Agent": "displayxr-model-builder-ci"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest + ".part", "wb") as out:
        for chunk in iter(lambda: r.read(1 << 22), b""):
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
    if done != source["size"] or h.hexdigest() != source["sha256"]:
        sys.exit(f"downloaded {done} bytes, sha256 {h.hexdigest()}: not the pinned checkpoint")
    os.replace(dest + ".part", dest)
    return dest


def run_build(builder, root, checkpoint, variants, licence):
    nonce = secrets.token_hex(8)
    stg = os.path.join(root, "staging", nonce)
    d = {k: os.path.join(stg, k) for k in ("source", "work", "out")}
    for p in d.values():
        os.makedirs(p)
    src = os.path.join(d["source"], "checkpoint")
    try:
        os.link(checkpoint, src)
    except OSError:
        shutil.copyfile(checkpoint, src)
    cancel = os.path.join(stg, "cancel")
    log = os.path.join(root, f"sharp-onnx-{nonce}.log")
    argv = [os.path.abspath(builder), "build", "--recipe", "sharp-onnx"]
    for v in variants:
        argv += ["--variant", v]
    argv += ["--source", f"checkpoint={src}", "--out", d["out"], "--work", d["work"],
             "--cancel-file", cancel, "--accept-license", licence]
    t0 = time.monotonic()
    events, bad, last_out, max_gap = [], 0, t0, 0.0
    with open(log, "ab") as errf, open(os.devnull, "rb") as nul:
        p = subprocess.Popen(argv, cwd=d["work"], env=min_env(d["work"]), stdin=nul, stdout=subprocess.PIPE,
                             stderr=errf, start_new_session=True)
        try:
            for raw in p.stdout:
                now = time.monotonic()
                max_gap = max(max_gap, now - last_out)
                last_out = now
                line = raw.decode("utf-8", "replace").rstrip("\n")
                try:
                    ev = json.loads(line)
                    ok = isinstance(ev, dict) and ev.get("v") == 1 and "event" in ev and len(raw) <= 16384
                except ValueError:
                    ok = False
                if not ok:
                    bad += 1
                    print(f"  NOT AN EVENT LINE: {line[:200]}", flush=True)
                    continue
                events.append(ev)
                if ev["event"] != "progress":
                    print(f"  {int(now - t0):5d}s {line}", flush=True)
            code = p.wait()
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL)
            raise
    got = {}
    for n in os.listdir(d["out"]):
        f = os.path.join(d["out"], n)
        if os.path.isfile(f) and not os.path.islink(f):
            got[n] = (sha256_file(f), os.path.getsize(f))
    subdirs = [n for n in os.listdir(d["out"]) if os.path.isdir(os.path.join(d["out"], n))]
    shutil.rmtree(stg, ignore_errors=True)
    return {"code": code, "events": events, "bad_lines": bad, "files": got, "subdirs": subdirs,
            "seconds": int(time.monotonic() - t0), "max_stdout_gap": round(max_gap, 1), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("builder")
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--one-run", action="store_true")
    a = ap.parse_args()
    pins, source = pack_pins()
    os.makedirs(a.root, exist_ok=True)
    if a.download:
        checkpoint = download(source, os.path.join(a.root, "checkpoint"))
    elif a.checkpoint:
        checkpoint = os.path.abspath(a.checkpoint)
    else:
        sys.exit("--checkpoint <file> or --download")
    licence = source["license"]["id"]
    runs = [a.variant] if a.one_run else [[v] for v in a.variant]
    failed = []
    for vs in runs:
        print(f"== build {' '.join(vs)}", flush=True)
        r = run_build(a.builder, a.root, checkpoint, vs, licence)
        want = set().union(*(pins[v][1] for v in vs))
        have = set(r["files"].values())
        names = [e["event"] for e in r["events"]]
        ok = (r["code"] == 0 and have == want and not r["bad_lines"] and not r["subdirs"]
              and names[:1] == ["start"] and names[-1:] == ["done"] and r["events"][-1].get("ok") is True)
        for v in vs:
            for sha, size in sorted(pins[v][1]):
                mark = "MATCH" if (sha, size) in have else "MISSING"
                print(f"  {v:5s} {pins[v][0]:16s} {sha} {size:>11d} {mark}")
        for sha, size in sorted(have - want):
            print(f"  UNEXPECTED {sha} {size}")
        print(f"  exit {r['code']}, {r['seconds']} s, longest stdout gap {r['max_stdout_gap']} s, "
              f"{len(r['events'])} events, log {r['log']}: {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failed.append(" ".join(vs))
    print("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
