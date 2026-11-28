#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""Contract checks on a built dxr-model-builder (the runtime's conversion-provider-discovery spec
section 5.6.2), with no checkpoint and no network: run it the way the model store does (argv
only, the minimal environment, working directory = --work) and check describe, stdout carrying
JSON event lines only, the exit codes for usage / licence / source mismatch, and cancel.

    python -I tools/model-builder/contract_test.py <path to dxr-model-builder[.exe]>

Also checks that no file the builder ships is a pinned model or source file (section 5.6.6).
Exit 0 = every check passed.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAILED = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILED.append(what)


def min_env(work):
    if os.name == "nt":
        sr = os.environ.get("SystemRoot", r"C:\Windows")
        env = {"SystemRoot": sr, "SystemDrive": os.environ.get("SystemDrive", "C:"), "windir": sr,
               "PATH": rf"{sr}\System32;{sr}", "NUMBER_OF_PROCESSORS": os.environ.get("NUMBER_OF_PROCESSORS", "2"),
               "PROCESSOR_ARCHITECTURE": os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")}
        for k in ("TMP", "TEMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HOME"):
            env[k] = work
    else:
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        for k in ("HOME", "TMPDIR", "TMP", "TEMP"):
            env[k] = work
    return env


def staging():
    root = tempfile.mkdtemp(prefix="mb-contract-")
    d = {k: os.path.join(root, k) for k in ("source", "work", "out")}
    for p in d.values():
        os.makedirs(p)
    d["cancel"] = os.path.join(root, "cancel")
    return d


def events(stdout):
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    parsed = []
    for ln in lines:
        try:
            o = json.loads(ln)
        except ValueError:
            return None
        if not (isinstance(o, dict) and o.get("v") == 1 and "event" in o and len(ln.encode()) < 16384):
            return None
        parsed.append(o)
    return parsed


def run_build(exe, s, extra, timeout=120):
    argv = [exe, "build", "--recipe", "sharp-onnx", *extra, "--out", s["out"], "--work", s["work"],
            "--cancel-file", s["cancel"]]
    p = subprocess.run(argv, cwd=s["work"], env=min_env(s["work"]), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, events(p.stdout), p.stderr


def main():
    exe = os.path.abspath(sys.argv[1])
    with open(os.path.join(REPO, "packs", "dxr-open-sharp.models.json"), encoding="utf-8") as f:
        pack = json.load(f)
    src = pack["models"][0]["local_build"]["sources"][0]

    # describe --json: one object, exit 0, within 60 s, pins equal to the pack's
    scratch = tempfile.mkdtemp(prefix="mb-describe-")
    t0 = time.monotonic()
    p = subprocess.run([exe, "describe", "--json"], cwd=scratch, env=min_env(scratch),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    took = time.monotonic() - t0
    check(p.returncode == 0, f"describe exits 0 ({took:.1f} s)")
    try:
        d = json.loads(p.stdout)
    except ValueError:
        d = {}
    check(d.get("v") == 1 and d.get("tool") == "open-model-builder", "describe: v 1, tool open-model-builder")
    rec = next((r for r in d.get("recipes", []) if r.get("recipe") == "sharp-onnx"), {})
    pin = (rec.get("sources") or [{}])[0]
    check(pin.get("sha256") == src["sha256"] and pin.get("size") == src["size"]
          and pin.get("license_id") == src["license"]["id"], "describe: checkpoint pin == the pack's source")
    want_variants = {m["local_build"]["variant"] for m in pack["models"]}
    check(want_variants <= set(rec.get("variants", [])), f"describe: variants cover the pack's {sorted(want_variants)}")
    mins = {m["local_build"]["tool"]["min_version"] for m in pack["models"]}
    ver = tuple(int(x) for x in str(d.get("version", "0.0.0")).split("."))
    check(all(ver >= tuple(int(x) for x in m.split(".")) for m in mins), f"describe: version {d.get('version')} >= min_version")

    # usage: unknown recipe / variant
    s = staging()
    rc, ev, _ = run_build(exe, s, ["--variant", "nope", "--source", "checkpoint=x", "--accept-license", "apple-amlr"])
    check(rc == 2 and ev is not None and ev[-1] == {"v": 1, "event": "done", "ok": False}, "unknown variant: exit 2, done ok=false")
    # licence: missing, before any file read
    rc, ev, _ = run_build(exe, s, ["--variant", "int4", "--source", f"checkpoint={os.path.join(s['source'], 'absent')}"])
    check(rc == 10 and ev is not None and any(e.get("code") == "license" for e in ev), "no --accept-license: exit 10")
    rc, _, _ = run_build(exe, s, ["--variant", "int4", "--source", "checkpoint=x", "--accept-license", "other"])
    check(rc == 10, "wrong --accept-license: exit 10")
    # source mismatch: wrong size (a file with no extension, as the store passes it)
    bad = os.path.join(s["source"], "checkpoint")
    with open(bad, "wb") as f:
        f.write(b"\0" * 1024)
    rc, ev, _ = run_build(exe, s, ["--variant", "int4", "--source", f"checkpoint={bad}", "--accept-license", "apple-amlr"])
    check(rc == 12 and ev is not None and ev[0].get("event") == "start", "wrong source size: exit 12, start first")
    # cancel during the source check: a sparse file of the pinned size hashes for seconds
    with open(bad, "wb") as f:
        f.truncate(src["size"])
    argv = [exe, "build", "--recipe", "sharp-onnx", "--variant", "int4", "--source", f"checkpoint={bad}",
            "--out", s["out"], "--work", s["work"], "--cancel-file", s["cancel"], "--accept-license", "apple-amlr"]
    proc = subprocess.Popen(argv, cwd=s["work"], env=min_env(s["work"]), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(3)
    open(s["cancel"], "w").close()
    t0 = time.monotonic()
    try:
        out, _ = proc.communicate(timeout=10)
        took = time.monotonic() - t0
        ev = events(out)
        check(proc.returncode == 17 and ev is not None and ev[-1].get("event") == "done",
              f"cancel file: exit 17 within 10 s ({took:.1f} s)")
        check(ev is not None and any(e.get("event") == "progress" for e in ev), "progress lines while hashing")
    except subprocess.TimeoutExpired:
        proc.kill()
        check(False, "cancel file: exit 17 within 10 s")
    os.remove(bad)
    check(not os.listdir(s["out"]), "nothing written to --out on a failed run")

    # no shipped file is a pinned model or source file
    pinned = {src["sha256"]}
    for m in pack["models"]:
        for v in m.get("variants", []):
            pinned |= {fl["sha256"] for fl in v.get("files", [])}
    hit = []
    for dirpath, _, names in os.walk(os.path.dirname(exe)):
        for n in names:
            fp = os.path.join(dirpath, n)
            if os.path.getsize(fp) < 1024 * 1024:
                continue
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            if h.hexdigest() in pinned:
                hit.append(fp)
    check(not hit, f"no shipped file is a pinned model or source file {hit or ''}")

    print("PASSED" if not FAILED else f"FAILED: {len(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
