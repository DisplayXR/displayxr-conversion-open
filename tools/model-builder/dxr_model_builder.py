#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""DisplayXR open model builder: builds research-only models on the user's machine from the
official upstream checkpoint, so DisplayXR never hosts the weights or any file made from them.

    dxr-model-builder describe --json
    dxr-model-builder build --recipe <id> --variant <v> [--variant <v> ...]
                            --source <role>=<path | https-url> [...]
                            --out <dir> --work <dir> --cancel-file <path>
                            --accept-license <licence-id> [...] [--threads <n>]

The contract is the runtime's conversion-provider-discovery spec, section 5.6.2: progress is
JSON lines on stdout (nothing else goes there), human text on stderr, the exit codes below, and
writes only into --work and --out. The one recipe today is `sharp-onnx` (scripts/export_sharp.py,
Apple's SHARP: image -> Gaussian splats), variants `int4`, `full`, `fp16`.
"""

import argparse
import errno
import json
import os
import signal
import sys
import threading
import time

TOOL_ID = "open-model-builder"
TOOL_VERSION = "1.0.0"

EXIT = {"ok": 0, "internal": 1, "usage": 2, "license": 10, "download": 11, "source-mismatch": 12,
        "output-mismatch": 13, "memory": 14, "disk": 15, "unsupported": 16, "cancelled": 17}

# ---------------------------------------------------------------------------- events

_EVENT_OUT = sys.stdout          # the real stdout: JSON event lines only
_EVENT_LOCK = threading.Lock()
_FINISHED = threading.Event()    # set once the last line (done) is written


def emit(event, **keys):
    line = json.dumps({"v": 1, "event": event, **keys}, separators=(",", ":"), ensure_ascii=False)
    with _EVENT_LOCK:
        if _FINISHED.is_set():
            return
        _EVENT_OUT.write(line + "\n")
        _EVENT_OUT.flush()


def finish(code_name, message=None):
    """Write the error (if any) and done lines, then leave at once with the contract's code."""
    code = EXIT[code_name]
    with _EVENT_LOCK:
        if _FINISHED.is_set():
            os._exit(code)
        if code:
            _EVENT_OUT.write(json.dumps({"v": 1, "event": "error", "code": code_name,
                                         "message": message or code_name}) + "\n")
        _EVENT_OUT.write(json.dumps({"v": 1, "event": "done", "ok": code == 0}) + "\n")
        _EVENT_OUT.flush()
        _FINISHED.set()
    if message:
        sys.stderr.write(f"dxr-model-builder: {code_name}: {message}\n")
    sys.stderr.flush()
    os._exit(code)


class BuildError(Exception):
    def __init__(self, code_name, message):
        super().__init__(message)
        self.code_name = code_name


# ---------------------------------------------------------------------------- recipes


def _sharp():
    """scripts/export_sharp.py, imported as a module (beside this file in the frozen builder)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (getattr(sys, "_MEIPASS", None), os.path.join(here, "..", "..", "scripts")):
        if d and os.path.isfile(os.path.join(d, "export_sharp.py")) and d not in sys.path:
            sys.path.insert(0, d)
    import export_sharp
    return export_sharp


SHARP_VARIANTS = ("int4", "full", "fp16")
SHARP_LICENSE = "apple-amlr"
# The build's peak (sources + work files + output, and memory): the pack's local_build.needs.
SHARP_NEEDS = {"ram_mb": 12288, "disk_mb": 12288, "os": ["windows>=10.0.19041", "linux", "macos"]}


def describe():
    es = _sharp()
    return {"v": 1, "tool": TOOL_ID, "version": TOOL_VERSION, "recipes": [{
        "recipe": "sharp-onnx",
        "variants": list(SHARP_VARIANTS),
        "sources": [{"role": "checkpoint", "sha256": es.CHECKPOINT_SHA256, "size": es.CHECKPOINT_SIZE,
                     "license_id": SHARP_LICENSE}],
        "needs": SHARP_NEEDS,
    }]}


# ---------------------------------------------------------------------------- build helpers


def _point_caches_at(work):
    """Every library cache and temp dir under --work (the store hands us a minimal environment)."""
    import tempfile

    tmp = os.path.join(work, "tmp")
    os.makedirs(tmp, exist_ok=True)
    for k in ("TMP", "TEMP", "TMPDIR"):
        os.environ[k] = tmp
    for k, sub in (("HOME", ""), ("USERPROFILE", ""), ("XDG_CACHE_HOME", "cache"), ("TORCH_HOME", "cache/torch"),
                   ("HF_HOME", "cache/hf"), ("PYTHONPYCACHEPREFIX", "cache/pyc"),
                   ("MPLCONFIGDIR", "cache/mpl"), ("TORCHINDUCTOR_CACHE_DIR", "cache/inductor")):
        os.environ[k] = os.path.join(work, *sub.split("/")) if sub else work
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    tempfile.tempdir = tmp


def _watch_cancel(path):
    """Poll the cancel file twice a second; on it, exit 17 at once (the store deletes staging)."""
    def run():
        while not _FINISHED.is_set():
            if path and os.path.exists(path):
                finish("cancelled", "the cancel file appeared")
            time.sleep(0.5)
    threading.Thread(target=run, name="cancel-watch", daemon=True).start()


def _on_signal(signum, _frame):
    finish("cancelled", f"signal {signum}")


class _Heartbeat:
    """progress lines while one long call runs (the store's watchdog only sees stdout)."""

    def __init__(self, phase, every=20.0):
        self.phase, self.every, self.stop = phase, every, threading.Event()

    def __enter__(self):
        t0 = time.monotonic()

        def run():
            while not self.stop.wait(self.every):
                emit("progress", phase=self.phase, done=int(time.monotonic() - t0), total=0)
        threading.Thread(target=run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()


def _classify(exc):
    """An exception from the export -> (code name, message)."""
    if isinstance(exc, BuildError):
        return exc.code_name, str(exc)
    if isinstance(exc, MemoryError):
        return "memory", "out of memory"
    if isinstance(exc, OSError) and (exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) in (112, 39)):
        return "disk", f"out of disk space: {exc}"
    text = str(exc)
    low = text.lower()
    if any(s in low for s in ("can't allocate memory", "not enough memory", "out of memory",
                              "bad_alloc", "defaultcpuallocator")):
        return "memory", text.splitlines()[0][:500] if text else "out of memory"
    if "no space left" in low or "not enough space on the disk" in low:
        return "disk", text.splitlines()[0][:500]
    return "internal", f"{type(exc).__name__}: {text.splitlines()[0][:500] if text else ''}"


def _fetch_source(es, url, work):
    """Standalone use: --source checkpoint=https://... (exit 11 on a failed download)."""
    import hashlib
    import urllib.request

    if not url.startswith("https://"):
        raise BuildError("usage", f"a source URL must be https: {url}")
    dest_dir = os.path.join(work, "source")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "checkpoint")
    emit("phase", phase="download-source", variant=None, step=1, steps=1)
    h, done, last = hashlib.sha256(), 0, 0.0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"{TOOL_ID}/{TOOL_VERSION}"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest + ".part", "wb") as out:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if done > es.CHECKPOINT_SIZE:
                    break
                now = time.monotonic()
                if now - last >= 0.5:
                    emit("progress", phase="download-source", done=done, total=es.CHECKPOINT_SIZE)
                    last = now
    except OSError as e:
        if isinstance(e, OSError) and (e.errno == errno.ENOSPC or getattr(e, "winerror", None) == 112):
            raise BuildError("disk", f"out of disk space downloading {url}") from e
        raise BuildError("download", f"{url}: {e}") from e
    emit("progress", phase="download-source", done=done, total=es.CHECKPOINT_SIZE)
    if done != es.CHECKPOINT_SIZE or h.hexdigest() != es.CHECKPOINT_SHA256:
        raise BuildError("source-mismatch", f"{url}: {done} bytes, sha256 {h.hexdigest()}: not the pinned checkpoint")
    os.replace(dest + ".part", dest)
    return dest


def _verify_source(es, path):
    """The pinned size and sha256, before a byte of it is used (exit 12 otherwise)."""
    import hashlib

    if not os.path.isfile(path):
        raise BuildError("usage", f"source file not found: {path}")
    size = os.path.getsize(path)
    emit("phase", phase="verify-source", variant=None, step=1, steps=1)
    if size != es.CHECKPOINT_SIZE:
        raise BuildError("source-mismatch", f"{path}: {size} bytes, the pinned checkpoint is {es.CHECKPOINT_SIZE}")
    h, done, last = hashlib.sha256(), 0, 0.0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last >= 0.5:
                emit("progress", phase="verify-source", done=done, total=size)
                last = now
    emit("progress", phase="verify-source", done=done, total=size)
    if h.hexdigest() != es.CHECKPOINT_SHA256:
        raise BuildError("source-mismatch", f"{path}: sha256 {h.hexdigest()} is not the pinned {es.CHECKPOINT_SHA256}")


def build_sharp(args, variants, sources):
    es = _sharp()

    def fail(msg):
        raise BuildError("internal", msg)
    es.fail = fail                    # export_sharp's own checks become exit codes, not sys.exit(1)

    work, out = os.path.abspath(args.work), os.path.abspath(args.out)
    src = sources["checkpoint"]
    if src.startswith(("https://", "http://")):
        checkpoint = _fetch_source(es, src, work)
    else:
        checkpoint = os.path.abspath(src)
    _verify_source(es, checkpoint)    # also after a download: the file on disk is what we read

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    # prepare, export, then one convert + write per variant
    steps = 2 + 2 * len(variants)
    step = 1
    emit("phase", phase="prepare", variant=None, step=step, steps=steps)
    ml_sharp_src = es.prepare_ml_sharp(None, work)
    with _Heartbeat("prepare"):
        predictor = es.load_predictor(ml_sharp_src, checkpoint)
    step += 1
    emit("phase", phase="export", variant=None, step=step, steps=steps)
    with _Heartbeat("export"):
        fp32 = es.export_fp32(predictor, os.path.join(work, "fp32"))
        es.check_fp32(fp32)
    del predictor
    import gc
    gc.collect()

    ok = True
    for variant in variants:
        name = es.VARIANTS[variant][0]
        provenance = es.provenance_for(variant)
        step += 1
        emit("phase", phase="convert", variant=variant, step=step, steps=steps)
        with _Heartbeat("convert"):
            if variant == "full":
                built = es.write_full(fp32, os.path.join(work, name), provenance)
            elif variant == "fp16":
                built = es.convert_fp16(fp32, work, os.path.join(work, name), provenance)
            else:
                p = es.quantize_int4(fp32, os.path.join(work, f"{name}.onnx"), provenance)
                built = [("model", p, es.sha256_file(p), os.path.getsize(p))]
        gc.collect()
        step += 1
        emit("phase", phase="write", variant=variant, step=step, steps=steps)
        expect = es.pack_expectation(name)
        for role, path, sha, size in built:   # data before model
            dest = os.path.join(out, sha)
            os.replace(path, dest)
            emit("output", variant=variant, role=role, file=sha, sha256=sha, size=size)
        got = {role: (sha, size) for role, _, sha, size in built}
        if expect is None or expect[2] != got:
            ok = False
            emit("warning", message=f"{name}: built {got}, the builder's pins say "
                                    f"{expect[2] if expect else 'nothing'}")
    import shutil
    shutil.rmtree(os.path.join(work, "fp32"), ignore_errors=True)
    if not ok:
        raise BuildError("output-mismatch", "the built files are not the pinned ones")


# ---------------------------------------------------------------------------- main


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise BuildError("usage", message)


def main(argv):
    ap = _Parser(prog="dxr-model-builder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="print what this builder builds (one JSON object)")
    d.add_argument("--json", action="store_true", required=True)
    b = sub.add_parser("build", help="build model files")
    b.add_argument("--recipe", required=True)
    b.add_argument("--variant", action="append", required=True)
    b.add_argument("--source", action="append", default=[], metavar="ROLE=PATH|URL")
    b.add_argument("--out", required=True)
    b.add_argument("--work", required=True)
    b.add_argument("--cancel-file", required=True)
    b.add_argument("--accept-license", action="append", default=[])
    b.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "describe":
        sys.__stdout__.write(json.dumps(describe(), separators=(",", ":")) + "\n")
        sys.__stdout__.flush()
        return

    _watch_cancel(args.cancel_file)
    if args.recipe != "sharp-onnx":
        raise BuildError("usage", f"unknown recipe {args.recipe!r} (this builder has: sharp-onnx)")
    variants = []
    for v in args.variant:
        if v not in SHARP_VARIANTS:
            raise BuildError("usage", f"unknown variant {v!r} of sharp-onnx (has: {', '.join(SHARP_VARIANTS)})")
        if v not in variants:
            variants.append(v)
    emit("start", tool=TOOL_ID, version=TOOL_VERSION, recipe=args.recipe, variants=variants)
    if set(args.accept_license) != {SHARP_LICENSE}:
        raise BuildError("license", f"sharp-onnx needs --accept-license {SHARP_LICENSE} (and only that), "
                                    f"got {args.accept_license or 'none'}")
    sources = {}
    for s in args.source:
        role, sep, val = s.partition("=")
        if not sep or not val:
            raise BuildError("usage", f"--source takes ROLE=PATH|URL, got {s!r}")
        sources[role] = val
    if set(sources) != {"checkpoint"}:
        raise BuildError("usage", f"sharp-onnx takes exactly one source, checkpoint=..., got {sorted(sources)}")
    if args.threads < 0:
        raise BuildError("usage", "--threads must be positive")
    for dname in (args.work, args.out):
        if not os.path.isdir(dname):
            raise BuildError("usage", f"not a directory: {dname}")
    if sys.maxsize < 2 ** 32:
        raise BuildError("unsupported", "a 64-bit system is needed")
    _point_caches_at(os.path.abspath(args.work))
    build_sharp(args, variants, sources)


def entry():
    # Human text (export_sharp's log lines, library warnings) goes to stderr; stdout carries events.
    sys.stdout = sys.stderr
    for name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), _on_signal)
            except (OSError, ValueError):
                pass
    try:
        main(sys.argv[1:])
    except BaseException as e:  # noqa: BLE001 - every failure becomes one of the contract's codes
        if isinstance(e, SystemExit) and e.code in (0, None):
            finish("ok")
        code, msg = ("internal", f"exit {e.code}") if isinstance(e, SystemExit) else _classify(e)
        if code == "internal" and not isinstance(e, BuildError):
            import traceback
            traceback.print_exc()
        finish(code, msg)
    if len(sys.argv) > 1 and sys.argv[1] == "describe":
        sys.stderr.flush()
        os._exit(0)
    finish("ok")


if __name__ == "__main__":
    entry()
