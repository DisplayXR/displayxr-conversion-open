#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""Validate a DisplayXR model-pack manifest (schema 2, discovery spec §5).

    check_model_pack.py [--allow-research] [--verify-store DIR] [--verify-blobs DIR] PACK...

Checks the schema, that every blob is identified by sha256 + size, that the
licence flags are an AND over each model and its dependencies, that the pack's
own `commercial` flag equals the AND over its models, that `defaults` name
existing models of the right task, and that no blob marked `prefetch` is
non-commercial (a signed installer must never pre-fetch one).

Without --allow-research a pack must be commercial-clean as a whole (the
default pack is). --verify-store checks a model store (blobs/<sha256> plus its
.ok stamp, size and hash); --verify-blobs checks a flat directory of
<sha256>.<format> files (what scripts/fetch_models.py writes).

Exit code 0 = valid, 1 = a check failed, 2 = usage.
"""

import argparse
import hashlib
import json
import os
import re
import sys

TASKS = {
    "depth-mono",
    "depth-video",
    "disparity-stereo",
    "views-mono",
    "views-from-depth",
    "splats-mono",
    "splats-stereo",
}
DEPTH_KINDS = {"metric", "metric-inverse", "affine-inverse", "affine-disparity", "affine", "stereo-disparity-px"}
# Licences that never permit commercial use: a model or dependency carrying one
# cannot be flagged commercial.
NON_COMMERCIAL = re.compile(r"(^|[-.])NC($|[-.])|NonCommercial|research", re.IGNORECASE)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Report:
    def __init__(self):
        self.errors = 0

    def fail(self, where, msg):
        self.errors += 1
        print(f"ERROR {where}: {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pack(path, args, rep):
    try:
        with open(path, encoding="utf-8") as f:
            pack = json.load(f)
    except (OSError, ValueError) as e:
        rep.fail(path, f"cannot read: {e}")
        return []
    where = os.path.basename(path)
    if pack.get("schema") != 2:
        rep.fail(where, "schema must be 2")
    pid = pack.get("pack")
    if not isinstance(pid, str) or not pid or "/" in pid:
        rep.fail(where, "pack id missing or contains '/'")
    base = pack.get("blobBaseUrl", "")
    if not (isinstance(base, str) and (base.startswith("https://") or base.startswith("file://"))):
        rep.fail(where, "blobBaseUrl must be https:// (or file:// for tests)")
    if "compat" in pack or "schema1" in pack:
        rep.fail(where, "schema 2 carries no schema-1 compatibility block")

    models = pack.get("models")
    if not isinstance(models, list) or not models:
        rep.fail(where, "models must be a non-empty list")
        return []
    names = {}
    blobs = []
    all_commercial = True
    prefetch_bytes = 0
    for m in models:
        name = m.get("name")
        mw = f"{where}:{name}"
        if not isinstance(name, str) or not name:
            rep.fail(where, "a model has no name")
            continue
        if name in names:
            rep.fail(mw, "duplicate model name")
        names[name] = m
        if not isinstance(m.get("version"), str) or not m["version"]:
            rep.fail(mw, "version missing")
        task = m.get("task", "")
        if task not in TASKS and not (isinstance(task, str) and task.startswith("aux:")):
            rep.fail(mw, f"unknown task '{task}'")
        io = m.get("io", {})
        if not isinstance(io.get("input"), dict) or not isinstance(io.get("output"), dict):
            rep.fail(mw, "io.input and io.output are required")
        else:
            res = io["input"].get("resolution", "")
            if not (re.match(r"^fixed-\d+x\d+$", res) or re.match(r"^dynamic-multiple-of-\d+$", res)):
                rep.fail(mw, f"io.input.resolution '{res}' is neither fixed-WxH nor dynamic-multiple-of-N")
            if task.startswith("depth") and io["output"].get("depth_kind") not in DEPTH_KINDS:
                rep.fail(mw, f"io.output.depth_kind '{io['output'].get('depth_kind')}' unknown")
        lic = m.get("license", {})
        spdx = lic.get("spdx")
        if not isinstance(spdx, str) or not spdx:
            rep.fail(mw, "license.spdx missing")
            spdx = ""
        commercial = lic.get("commercial")
        if not isinstance(commercial, bool):
            rep.fail(mw, "license.commercial must be true or false")
            commercial = False
        if commercial and NON_COMMERCIAL.search(spdx):
            rep.fail(mw, f"license.commercial is true but the licence is {spdx}")
        effective = commercial
        for dep in lic.get("dependencies", []):
            dc = dep.get("commercial")
            if not isinstance(dc, bool):
                rep.fail(mw, f"dependency '{dep.get('name')}' has no commercial flag")
                dc = False
            if dc and NON_COMMERCIAL.search(dep.get("spdx", "")):
                rep.fail(mw, f"dependency '{dep.get('name')}' is flagged commercial under {dep.get('spdx')}")
            effective = effective and dc
        all_commercial = all_commercial and effective
        prov = m.get("provenance", {})
        if not isinstance(prov.get("source_url"), str) or not prov["source_url"].startswith("https://"):
            rep.fail(mw, "provenance.source_url (https) is required")
        prefetch = m.get("prefetch") is True
        if prefetch and not effective:
            rep.fail(mw, "prefetch is set on a non-commercial model: a signed installer must never pre-fetch it")
        variants = m.get("variants")
        if not isinstance(variants, list) or not variants:
            rep.fail(mw, "no variants")
            continue
        vids = set()
        for v in variants:
            vid = v.get("id")
            if not isinstance(vid, str) or not vid or vid in vids:
                rep.fail(mw, f"variant id '{vid}' missing or duplicate")
            vids.add(vid)
            if not isinstance(v.get("ep"), list) or not v["ep"]:
                rep.fail(mw, f"variant {vid}: ep list missing")
            files = v.get("files", [])
            roles = [f.get("role") for f in files]
            if roles.count("model") != 1:
                rep.fail(mw, f"variant {vid}: exactly one file with role 'model' is required")
            for f in files:
                sha, size = f.get("sha256", ""), f.get("size")
                if not SHA256.match(sha or ""):
                    rep.fail(mw, f"variant {vid}: sha256 must be 64 lower-case hex characters")
                if not isinstance(size, int) or size <= 0:
                    rep.fail(mw, f"variant {vid}: size must be a positive integer")
                    size = 0
                fmt = f.get("format", "")
                if not re.match(r"^[a-z0-9]+$", fmt or ""):
                    rep.fail(mw, f"variant {vid}: format '{fmt}' invalid")
                blobs.append((sha, size, fmt, name, prefetch))
                if prefetch:
                    prefetch_bytes += size
    pc = pack.get("commercial")
    if not isinstance(pc, bool):
        rep.fail(where, "pack 'commercial' flag missing")
    elif pc != all_commercial:
        rep.fail(where, f"pack 'commercial' is {pc} but the AND over its models is {all_commercial}")
    if not args.allow_research and not all_commercial:
        rep.fail(where, "pack is not commercial-clean (pass --allow-research for a research pack)")

    defaults = pack.get("defaults")
    if not isinstance(defaults, dict) or not defaults:
        rep.fail(where, "defaults missing")
    else:
        for task, levels in defaults.items():
            for q in ("low", "medium", "high"):
                ref = levels.get(q) if isinstance(levels, dict) else None
                if ref not in names:
                    rep.fail(where, f"defaults[{task}][{q}] = '{ref}' names no model")
                elif names[ref].get("task") != task:
                    rep.fail(where, f"defaults[{task}][{q}] = '{ref}' is a {names[ref].get('task')} model")
    seen = {}
    for sha, size, fmt, name, _ in blobs:
        if sha in seen and seen[sha] != size:
            rep.fail(where, f"blob {sha} listed with two sizes")
        seen[sha] = size
    print(
        f"{where}: {len(models)} models, {len(seen)} blobs, commercial={all_commercial}, "
        f"prefetch {prefetch_bytes / 1e6:.1f} MB"
    )
    return blobs


def verify_blobs(blobs, directory, store, rep):
    for sha, size, fmt, name, _ in {b[0]: b for b in blobs}.values():
        if store:
            path = os.path.join(directory, "blobs", sha)
            if not os.path.isfile(path + ".ok"):
                rep.fail(name, f"{sha}: no .ok stamp in the store")
                continue
        else:
            path = os.path.join(directory, f"{sha}.{fmt}")
        if not os.path.isfile(path):
            rep.fail(name, f"{sha}: missing ({path})")
            continue
        if os.path.getsize(path) != size:
            rep.fail(name, f"{sha}: size {os.path.getsize(path)} != {size}")
            continue
        got = sha256_file(path)
        if got != sha:
            rep.fail(name, f"{sha}: content hashes to {got} (tampered or corrupt)")
        else:
            print(f"ok {name}: {sha} ({size} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packs", nargs="+")
    ap.add_argument("--allow-research", action="store_true")
    ap.add_argument("--verify-store", metavar="DIR")
    ap.add_argument("--verify-blobs", metavar="DIR")
    args = ap.parse_args()
    rep = Report()
    for p in args.packs:
        blobs = check_pack(p, args, rep)
        if args.verify_store:
            verify_blobs(blobs, args.verify_store, True, rep)
        if args.verify_blobs:
            verify_blobs(blobs, args.verify_blobs, False, rep)
    print("OK" if rep.errors == 0 else f"FAILED: {rep.errors} error(s)")
    return 0 if rep.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
