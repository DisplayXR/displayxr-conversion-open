#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""Download a pack's blobs into a flat directory as <sha256>.<format>, verified.

    fetch_models.py PACK DIR [--models name,name] [--prefetch-only]

For development and CI only (the inference tests read the directory through a
file:// blob base). On a user's machine the runtime's host owns downloads and
the model store; the provider never downloads anything.

Each blob is written to a temporary name, checked for size and sha256, and
renamed only when both match. A blob already present with the right hash is
skipped.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("dir")
    ap.add_argument("--models", help="comma-separated model names (default: all)")
    ap.add_argument("--prefetch-only", action="store_true", help="only models marked prefetch")
    args = ap.parse_args()
    with open(args.pack, encoding="utf-8") as f:
        pack = json.load(f)
    want = set(args.models.split(",")) if args.models else None
    os.makedirs(args.dir, exist_ok=True)
    base = pack["blobBaseUrl"].rstrip("/")
    failures = 0
    done = set()
    for m in pack["models"]:
        if want is not None and m["name"] not in want:
            continue
        if args.prefetch_only and m.get("prefetch") is not True:
            continue
        for v in m["variants"]:
            for blob in v["files"]:
                sha, size, fmt = blob["sha256"], blob["size"], blob["format"]
                if sha in done:
                    continue
                done.add(sha)
                dst = os.path.join(args.dir, f"{sha}.{fmt}")
                if os.path.isfile(dst) and os.path.getsize(dst) == size and sha256_file(dst) == sha:
                    print(f"have {m['name']}: {sha}")
                    continue
                url = f"{base}/{sha}.{fmt}"
                tmp = dst + ".part"
                print(f"get  {m['name']}: {url}")
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                got_size = os.path.getsize(tmp)
                got = sha256_file(tmp)
                if got_size != size or got != sha:
                    print(f"FAIL {m['name']}: size {got_size} / sha256 {got} do not match the pack")
                    os.remove(tmp)
                    failures += 1
                    continue
                os.replace(tmp, dst)
                print(f"ok   {m['name']}: {sha} ({size} bytes)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
