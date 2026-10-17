#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""Rewrite the streaming video-depth models so the Core ML execution provider takes them whole.

    rewrite_video_depth_coreml.py PACK MODELS_DIR OUT_DIR [--models name,name] [--check]

Reads each selected model's blob from MODELS_DIR as <sha256>.onnx (scripts/fetch_models.py fills
it), refuses it unless its size and sha256 match PACK, and writes the rewritten graph to OUT_DIR
as <new sha256>.onnx, printing the new sha256 and size for the pack entry.

What it changes, and only this: every Gather whose indices are a constant SCALAR becomes
Slice(k, k + 1, axis) + Squeeze(axis). Same maths, same weights, same tensor names and shapes.
The streaming export splits q / k / v in each attention block with such a Gather, and ONNX
Runtime's Core ML execution provider rejects scalar indices ("Gather does not support scalar
'indices'"), so the graph fell apart into one Core ML partition per block with a CPU hop
between them. After the rewrite Core ML takes the graph in one partition.

--check runs the original and the rewritten model on the ONNX Runtime CPU execution provider
(Python package `onnxruntime`) on random inputs, first and continued frame, and fails unless
every output matches within 1e-5 relative.

Needs the `onnx` Python package (and `numpy`). The output bytes depend on the onnx version
that serialises them; the version is printed with the hash, so a blob is reproducible with the
same version. For development: publishing a blob is a separate, manual step.
"""

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rewrite(model, onnx, helper, numpy_helper, np):
    """Gather(data, scalar constant k, axis) -> Slice + Squeeze. Returns the count."""
    g = model.graph
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)
    if opset < 13:
        raise SystemExit(f"opset {opset}: Squeeze with an axes input needs opset 13 or later")
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    nodes, inits, count = [], [], 0
    for n in g.node:
        idx = consts.get(n.input[1]) if n.op_type == "Gather" and len(n.input) > 1 else None
        if idx is None or idx.ndim != 0:
            nodes.append(n)
            continue
        k = int(idx)
        axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
        tag = f"{n.name or 'gather'}_coreml{count}"
        end = k + 1 if k != -1 else np.iinfo(np.int64).max
        inits += [
            numpy_helper.from_array(np.array([k], np.int64), tag + "_start"),
            numpy_helper.from_array(np.array([end], np.int64), tag + "_end"),
            numpy_helper.from_array(np.array([axis], np.int64), tag + "_axis"),
        ]
        nodes.append(helper.make_node("Slice", [n.input[0], tag + "_start", tag + "_end", tag + "_axis"],
                                      [tag + "_slice"], name=tag + "_Slice"))
        nodes.append(helper.make_node("Squeeze", [tag + "_slice", tag + "_axis"], [n.output[0]],
                                      name=tag + "_Squeeze"))
        count += 1
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(inits)
    return count


def check(src, dst, np):
    import onnxruntime as ort

    a = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    feeds = {}
    for i in a.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        feeds[i.name] = rng.random(shape, dtype=np.float32)
    names = [o.name for o in a.get_outputs()]
    worst = 0.0
    for frame in range(2):
        if "is_first" in feeds:
            feeds["is_first"] = np.array([1.0 if frame == 0 else 0.0], np.float32)
        oa = a.run(names, feeds)
        ob = b.run(names, feeds)
        for n, x, y in zip(names, oa, ob):
            rel = float(np.max(np.abs(x - y)) / (np.max(np.abs(x)) + 1e-12))
            worst = max(worst, rel)
            if rel > 1e-5:
                raise SystemExit(f"check FAILED: output {n} frame {frame} differs by {rel:.2e} (relative)")
        if "cache_in" in feeds and "cache_out" in names:
            feeds["cache_in"] = oa[names.index("cache_out")]
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("models_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--models", default="vda-small-stream-518x294,vda-small-stream-364x210")
    ap.add_argument("--check", action="store_true", help="compare outputs on the ONNX Runtime CPU EP")
    args = ap.parse_args()

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    with open(args.pack, encoding="utf-8") as f:
        pack = json.load(f)
    want = [m for m in args.models.split(",") if m]
    by_name = {m["name"]: m for m in pack["models"]}
    os.makedirs(args.out_dir, exist_ok=True)
    for name in want:
        m = by_name.get(name)
        if m is None:
            raise SystemExit(f"{name}: not in {args.pack}")
        blob = next(f for f in m["variants"][0]["files"] if f["role"] == "model")
        src = os.path.join(args.models_dir, f"{blob['sha256']}.onnx")
        if not os.path.isfile(src) or os.path.getsize(src) != blob["size"] or sha256_file(src) != blob["sha256"]:
            raise SystemExit(f"{name}: {src} missing or not the pack's blob (size / sha256)")
        model = onnx.load(src)
        count = rewrite(model, onnx, helper, numpy_helper, np)
        if count == 0:
            raise SystemExit(f"{name}: no Gather with a scalar constant index; nothing to do")
        onnx.checker.check_model(model)
        tmp = os.path.join(args.out_dir, f".{name}.part")
        onnx.save(model, tmp)
        digest, size = sha256_file(tmp), os.path.getsize(tmp)
        dst = os.path.join(args.out_dir, f"{digest}.onnx")
        os.replace(tmp, dst)
        line = f"{name}: rewrote {count} Gather nodes -> {dst}\n  sha256 {digest}  size {size}  (onnx {onnx.__version__})"
        if args.check:
            line += f"\n  check: outputs match the original (worst relative difference {check(src, dst, np):.1e})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
