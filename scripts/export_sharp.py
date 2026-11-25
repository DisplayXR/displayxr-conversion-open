#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""Build the local SHARP (image -> Gaussian splats) ONNX models from Apple's official checkpoint.

    export_sharp.py --accept-research-license [--variant int4|full|fp16|both] [--checkpoint FILE]
                    [--ml-sharp DIR] [--work DIR] [--store DIR | --no-store] [--keep-fp32]
                    [--check IMAGE]

SHARP's weights are licensed for research only (Apple Machine Learning Research Model
licence, `apple-amlr`), and the licence follows every derivative. DisplayXR therefore never
bundles, mirrors or publishes them, nor any file made from them: you run this script on your
own machine, from the official checkpoint you download yourself (ADR-057 section 5).

What it does, in order:

1. Takes the official checkpoint (`--checkpoint FILE`), or downloads it from the official
   Hugging Face repository at a pinned revision. Either way it refuses any file whose size and
   SHA-256 are not exactly the pinned ones. No third-party re-upload is ever accepted.
2. Takes Apple's reference code `apple/ml-sharp` (`--ml-sharp DIR`), or clones it, and refuses
   any checkout that is not the pinned commit.
3. Exports the fp32 predictor to ONNX (opset 18, standard ai.onnx operators only, a fixed
   1x3x1536x1536 image input) into the work directory (2.6 GB, external weights).
4. Builds the variant(s) asked for from that one export (one download serves both):
   - `full` (`sharp-full-1536`): the fp32 export itself, re-saved as two files, the graph and
     ONE external-data file. The graph names the data file by the data's own SHA-256, so the
     two sit side by side in the store's blobs/ directory and the inference runtime finds the
     weights next to the graph (ONNX's protobuf cannot hold 2.6 GB in one file). About
     2.6 GB; numerically the PyTorch model.
   - `int4` (`sharp-int4-1536`): ONNX Runtime's MatMulNBitsQuantizer (4-bit, block 32,
     symmetric): only the weights of MatMul nodes with a constant weight become 4-bit
     (com.microsoft MatMulNBits); Conv weights, every other weight and every activation stay
     fp32. One self-contained .onnx file (about 0.58 GB).
   - `fp16` (`sharp-fp16-1536`, DirectML only): the fp32 export with its static shape
     arithmetic folded first (in Python, exactly-defined operators only: see fold_static), then
     converted to float16 (onnxconverter-common) with fp32 inputs and outputs (checked: every
     graph input and output keeps the fp32 model's type). The whole graph runs in fp16 except
     onnxconverter-common's default block list (Resize, Range, Min / Max, CumSum, ...): the
     full-resolution convolutions of the decoders and the heads are where the time and the
     memory are, and keeping any of them fp32 costs about 2.6x on DirectML. Graph + one
     external-data file named by its SHA-256, like `full` (about 1.3 GB). Half the GPU memory
     of `full`. No ONNX Runtime step: its optimised-graph output is not the same bytes on
     every OS.
   The fp32 intermediate is deleted unless --keep-fp32.
5. Writes the result into the DisplayXR model store (blobs/<sha256> plus its .ok stamp, the
   store's own write protocol; for `full` the data file first, the graph last), so the
   `dxr-open-sharp` pack resolves it, and prints each SHA-256 and size. The pack lists exactly
   these bytes: both exports are byte-deterministic with the pinned package versions in
   scripts/requirements-export-sharp.txt, and the script checks the hashes it produced against
   the pack and says so if they differ.

--variant picks what to build. The default is the variant this machine's provider picks by
itself (its `MODEL=auto` rule: `full` when the splat capability runs on a GPU and the machine
has at least 16 GB of memory, else `int4`); the script says which and why. `both` builds both
from one export.

--check IMAGE runs the PyTorch fp32 reference and each built model (ONNX Runtime, CPU) on one
image and prints the per-tensor relative error and the median relative depth error. It takes
a few minutes and about 16 GB of memory (ONNX Runtime's CPU execution provider peaks near 14 GB
on int4).

Needs: Python 3.11+, git, about 8 GB free memory, about 10 GB free disk in --work, and the
packages pinned in scripts/requirements-export-sharp.txt (CPU-only PyTorch is enough).

Exit code 0 = done, 1 = a check failed (bad hash, wrong commit, parity), 2 = usage.
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request

SCRIPT_VERSION = "1.4.0"

# The official checkpoint (Apple, research-only). Same bytes as Apple's CDN copy.
CHECKPOINT_NAME = "sharp_2572gikvuh.pt"
CHECKPOINT_REVISION = "50753d96c1f7ad0bae4184ddfb6aa98bca975f29"
CHECKPOINT_URL = f"https://huggingface.co/apple/Sharp/resolve/{CHECKPOINT_REVISION}/{CHECKPOINT_NAME}"
CHECKPOINT_SHA256 = "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"
CHECKPOINT_SIZE = 2809738232
LICENSE_URL = f"https://huggingface.co/apple/Sharp/blob/{CHECKPOINT_REVISION}/LICENSE"

# Apple's reference model code.
ML_SHARP_REPO = "https://github.com/apple/ml-sharp.git"
ML_SHARP_COMMIT = "aed6527499ef91cba3b54c18d49a870f25947190"

RESOLUTION = 1536
OPSET = 18
INPUTS = ["image", "disparity_factor"]
OUTPUTS = ["mean_vectors", "singular_values", "quaternions", "colors", "opacities"]

PACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packs", "dxr-open-sharp.models.json")
# variant -> (pack model, the script version whose recipe made these bytes). That version is
# written into the model's metadata, so it is part of the bytes: it changes only when the
# recipe does, never with an unrelated script change.
VARIANTS = {
    "int4": ("sharp-int4-1536", "1.0.0"),
    "full": ("sharp-full-1536", "1.1.0"),
    "fp16": ("sharp-fp16-1536", "1.4.0"),
}
# fp16: static shape arithmetic is folded in Python (fold_static) before the conversion, with
# operators whose result is exactly defined (integer arithmetic, IEEE + - * / sqrt, data
# movement): no ONNX Runtime graph optimiser, whose saved graph is not the same bytes on every
# OS, and no transcendental functions, whose last bit depends on the C library.
FOLD_OPS = frozenset((
    "Shape", "Constant", "Identity", "Cast", "Slice", "Gather", "Concat", "Unsqueeze", "Squeeze",
    "Reshape", "Expand", "ConstantOfShape", "Range", "Transpose", "Tile", "Where", "Equal", "Less",
    "Greater", "LessOrEqual", "GreaterOrEqual", "Not", "And", "Or", "Add", "Sub", "Mul", "Div",
    "Mod", "Neg", "Abs", "Floor", "Ceil", "Sqrt", "Reciprocal", "Min", "Max",
))
FOLD_MAX_ELEMENTS = 4096  # shape arithmetic, never a weight or an activation
STORE_OWNER_MARKER = "displayxr-runtime.owner"
GPU_RAM_FLOOR_GB = 16  # the provider's MODEL=auto rule: full needs a GPU and this much memory


def log(msg):
    print(msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------- inputs


def verify_checkpoint(path):
    size = os.path.getsize(path)
    if size != CHECKPOINT_SIZE:
        fail(f"{path}: {size} bytes, the official checkpoint is {CHECKPOINT_SIZE}; refusing it")
    log(f"hashing {path} ...")
    got = sha256_file(path)
    if got != CHECKPOINT_SHA256:
        fail(f"{path}: sha256 {got} is not the official checkpoint's {CHECKPOINT_SHA256}; refusing it")
    log(f"checkpoint ok: {CHECKPOINT_SHA256} ({size} bytes)")


def download_checkpoint(dest_dir):
    """Stream the pinned file to <dest>.part, hashing as it arrives; rename only when it matches."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, CHECKPOINT_NAME)
    if os.path.isfile(dest):
        verify_checkpoint(dest)
        return dest
    log(f"downloading {CHECKPOINT_URL}\n  ({CHECKPOINT_SIZE / 1e9:.2f} GB, licence: {LICENSE_URL})")
    req = urllib.request.Request(CHECKPOINT_URL, headers={"User-Agent": "displayxr-export-sharp"})
    fd, part = tempfile.mkstemp(prefix=CHECKPOINT_NAME + ".", suffix=".part", dir=dest_dir)
    h = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as r, os.fdopen(fd, "wb") as out:
            last = -1
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                pct = done * 100 // CHECKPOINT_SIZE
                if pct != last and pct % 5 == 0:
                    log(f"  {pct:3d}%  {done / 1e9:.2f} GB")
                    last = pct
        if done != CHECKPOINT_SIZE or h.hexdigest() != CHECKPOINT_SHA256:
            fail(f"downloaded {done} bytes hashing to {h.hexdigest()}: not the official checkpoint")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)
    log(f"checkpoint ok: {CHECKPOINT_SHA256} ({done} bytes)")
    return dest


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def prepare_ml_sharp(path, work):
    if path is None:
        path = os.path.join(work, "ml-sharp")
        if not os.path.isdir(os.path.join(path, ".git")):
            log(f"cloning {ML_SHARP_REPO} ...")
            subprocess.run(["git", "clone", "--quiet", ML_SHARP_REPO, path], check=True)
        if git("rev-parse", "HEAD", cwd=path) != ML_SHARP_COMMIT:
            git("-c", "advice.detachedHead=false", "checkout", "--quiet", ML_SHARP_COMMIT, cwd=path)
    head = git("rev-parse", "HEAD", cwd=path)
    if head != ML_SHARP_COMMIT:
        fail(f"{path} is at {head}; the export is pinned to apple/ml-sharp {ML_SHARP_COMMIT}")
    if git("status", "--porcelain", "--untracked-files=no", cwd=path):
        fail(f"{path} has local modifications; use a clean checkout of {ML_SHARP_COMMIT}")
    src = os.path.join(path, "src")
    if not os.path.isfile(os.path.join(src, "sharp", "models", "__init__.py")):
        fail(f"{path} does not look like an apple/ml-sharp checkout")
    log(f"ml-sharp ok: {ML_SHARP_COMMIT}")
    return src


# ---------------------------------------------------------------------------- model


def load_predictor(ml_sharp_src, checkpoint):
    import torch

    sys.path.insert(0, ml_sharp_src)
    from sharp.models import PredictorParams, create_predictor

    torch.set_grad_enabled(False)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model = create_predictor(PredictorParams())
    model.load_state_dict(state)
    model.eval()
    del state
    gc.collect()

    class Predictor(torch.nn.Module):
        """The upstream predictor with its NamedTuple output flattened to five tensors."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image, disparity_factor):
            g = self.m(image, disparity_factor)
            return g.mean_vectors, g.singular_values, g.quaternions, g.colors, g.opacities

    return Predictor(model).eval()


def example_inputs():
    """A fixed, content-free example: the traced graph does not depend on pixel values."""
    import torch

    ramp = torch.linspace(0.0, 1.0, RESOLUTION, dtype=torch.float32)
    image = torch.stack([ramp[None, :].expand(RESOLUTION, -1), ramp[:, None].expand(-1, RESOLUTION),
                         torch.full((RESOLUTION, RESOLUTION), 0.5)])[None].contiguous()
    return image, torch.tensor([1.0], dtype=torch.float32)


def export_fp32(predictor, out_dir):
    import torch

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sharp_fp32.onnx")
    for stale in (path, path + ".data"):
        if os.path.exists(stale):
            os.remove(stale)
    log(f"exporting fp32 ONNX (opset {OPSET}) ...")
    torch.onnx.export(
        predictor,
        example_inputs(),
        path,
        input_names=INPUTS,
        output_names=OUTPUTS,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    return path


def check_fp32(path):
    import onnx

    m = onnx.load(path, load_external_data=False)
    domains = {n.domain for n in m.graph.node} - {"", "ai.onnx"}
    if domains:
        fail(f"fp32 export uses non-standard operator domains {sorted(domains)}")
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
    if shapes.get("image") != [1, 3, RESOLUTION, RESOLUTION] or shapes.get("disparity_factor") != [1]:
        fail(f"unexpected fp32 input shapes {shapes}")
    if [o.name for o in m.graph.output] != OUTPUTS:
        fail(f"unexpected fp32 outputs {[o.name for o in m.graph.output]}")
    log(f"fp32 export ok: {len(m.graph.node)} nodes, standard ai.onnx operators only")


def quantize_int4(fp32_path, out_path, provenance):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    # The quantizer logs one INFO line per node it skips (thousands).
    logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    log("quantizing MatMul weights to 4 bits (block 32, symmetric) ...")
    model = onnx.load(fp32_path)
    q = MatMulNBitsQuantizer(model, block_size=32, is_symmetric=True, bits=4)
    q.process()
    # Newer onnxruntime wraps the ModelProto (onnx_ir); older returns an ONNXModel.
    out = q.model.model if hasattr(q.model, "model") else q.model
    del q, model
    gc.collect()
    del out.metadata_props[:]
    for k, v in provenance.items():
        out.metadata_props.add(key=k, value=v)
    if os.path.exists(out_path):
        os.remove(out_path)
    onnx.save(out, out_path)  # one self-contained file: well under protobuf's 2 GB limit
    n4 = sum(1 for n in out.graph.node if n.op_type == "MatMulNBits")
    nconv = sum(1 for n in out.graph.node if n.op_type == "Conv")
    nmm = sum(1 for n in out.graph.node if n.op_type == "MatMul")
    log(f"int4 ok: {n4} MatMulNBits, {nmm} MatMul left fp32 (activation x activation), {nconv} Conv fp32")
    return out_path


def write_split(model, out_dir, label):
    """@p model as <sha(graph)> + <sha(data)>: the graph names its one data file by the data's
    SHA-256, so the two files find each other inside the store's blobs/ directory. Consumes
    @p model (its tensors move into the data file)."""
    import onnx
    from onnx import external_data_helper as edh

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    graph, data = os.path.join(out_dir, "graph.onnx"), os.path.join(out_dir, "weights.data")
    onnx.save_model(model, graph, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(data), size_threshold=1024, convert_attribute=False)
    del model
    gc.collect()
    data_sha, data_size = sha256_file(data), os.path.getsize(data)
    os.replace(data, os.path.join(out_dir, data_sha))
    model = onnx.load(graph, load_external_data=False)
    n = 0
    for t in edh._get_all_tensors(model):
        if edh.uses_external_data(t):
            for e in t.external_data:
                if e.key == "location":
                    e.value = data_sha
                    n += 1
    onnx.save(model, graph)
    graph_sha, graph_size = sha256_file(graph), os.path.getsize(graph)
    final = os.path.join(out_dir, graph_sha)
    os.replace(graph, final)
    log(f"{label} ok: {n} external tensors in one data file ({data_size} bytes), graph {graph_size} bytes")
    return [("data", os.path.join(out_dir, data_sha), data_sha, data_size), ("model", final, graph_sha, graph_size)]


def provenance_for(variant):
    """The metadata written into @p variant's model (part of its bytes)."""
    name, recipe = VARIANTS[variant]
    return {
        "dxr.model": name,
        "dxr.converter": f"displayxr-conversion-open scripts/export_sharp.py {recipe}",
        "dxr.parent.url": CHECKPOINT_URL,
        "dxr.parent.sha256": CHECKPOINT_SHA256,
        "dxr.code": f"apple/ml-sharp@{ML_SHARP_COMMIT}",
        "dxr.license": "apple-amlr (research only) " + LICENSE_URL,
    }


def set_provenance(model, provenance):
    del model.metadata_props[:]
    for k, v in provenance.items():
        model.metadata_props.add(key=k, value=v)


def write_full(fp32_path, out_dir, provenance):
    """The fp32 export as one graph + one external-data file (write_split)."""
    import onnx

    log("writing the full (fp32) model: one graph + one external-data file ...")
    model = onnx.load(fp32_path)
    set_provenance(model, provenance)
    return write_split(model, out_dir, "full")


def fold_static(model, base_dir):
    """Fold @p model's static shape arithmetic in place, in Python: every node of FOLD_OPS whose
    inputs are all known (small initializers, small Constants, the Shape of a tensor whose shape
    shape inference fixed) and whose outputs have at most FOLD_MAX_ELEMENTS elements is
    evaluated with ONNX's reference implementation and replaced by initializers named after its
    outputs. The traced attention computes its scale as Shape -> Slice -> Cast -> Sqrt per block;
    a float16 copy of that arithmetic has no CPU kernel to fold it at load time. Every step is
    a function of the graph alone (node order in, node order out): no hash order, no ONNX
    Runtime, no transcendental, so the bytes are the same on every OS. Returns the folded
    node count."""
    import itertools

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    g = model.graph
    shapes = {}
    for v in itertools.chain(g.input, g.value_info, g.output):
        dims = v.type.tensor_type.shape.dim
        if v.type.tensor_type.HasField("shape") and all(d.HasField("dim_value") for d in dims):
            shapes[v.name] = [d.dim_value for d in dims]
    for t in g.initializer:
        shapes[t.name] = list(t.dims)

    def numel(name):
        s = shapes.get(name)
        return None if s is None else int(np.prod(s, dtype=np.int64))

    known = {}
    for t in g.initializer:
        if numel(t.name) <= FOLD_MAX_ELEMENTS:
            known[t.name] = numpy_helper.to_array(t, base_dir)
    opset = [o for o in model.opset_import if o.domain in ("", "ai.onnx")]
    folded = []
    for node in g.node:
        if node.op_type not in FOLD_OPS or node.domain not in ("", "ai.onnx"):
            continue
        ins = [x for x in node.input if x]
        if node.op_type == "Shape":
            s = shapes.get(node.input[0])
            if s is None:
                continue
            a = {at.name: at.i for at in node.attribute}
            start, end = a.get("start", 0), a.get("end", len(s))
            outs = [np.array(s[slice(start, end)], dtype=np.int64)]
        else:
            if any(x not in known for x in ins):
                continue
            if node.op_type == "Constant" and any(
                    at.type == onnx.AttributeProto.TENSOR and (int(np.prod(at.t.dims, dtype=np.int64)) > FOLD_MAX_ELEMENTS
                                                              or at.t.data_location == onnx.TensorProto.EXTERNAL)
                    for at in node.attribute):
                continue  # a weight, not shape arithmetic
            if node.op_type != "Constant" and any(numel(o) is None or numel(o) > FOLD_MAX_ELEMENTS
                                                  for o in node.output):
                continue
            one = helper.make_graph(
                [node], "fold",
                [helper.make_tensor_value_info(x, helper.np_dtype_to_tensor_dtype(known[x].dtype),
                                               known[x].shape) for x in dict.fromkeys(ins)],
                [helper.make_tensor_value_info(o, onnx.TensorProto.UNDEFINED, None) for o in node.output])
            outs = ReferenceEvaluator(helper.make_model(one, opset_imports=opset)).run(
                None, {x: known[x] for x in dict.fromkeys(ins)})
            outs = [np.asarray(o) for o in outs]
            if any(o.size > FOLD_MAX_ELEMENTS for o in outs):
                continue
        for o, v in zip(node.output, outs):
            known[o] = v
        folded.append(node)
    if not folded:
        return 0
    gone = {id(n) for n in folded}
    kept = [n for n in g.node if id(n) not in gone]
    used = set(itertools.chain.from_iterable(n.input for n in kept)) | {o.name for o in g.output}
    produced = [o for n in folded for o in n.output if o in used]
    old_inits = [t for t in g.initializer if t.name in used]
    del g.node[:]
    g.node.extend(kept)
    del g.initializer[:]
    g.initializer.extend(old_inits)
    g.initializer.extend(numpy_helper.from_array(known[o], o) for o in produced)
    stale = set(produced)
    vis = [v for v in g.value_info if v.name not in stale and v.name in used]
    del g.value_info[:]
    g.value_info.extend(vis)
    return len(folded)


def io_types(model):
    return {v.name: v.type.tensor_type.elem_type for v in list(model.graph.input) + list(model.graph.output)}


def convert_fp16(fp32_path, work, out_dir, provenance):
    """fp32 export -> fold_static -> float16 (fp32 I/O; only onnxconverter-common's default block
    list stays fp32) -> write_split."""
    import onnx
    from onnx import external_data_helper as edh
    from onnx import shape_inference
    from onnxconverter_common import float16

    log("folding the static shape arithmetic, converting to float16 ...")
    fp32_dir = os.path.dirname(os.path.abspath(fp32_path))
    # Shape inference on the graph alone (protobuf cannot serialise > 2 GB), then the weights.
    model = shape_inference.infer_shapes(onnx.load(fp32_path, load_external_data=False))
    want_io = io_types(model)
    nfold = fold_static(model, fp32_dir)
    edh.load_external_data_for_model(model, fp32_dir)
    block_ops = set(float16.DEFAULT_OP_BLOCK_LIST)
    blocked = [n.name for n in model.graph.node if n.op_type in block_ops]
    # An initializer shared by an fp32 node and a converted one would stay fp32 for both
    # (onnxconverter-common keeps any initializer a blocked node reads): give the fp32 nodes a copy.
    blocked_set = set(blocked)
    inits = {i.name: i for i in model.graph.initializer}
    users = {}
    for n in model.graph.node:
        for k, x in enumerate(n.input):
            if x in inits:
                users.setdefault(x, []).append((n, k))
    split = 0
    for name in sorted(users):
        us = users[name]
        if len({n.name in blocked_set for n, _ in us}) == 2 and inits[name].data_type == onnx.TensorProto.FLOAT:
            copy = onnx.TensorProto()
            copy.CopyFrom(inits[name])
            copy.name = name + "__fp32"
            model.graph.initializer.append(copy)
            for n, k in us:
                if n.name in blocked_set:
                    n.input[k] = copy.name
            split += 1
    del inits, users
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # one "value truncated" line per clipped tiny weight
        model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True,
                                                 op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST))
    g = model.graph
    # A graph output produced by a blocked (fp32) node: onnxconverter-common's cast-after-blocked
    # pass walks graph.output too, so it retypes that output to float16 and appends a Cast to it,
    # although keep_io_types promised fp32. Undo: the producer writes the output again, in fp32.
    producer = {o: n for n in g.node for o in n.output}
    consumers = {x for n in g.node for x in n.input}
    for out in g.output:
        if out.type.tensor_type.elem_type == want_io[out.name]:
            continue
        cast = producer.get(out.name)
        src = cast.input[0] if cast is not None and cast.op_type == "Cast" else None
        up = producer.get(src)
        if up is None or out.name in consumers:
            fail(f"fp16: output {out.name} became type {out.type.tensor_type.elem_type} and cannot be restored")
        up.output[list(up.output).index(src)] = out.name
        g.node.remove(cast)
        for v in [v for v in g.value_info if v.name == src]:
            g.value_info.remove(v)
        out.type.tensor_type.elem_type = want_io[out.name]
    # onnxconverter-common retypes a Cast's output to float16 but can leave its 'to' at float
    # (value_info only: a graph output keeps the type keep_io_types gave it).
    outputs = {o.name for o in g.output}
    vi = {v.name: v for v in g.value_info}
    for n in g.node:
        if n.op_type == "Cast" and n.output[0] not in outputs:
            v = vi.get(n.output[0])
            for a in n.attribute:
                if (a.name == "to" and a.i == onnx.TensorProto.FLOAT and v is not None
                        and v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16):
                    a.i = onnx.TensorProto.FLOAT16
    got_io = io_types(model)
    if got_io != want_io:
        bad = {k: (want_io.get(k), v) for k, v in got_io.items() if want_io.get(k) != v}
        fail(f"fp16: graph inputs/outputs changed type (name: (fp32 model, fp16 model)): {bad}")
    n16 = sum(1 for i in g.initializer if i.data_type == onnx.TensorProto.FLOAT16)
    n32 = sum(1 for i in g.initializer if i.data_type == onnx.TensorProto.FLOAT)
    log(f"fp16: {nfold} static nodes folded; {len(blocked)} nodes kept fp32 (default block list: "
        f"{', '.join(sorted({n.op_type for n in g.node if n.name in set(blocked)}))}), "
        f"{split} shared initializers split, {n16} fp16 / {n32} fp32 "
        f"initializers, {sum(1 for n in g.node if n.op_type == 'Cast')} Cast nodes; inputs and outputs fp32")
    set_provenance(model, provenance)
    return write_split(model, out_dir, "fp16")


# ---------------------------------------------------------------------------- machine


def total_ram_gb():
    """Physical memory in GB (portable; None when it cannot be read)."""
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = MemoryStatus()
            ms.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return None
            return ms.ullTotalPhys / 2**30
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
            return int(out.stdout.strip()) / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None


def windows_discrete_gpu():
    """Windows: the name of a discrete NVIDIA / AMD video controller, or None. A heuristic for the
    default variant only (the provider itself ranks DXGI adapters by dedicated memory)."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_VideoController).Name"],
                             capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    for name in out.stdout.splitlines():
        n = name.strip()
        # "AMD Radeon(TM) Graphics" is an APU's iGPU; a discrete Radeon carries a model name.
        if "NVIDIA" in n.upper() or ("RADEON" in n.upper() and n.upper() != "AMD RADEON(TM) GRAPHICS"):
            return n
    return None


def splat_gpu():
    """(bool, why): does the provider's splat capability run on a GPU here? Its EP=auto ranking:
    ONNX Runtime + DirectML on a discrete NVIDIA / AMD GPU (Windows), else OpenVINO's GPU device
    when it is an Intel Arc / Xe-class GPU (never NVIDIA / AMD through OpenCL, never a UHD-class
    iGPU), else a CPU backend."""
    system = platform.system()
    if system == "Darwin":
        return False, "macOS: no OpenVINO, the splat capability runs on ONNX Runtime's CPU"
    if system == "Windows":
        dgpu = windows_discrete_gpu()
        if dgpu:
            return True, f"Windows: DirectML on {dgpu}"
    try:
        import openvino

        core = openvino.Core()
        gpus = [d for d in core.available_devices if d.startswith("GPU")]
        for d in gpus:
            name = str(core.get_property(d, "FULL_DEVICE_NAME"))
            if "intel" in name.lower() and "uhd" not in name.lower() and "hd graphics" not in name.lower():
                return True, f"OpenVINO {d}: {name}"
        return False, f"OpenVINO GPU devices: {', '.join(gpus) or 'none'} (none is an Intel Arc / Xe GPU)"
    except Exception:  # noqa: BLE001 - openvino is optional here; any failure falls to the heuristic
        pass
    if system == "Windows":
        return False, "Windows: no discrete NVIDIA / AMD GPU found (pip install openvino to check an Intel GPU)"
    render = os.path.isdir("/dev/dri") and any(n.startswith("renderD") for n in os.listdir("/dev/dri"))
    icd = os.path.isdir("/etc/OpenCL/vendors") and bool(os.listdir("/etc/OpenCL/vendors"))
    return render and icd, (f"GPU render node {'present' if render else 'absent'}, OpenCL driver "
                            f"{'present' if icd else 'absent'}; OpenVINO's GPU device needs both")


def machine_variant():
    """The provider's MODEL=auto rule, applied to this machine: (variant, why)."""
    gpu, gpu_why = splat_gpu()
    ram = total_ram_gb()
    # The provider compares the memory the OS reports (a "16 GB" machine reports a bit less).
    ram_ok = ram is not None and ram * 1024 >= GPU_RAM_FLOOR_GB * 1024 - 1024
    ram_txt = "unknown memory" if ram is None else f"{ram:.1f} GB of memory"
    if gpu and ram_ok:
        return "full", f"a GPU ({gpu_why}) and {ram_txt} (at least {GPU_RAM_FLOOR_GB} GB)"
    return "int4", (f"{'a GPU' if gpu else 'no GPU'} ({gpu_why}) and {ram_txt}"
                    f"{'' if ram_ok else f' (under {GPU_RAM_FLOOR_GB} GB)'}")


# ---------------------------------------------------------------------------- store


def store_root(explicit):
    """The runtime's own root resolution (discovery spec section 5.4)."""
    if explicit:
        return explicit, "--store"
    env = os.environ.get("DXR_CONV_MODELS_ROOT")
    if env:
        return env, "DXR_CONV_MODELS_ROOT"
    if platform.system() == "Windows":
        import winreg

        for sub, value in (("Software\\DisplayXR\\Models", "Path"), ("Software\\DisplayXR\\Browser", "ModelsPath")):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, sub, 0,
                                    winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
                    v, _ = winreg.QueryValueEx(k, value)
                    if v:
                        return os.path.expandvars(v), f"HKLM {sub}\\{value}"
            except OSError:
                pass
        return os.path.join(os.environ.get("ProgramData", "C:\\ProgramData"), "DisplayXR", "Models"), "%ProgramData%"
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    root = os.path.join(xdg, "displayxr", "models")
    probe = root
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    if os.access(probe, os.W_OK):
        return root, "$XDG_DATA_HOME"
    return "/var/lib/displayxr/models", "system default"


def store_put(root, src, sha, size):
    """blobs/<sha>.<nonce>.part -> rename (never over an existing blob) -> .ok stamp last."""
    if not os.path.isdir(root):
        if platform.system() == "Windows":
            fail(f"model store {root} does not exist; create it with `displayxr-cli conv models init` first")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, STORE_OWNER_MARKER), "w", encoding="utf-8") as f:
            f.write("displayxr-runtime")
    blobs = os.path.join(root, "blobs")
    os.makedirs(blobs, exist_ok=True)
    blob, stamp = os.path.join(blobs, sha), os.path.join(blobs, sha + ".ok")
    if os.path.isfile(blob) and os.path.getsize(blob) == size and sha256_file(blob) == sha:
        log(f"store already holds {sha}")
    else:
        for stale in (blob, stamp):
            if os.path.exists(stale):
                os.remove(stale)
        part = os.path.join(blobs, f"{sha}.export-{os.getpid()}.part")
        shutil.copyfile(src, part)
        if os.path.getsize(part) != size or sha256_file(part) != sha:
            os.remove(part)
            fail("the copy into the store does not hash to the exported file")
        os.rename(part, blob)
    with open(stamp + ".tmp", "w", encoding="ascii", newline="\n") as f:
        f.write(f"{sha} {size}\n")
    os.replace(stamp + ".tmp", stamp)
    return blob


def pack_expectation(model_name):
    """(pack id, variant id, {role: (sha256, size)}) the pack lists for @p model_name, or None."""
    try:
        with open(PACK_FILE, encoding="utf-8") as f:
            pack = json.load(f)
    except (OSError, ValueError):
        return None
    for m in pack.get("models", []):
        if m.get("name") == model_name:
            for v in m.get("variants", []):
                files = {fl.get("role"): (fl.get("sha256"), fl.get("size")) for fl in v.get("files", [])}
                return pack.get("pack"), v.get("id"), files
    return None


# ---------------------------------------------------------------------------- check


def check_inputs(image_path):
    import numpy as np
    from PIL import ExifTags, Image

    img = Image.open(image_path)
    exif = img.getexif().get_ifd(0x8769)
    tags = {ExifTags.TAGS.get(k): v for k, v in exif.items()}
    # Upstream's rule (sharp.utils.io.load_rgb): the 35 mm-equivalent focal length, else the
    # raw focal length (x 8.4 when it is below 10 mm), else 30 mm.
    f35 = float(tags.get("FocalLengthIn35mmFilm") or 0)
    if f35 < 1:
        f35 = float(tags.get("FocalLength") or 30.0)
        if f35 < 10:
            f35 *= 8.4
    img = img.convert("RGB")
    w, h = img.size
    f_px = f35 * (w * w + h * h) ** 0.5 / (36 ** 2 + 24 ** 2) ** 0.5
    x = np.asarray(img.resize((RESOLUTION, RESOLUTION), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 255.0
    d = np.array([f_px / w], np.float32)
    return x, d


def check_reference(predictor, inputs, image_path):
    """PyTorch fp32 outputs, computed before quantization so the predictor can be freed."""
    import torch

    log(f"check: PyTorch fp32 on {os.path.basename(image_path)} ...")
    return [t[0].numpy().copy() for t in predictor(*(torch.from_numpy(a) for a in inputs))]


def check_model(name, model_path, inputs, ref):
    import numpy as np
    import onnxruntime as ort

    log(f"check: {name} on ONNX Runtime CPU ...")
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False  # the arena roughly doubles peak memory on this graph
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    got = [t[0] for t in sess.run(OUTPUTS, dict(zip(INPUTS, inputs)))]
    del sess
    for name, a, b in zip(OUTPUTS, got, ref):
        rel = float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))
        log(f"  {name:16s} relative L2 error {rel:.4f}")
    dz = np.abs(got[0][:, 2] - ref[0][:, 2]) / np.abs(ref[0][:, 2])
    worst = float(np.median(dz))
    log(f"  depth: median relative error {worst:.4f}, p95 {float(np.percentile(dz, 95)):.4f}")
    if worst > 0.03:
        fail(f"{name}: median relative depth error {worst:.3f} is above 3 %: the export does not match")


# ---------------------------------------------------------------------------- main


def missing_packages(variants):
    """The pip names of the packages @p variants need that cannot be imported here."""
    import importlib.util

    need = [("torch", "torch"), ("onnx", "onnx"), ("onnxruntime", "onnxruntime"), ("numpy", "numpy"),
            ("PIL", "pillow"), ("timm", "timm"), ("onnxscript", "onnxscript")]
    if "fp16" in variants:
        need.append(("onnxconverter_common", "onnxconverter-common"))
    return [pip for mod, pip in need if importlib.util.find_spec(mod) is None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-research-license", action="store_true",
                    help=f"confirm you accept Apple's research-only model licence ({LICENSE_URL})")
    ap.add_argument("--variant", choices=["int4", "full", "fp16", "both"],
                    help="what to build (default: the variant this machine's provider picks, MODEL=auto; "
                         "both = full + int4; fp16 is for DirectML only and is never a default)")
    ap.add_argument("--checkpoint", help=f"the official {CHECKPOINT_NAME} (downloaded into --work when omitted)")
    ap.add_argument("--ml-sharp", help=f"a clean apple/ml-sharp checkout at {ML_SHARP_COMMIT[:12]} (cloned when omitted)")
    ap.add_argument("--work", default=os.path.join(os.path.expanduser("~"), "displayxr-sharp-export"),
                    help="work directory for the download, the clone and the fp32 intermediate")
    ap.add_argument("--store", help="model store root (default: the runtime's own resolution)")
    ap.add_argument("--no-store", action="store_true", help="leave the built files in --work only")
    ap.add_argument("--keep-fp32", action="store_true", help="keep the fp32 intermediate in --work")
    ap.add_argument("--check", metavar="IMAGE", help="compare each built model against PyTorch fp32 on IMAGE")
    args = ap.parse_args()

    if not args.accept_research_license:
        print(f"SHARP is licensed for research use only: {LICENSE_URL}\n"
              "Read the licence, then re-run with --accept-research-license.", file=sys.stderr)
        return 2

    if args.variant is None:
        pick, why = machine_variant()
        log(f"--variant {pick} (this machine's provider would pick it: {why}; --variant both builds both)")
        variants = [pick]
    else:
        variants = ["full", "int4"] if args.variant == "both" else [args.variant]

    missing = missing_packages(variants)
    if missing:
        req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-export-sharp.txt")
        print(f"export_sharp.py {SCRIPT_VERSION} needs Python packages this environment does not have: "
              f"{', '.join(missing)}.\nInstall the pinned set into this environment (it also fixes the "
              f"versions the pack's hashes need):\n\n    {sys.executable} -m pip install -r {req}\n",
              file=sys.stderr)
        return 2

    import importlib.metadata

    import numpy
    import onnx
    import onnxruntime
    import torch

    try:
        occ = importlib.metadata.version("onnxconverter-common")
    except importlib.metadata.PackageNotFoundError:
        occ = "absent"

    versions = (f"python {platform.python_version()}, torch {torch.__version__}, onnx {onnx.__version__}, "
                f"onnxruntime {onnxruntime.__version__}, onnxconverter-common {occ}, "
                f"numpy {numpy.__version__}")
    log(f"export_sharp.py {SCRIPT_VERSION}: {versions}")

    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    if args.checkpoint:
        checkpoint = os.path.abspath(args.checkpoint)
        verify_checkpoint(checkpoint)
    else:
        checkpoint = download_checkpoint(os.path.join(work, "checkpoint"))
    ml_sharp_src = prepare_ml_sharp(os.path.abspath(args.ml_sharp) if args.ml_sharp else None, work)

    predictor = load_predictor(ml_sharp_src, checkpoint)
    fp32_dir = os.path.join(work, "fp32")
    fp32 = export_fp32(predictor, fp32_dir)
    check_fp32(fp32)
    if args.check:
        inputs = check_inputs(args.check)
        reference = check_reference(predictor, inputs, args.check)
    del predictor
    gc.collect()

    # variant -> [(role, path, sha256, size)], the model-role file last
    built = {}
    for variant in variants:
        name = VARIANTS[variant][0]
        provenance = provenance_for(variant)
        if variant == "full":
            built[variant] = write_full(fp32, os.path.join(work, name), provenance)
        elif variant == "fp16":
            built[variant] = convert_fp16(fp32, work, os.path.join(work, name), provenance)
        else:
            path = quantize_int4(fp32, os.path.join(work, f"{name}.onnx"), provenance)
            built[variant] = [("model", path, sha256_file(path), os.path.getsize(path))]
        gc.collect()
    if not args.keep_fp32:
        shutil.rmtree(fp32_dir, ignore_errors=True)

    if args.check:
        for variant in variants:
            check_model(VARIANTS[variant][0], built[variant][-1][1], inputs, reference)

    ok = True
    root = source = None
    if not args.no_store:
        root, source = store_root(args.store)
    for variant in variants:
        name = VARIANTS[variant][0]
        log("")
        for role, path, sha, size in built[variant]:
            log(f"{name} {role:5s}  sha256 {sha}  size {size}")
        log(f"  ({versions})")
        expect = pack_expectation(name)
        got = {role: (sha, size) for role, _, sha, size in built[variant]}
        if expect is None:
            log(f"  (no {name} entry in packs/dxr-open-sharp.models.json next to this script to compare against)")
        elif expect[2] == got:
            log(f"  matches the pack entry {expect[0]}/{name}/{expect[1]}")
        else:
            ok = False
            log(f"  WARNING: the pack lists {expect[2]}; these bytes differ.\n"
                "  The model store will not hand them out under the pack entry. Use the pinned package\n"
                "  versions (scripts/requirements-export-sharp.txt) to reproduce the listed bytes.")
        if root is not None:
            for role, path, sha, size in built[variant]:  # the model file last
                blob = store_put(root, path, sha, size)
                log(f"stored: {blob}  (store {root}, {source})")
    if root is not None:
        log("`displayxr-cli conv models list` shows them under the dxr-open-sharp pack once that pack is registered.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
