#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""Build the local SHARP (image -> Gaussian splats) ONNX model from Apple's official checkpoint.

    export_sharp.py --accept-research-license [--checkpoint FILE] [--ml-sharp DIR]
                    [--work DIR] [--store DIR | --no-store] [--keep-fp32] [--check IMAGE]

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
   1x3x1536x1536 image input) into the work directory. This file is an intermediate (2.6 GB,
   external weights) and is deleted unless --keep-fp32.
4. Quantizes it with ONNX Runtime's MatMulNBitsQuantizer (4-bit, block 32, symmetric): only
   the weights of MatMul nodes with a constant weight become 4-bit (com.microsoft MatMulNBits);
   Conv weights, every other weight and every activation stay fp32. The result is one
   self-contained .onnx file (about 0.58 GB).
5. Writes that file into the DisplayXR model store (blobs/<sha256> plus its .ok stamp, the
   store's own write protocol), so the `dxr-open-sharp` pack resolves it, and prints the
   SHA-256 and size. The pack lists exactly these bytes: the export is byte-deterministic with
   the pinned package versions in scripts/requirements-export-sharp.txt, and the script checks
   the hash it produced against the pack and says so if they differ.

--check IMAGE runs the PyTorch fp32 reference and the int4 model (ONNX Runtime, CPU) on one
image and prints the per-tensor relative error and the median relative depth error. It takes
a few minutes and about 16 GB of memory (ONNX Runtime's CPU execution provider peaks near 14 GB here).

Needs: Python 3.11+, git, about 8 GB free memory, about 7 GB free disk in --work, and the
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

SCRIPT_VERSION = "1.0.0"

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
PACK_MODEL = "sharp-int4-1536"
STORE_OWNER_MARKER = "displayxr-runtime.owner"


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


def pack_expectation():
    try:
        with open(PACK_FILE, encoding="utf-8") as f:
            pack = json.load(f)
    except (OSError, ValueError):
        return None
    for m in pack.get("models", []):
        if m.get("name") == PACK_MODEL:
            for v in m.get("variants", []):
                for fl in v.get("files", []):
                    if fl.get("role") == "model":
                        return pack.get("pack"), v.get("id"), fl.get("sha256"), fl.get("size")
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


def check_int4(int4_path, inputs, ref):
    import numpy as np
    import onnxruntime as ort

    log("check: int4 on ONNX Runtime CPU ...")
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False  # the arena roughly doubles peak memory on this graph
    sess = ort.InferenceSession(int4_path, so, providers=["CPUExecutionProvider"])
    got = [t[0] for t in sess.run(OUTPUTS, dict(zip(INPUTS, inputs)))]
    del sess
    for name, a, b in zip(OUTPUTS, got, ref):
        rel = float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))
        log(f"  {name:16s} relative L2 error {rel:.4f}")
    dz = np.abs(got[0][:, 2] - ref[0][:, 2]) / np.abs(ref[0][:, 2])
    worst = float(np.median(dz))
    log(f"  depth: median relative error {worst:.4f}, p95 {float(np.percentile(dz, 95)):.4f}")
    if worst > 0.03:
        fail(f"median relative depth error {worst:.3f} is above 3 %: the export does not match")


# ---------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-research-license", action="store_true",
                    help=f"confirm you accept Apple's research-only model licence ({LICENSE_URL})")
    ap.add_argument("--checkpoint", help=f"the official {CHECKPOINT_NAME} (downloaded into --work when omitted)")
    ap.add_argument("--ml-sharp", help=f"a clean apple/ml-sharp checkout at {ML_SHARP_COMMIT[:12]} (cloned when omitted)")
    ap.add_argument("--work", default=os.path.join(os.path.expanduser("~"), "displayxr-sharp-export"),
                    help="work directory for the download, the clone and the fp32 intermediate")
    ap.add_argument("--store", help="model store root (default: the runtime's own resolution)")
    ap.add_argument("--no-store", action="store_true", help="leave the int4 file in --work only")
    ap.add_argument("--keep-fp32", action="store_true", help="keep the fp32 intermediate in --work")
    ap.add_argument("--check", metavar="IMAGE", help="compare the int4 model against PyTorch fp32 on IMAGE")
    args = ap.parse_args()

    if not args.accept_research_license:
        print(f"SHARP is licensed for research use only: {LICENSE_URL}\n"
              "Read the licence, then re-run with --accept-research-license.", file=sys.stderr)
        return 2

    import numpy
    import onnx
    import onnxruntime
    import torch

    versions = (f"python {platform.python_version()}, torch {torch.__version__}, onnx {onnx.__version__}, "
                f"onnxruntime {onnxruntime.__version__}, numpy {numpy.__version__}")
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

    provenance = {
        "dxr.model": PACK_MODEL,
        "dxr.converter": f"displayxr-conversion-open scripts/export_sharp.py {SCRIPT_VERSION}",
        "dxr.parent.url": CHECKPOINT_URL,
        "dxr.parent.sha256": CHECKPOINT_SHA256,
        "dxr.code": f"apple/ml-sharp@{ML_SHARP_COMMIT}",
        "dxr.license": "apple-amlr (research only) " + LICENSE_URL,
    }
    int4 = quantize_int4(fp32, os.path.join(work, f"{PACK_MODEL}.onnx"), provenance)
    if not args.keep_fp32:
        shutil.rmtree(fp32_dir, ignore_errors=True)
    size = os.path.getsize(int4)
    sha = sha256_file(int4)

    if args.check:
        check_int4(int4, inputs, reference)

    log("")
    log(f"{PACK_MODEL}.onnx  sha256 {sha}  size {size}")
    log(f"  ({versions})")
    expect = pack_expectation()
    if expect is None:
        log("  (no packs/dxr-open-sharp.models.json next to this script to compare against)")
    elif expect[2] == sha and expect[3] == size:
        log(f"  matches the pack entry {expect[0]}/{PACK_MODEL}/{expect[1]}")
    else:
        log(f"  WARNING: the pack lists sha256 {expect[2]} size {expect[3]}; these bytes differ.\n"
            "  The model store will not hand them out under the pack entry. Use the pinned package\n"
            "  versions (scripts/requirements-export-sharp.txt) to reproduce the listed bytes.")

    if not args.no_store:
        root, source = store_root(args.store)
        blob = store_put(root, int4, sha, size)
        log(f"stored: {blob}  (store {root}, {source})")
        log("`displayxr-cli conv models list` shows it under the dxr-open-sharp pack once that pack is registered.")
    return 0 if expect is None or (expect[2] == sha and expect[3] == size) else 1


if __name__ == "__main__":
    sys.exit(main())
