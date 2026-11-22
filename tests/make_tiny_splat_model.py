#!/usr/bin/env python3
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
"""Write a tiny synthetic splats-mono model, its pack and the expected output.

    make_tiny_splat_model.py OUT_DIR

The model has the I/O contract of a real photo-to-splat model (discovery spec
§5, a "splats-mono" pack entry with io.output.splat_space "ndc"): inputs
`image` [1,3,32,32] and `disparity_factor` [1], outputs `mean_vectors`,
`singular_values`, `quaternions`, `colors` and `opacities` for 512 Gaussians
(two 16 x 16 layers). Geometry is fixed except for depth, which is scaled by
1 / disparity_factor; colours are the 2 x 2 average of the picture. So for a
uniform grey picture of a known size and focal, every Gaussian the provider
writes is predictable: expect.json carries a sample of them, computed here
independently of the provider (numpy), for the fake host to compare.

Nothing here is a real model: it exists so CI exercises the splat lane
(preprocess, inference on ONNX Runtime and on OpenVINO's CPU device,
unprojection, covariance re-diagonalisation, PLY, facts) without a GPU and
without any model download. Needs the `onnx` and `numpy` packages; exits 77
(CTest "skipped") without them.
"""

import hashlib
import json
import os
import sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as e:  # pragma: no cover
    print(f"SKIPPED: {e}")
    sys.exit(77)

GRID = 32
LAYER = (GRID // 2) ** 2
N = 2 * LAYER
WIDTH, HEIGHT, FOCAL = 400, 300, 360.0  # the picture the fake host sends (uniform grey 128)
GREY = 128.0 / 255.0
SH_C0 = 0.28209479177387814


def constants(rng):
    ys, xs = np.meshgrid(np.arange(GRID // 2), np.arange(GRID // 2), indexing="ij")
    x = ((xs + 0.5) / (GRID // 2) * 2.0 - 1.0).reshape(-1)
    y = ((ys + 0.5) / (GRID // 2) * 2.0 - 1.0).reshape(-1)
    means = np.zeros((N, 3), np.float32)
    means[:LAYER, 0], means[:LAYER, 1], means[:LAYER, 2] = x, y, 2.0
    means[LAYER:, 0], means[LAYER:, 1], means[LAYER:, 2] = x, y, 3.0 + 0.5 * x  # a tilted back layer
    scales = rng.uniform(0.003, 0.03, (N, 3)).astype(np.float32)
    quats = rng.normal(size=(N, 4)).astype(np.float32)
    quats *= rng.uniform(0.5, 2.0, (N, 1)).astype(np.float32)  # unnormalised on purpose
    opac = rng.uniform(0.05, 0.95, (N,)).astype(np.float32)
    return means, scales, quats, opac


def build_model(means, scales, quats, opac):
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, GRID, GRID])
    factor = helper.make_tensor_value_info("disparity_factor", TensorProto.FLOAT, [1])
    outs = [
        helper.make_tensor_value_info("mean_vectors", TensorProto.FLOAT, [1, N, 3]),
        helper.make_tensor_value_info("singular_values", TensorProto.FLOAT, [1, N, 3]),
        helper.make_tensor_value_info("quaternions", TensorProto.FLOAT, [1, N, 4]),
        helper.make_tensor_value_info("colors", TensorProto.FLOAT, [1, N, 3]),
        helper.make_tensor_value_info("opacities", TensorProto.FLOAT, [1, N]),
    ]
    init = [
        numpy_helper.from_array(means[None], "means_c"),
        numpy_helper.from_array(scales[None], "scales_c"),
        numpy_helper.from_array(quats[None], "quats_c"),
        numpy_helper.from_array(opac[None], "opac_c"),
        numpy_helper.from_array(np.ones((2,), np.float32), "ones2"),
        numpy_helper.from_array(np.array([1, 3, LAYER], np.int64), "shape_c"),
    ]
    nodes = [
        # depth / disparity_factor
        helper.make_node("Reciprocal", ["disparity_factor"], ["inv_df"]),
        helper.make_node("Concat", ["ones2", "inv_df"], ["mvec"], axis=0),
        helper.make_node("Mul", ["means_c", "mvec"], ["mean_vectors"]),
        helper.make_node("Identity", ["scales_c"], ["singular_values"]),
        helper.make_node("Identity", ["quats_c"], ["quaternions"]),
        helper.make_node("Identity", ["opac_c"], ["opacities"]),
        # colours: 2x2 average of the picture, one per Gaussian of each layer
        helper.make_node("AveragePool", ["image"], ["pooled"], kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Reshape", ["pooled", "shape_c"], ["flat"]),
        helper.make_node("Transpose", ["flat"], ["layer_cols"], perm=[0, 2, 1]),
        helper.make_node("Concat", ["layer_cols", "layer_cols"], ["colors"], axis=1),
    ]
    graph = helper.make_graph(nodes, "tiny_splats", [image, factor], outs, init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="displayxr-test")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def quat_rot(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def srgb(v):
    v = min(max(v, 0.0), 1.0)
    return v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def expected(means, scales, quats, opac):
    df = FOCAL / WIDTH
    S = np.array([WIDTH / (2 * FOCAL), HEIGHT / (2 * FOCAL), 1.0])
    dc = (srgb(GREY) - 0.5) / SH_C0
    samples = []
    for i in list(range(0, N, 37)) + [N - 1]:
        m = means[i].astype(np.float64) * np.array([1.0, 1.0, 1.0 / df])
        R = quat_rot(quats[i].astype(np.float64))
        cov = R @ np.diag(scales[i].astype(np.float64) ** 2) @ R.T
        cov = cov * S[:, None] * S[None, :]
        a = float(opac[i])
        samples.append({
            "i": i,
            "xyz": (m * S).tolist(),
            "f_dc": [dc, dc, dc],
            "opacity": float(np.log(a / (1 - a))),
            "cov": [cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2]],
        })
    return {"width": WIDTH, "height": HEIGHT, "focal": FOCAL, "count": N, "tol": 2e-4, "samples": samples}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    consts = constants(np.random.default_rng(2026))
    data = build_model(*consts).SerializeToString()
    sha = hashlib.sha256(data).hexdigest()
    with open(os.path.join(out, f"{sha}.onnx"), "wb") as f:
        f.write(data)
    pack = {
        "schema": 2,
        "pack": "dxr-open-test-splats",
        "generated": "2026-11-22",
        "provider": "open-conversion",
        "blobBaseUrl": "file://" + os.path.abspath(out).replace(os.sep, "/"),
        "commercial": False,
        "models": [{
            "name": "tiny-splats",
            "version": "1.0.0",
            "task": "splats-mono",
            "io": {
                "input": {"color_space": "srgb", "range": [0, 1], "layout": "nchw-rgb",
                          "resolution": f"fixed-{GRID}x{GRID}", "native_long_edge": GRID, "intrinsics_in": "focal"},
                "output": {"coords": "opencv", "splat_format": "sharp-projective", "color_space": "linear",
                           "count": N, "intrinsics_out": "none"},
                "tensors": {"image": "image", "disparity_factor": "disparity_factor", "means": "mean_vectors",
                            "scales": "singular_values", "rotations": "quaternions", "colors": "colors",
                            "opacities": "opacities"},
            },
            "license": {"spdx": "LicenseRef-test-only", "link": "https://example.invalid/test", "commercial": False,
                        "attribution": "synthetic test model, no weights of any real model"},
            "provenance": {"converted_by": "tests/make_tiny_splat_model.py"},
            "prefetch": False,
            "variants": [{
                "id": "fp32", "backend": "onnx", "ep": ["openvino", "cpu"], "precision": "fp32",
                "files": [{"role": "model", "format": "onnx", "sha256": sha, "size": len(data)}],
            }],
        }],
        "defaults": {"splats-mono": {"low": "tiny-splats", "medium": "tiny-splats", "high": "tiny-splats"}},
    }
    with open(os.path.join(out, "tiny.models.json"), "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)
    with open(os.path.join(out, "expect.json"), "w", encoding="utf-8") as f:
        json.dump(expected(*consts), f, indent=1)
    print(f"tiny splat model {sha} ({len(data)} bytes) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
