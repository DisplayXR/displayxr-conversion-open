# Third-party components

The provider's own source is [BSL-1.0](LICENSE). It ships next to, and loads at run time, the
components below; none is linked into the provider binary and none is committed to this
repository (the build fetches them as pinned, hash-checked release archives, see
`cmake/OnnxRuntime.cmake` and `cmake/OpenVINO.cmake`; the models are fetched by the runtime's
model store, or built locally by the user for a research-only pack).

## Inference runtime

| Component | Version | Licence | Where it comes from | Platforms |
|---|---|---|---|---|
| ONNX Runtime | 1.24.4 | MIT | `Microsoft.ML.OnnxRuntime.DirectML` NuGet package (Windows), `Microsoft.ML.OnnxRuntime` NuGet package (Linux x64 / arm64, macOS arm64) | all |
| DirectML | 1.15.4 | Microsoft Software License Terms for DirectML (redistributable; commercial redistribution in applications permitted; not an open-source licence) | `Microsoft.AI.DirectML` NuGet package | Windows |
| OpenVINO (core, C API, ONNX + IR front ends, CPU + GPU device plug-ins) | 2026.4.1 | Apache-2.0 | Intel's release archives `openvino_toolkit_ubuntu22_2026.4.1…_x86_64.tgz` / `openvino_toolkit_windows_vc_mt_2026.4.1…_x86_64.zip` from storage.openvinotoolkit.org | Linux x86_64, Windows x64 |
| oneTBB (with OpenVINO) | 2021.13 | Apache-2.0 | same archives | Linux x86_64, Windows x64 |
| hwloc (oneTBB's NUMA binding, Linux) | 2.x | BSD-3-Clause | same archive | Linux x86_64 |

The licence texts are installed under `licenses/` next to the binary (`onnxruntime-LICENSE.txt`,
`onnxruntime-ThirdPartyNotices.txt`, `DirectML-LICENSE.txt`, and with OpenVINO
`openvino-LICENSE`, the archive's `openvino-*third-party-programs.txt` and `oneTBB-LICENSE.txt`).

OpenVINO's GPU device plug-in drives the system's OpenCL driver (Linux: `libOpenCL.so.1` and
Intel's compute runtime; Windows: the graphics driver). Nothing redistributes it; without it the
splat capability runs on OpenVINO's CPU device.

On Windows `onnxruntime.dll` imports the Microsoft Visual C++ 2015-2022 runtime
(`MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`); the installer that ships the
provider must make sure it is present. The provider binary itself links the C runtime
statically.

## Models (the default pack, `packs/dxr-open-default.models.json`)

| Model | Upstream | Licence | Blob |
|---|---|---|---|
| Video Depth Anything, Small (streaming export with an explicit temporal cache) | [DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything), weights `depth-anything/Video-Depth-Anything-Small` | Apache-2.0 (only the Small variant; Base and Large are CC-BY-NC-4.0 and are not listed) | DisplayXR export, `displayxr-models` release |
| Depth Anything V2, Small | `depth-anything/Depth-Anything-V2-Small`, ONNX by `onnx-community/depth-anything-v2-small` (`onnx/model_fp16.onnx`, byte-identical) | Apache-2.0 (only the Small variant) | same file as upstream |
| nunif iw3 `light_inpaint_v1` | [nagadomi/nunif](https://github.com/nagadomi/nunif) | MIT | DisplayXR export, `displayxr-models` release |

Both depth models build on the DINOv2 ViT-S/14 backbone (Apache-2.0).

## Splat models (separate packs, never bundled)

A photo-to-splat model is not in the default pack. A research-only model such as Apple's SHARP
(Apple Machine Learning Research Model licence, research use only, following derivatives) is a
pack of its own whose weights are never shipped, mirrored or pre-fetched: the user builds them
locally from the official checkpoint (see that pack and its export script). The provider reports
such a capability as `RESEARCH_ONLY`, which the runtime selects only with `AllowResearchOnly`.
