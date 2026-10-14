# Third-party components

The provider's own source is [BSL-1.0](LICENSE). It ships next to, and loads at run time, the
components below; none is linked into the provider binary and none is committed to this
repository (the build fetches them as pinned, hash-checked release archives, see
`cmake/OnnxRuntime.cmake`; the models are fetched by the runtime's model store).

## Inference runtime

| Component | Version | Licence | Where it comes from | Platforms |
|---|---|---|---|---|
| ONNX Runtime | 1.24.4 | MIT | `Microsoft.ML.OnnxRuntime.DirectML` NuGet package (Windows), `onnxruntime-{linux-x64,linux-aarch64,osx-arm64}-1.24.4.tgz` GitHub release assets | all |
| DirectML | 1.15.4 | Microsoft Software License Terms for DirectML (redistributable; commercial redistribution in applications permitted; not an open-source licence) | `Microsoft.AI.DirectML` NuGet package | Windows |

The licence texts are installed under `licenses/` next to the binary (`onnxruntime-LICENSE.txt`,
`onnxruntime-ThirdPartyNotices.txt`, `DirectML-LICENSE.txt`).

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
