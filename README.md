# DisplayXR open-model conversion provider

`DisplayXR-OpenConversion` is a [DisplayXR runtime](https://github.com/DisplayXR/displayxr-runtime)
**conversion provider** (ADR-053): a plug-in that turns 2D into 3D for every app on the machine,
with open models only. It is the provider every box gets without a vendor module: id
`open-conversion`, **ProbeOrder 200** (after vendor providers at 50 and the legacy `dp-lift`
adapter at 150, before the sim provider at 250).

Apps never see it. They ask the runtime for lift through `XR_DXR_lift`; the runtime selects a
provider per task, owns every thread, queue and policy, and calls this one for pixels and units.

| Capability | Task | Lanes | Model (default pack) | Output |
|---|---|---|---|---|
| `cap_id` 1 | `DEPTH_VIDEO`, `TEMPORAL` | stream | Video Depth Anything **Small**, streaming export with an explicit per-stream temporal cache | R32F affine disparity |
| `cap_id` 2 | `DEPTH_MONO` | stream + job | Depth Anything V2 **Small** | R32F affine disparity |

- **Depth only.** The provider declares no `VIEWS_*` capability: the runtime composes
  `VIEWS_MONO` from a depth capability and its reference view synthesis (discovery spec §1.1,
  §6.2).
- **Honest units.** Both models produce relative disparity — larger is nearer, unknown scale and
  shift — reported as `XRT_CONV_DEPTH_AFFINE_DISPARITY`, `PLANAR`, intrinsics `NONE`, with the
  per-frame range and 5th / 95th percentiles before any normalisation. No metric depth, focal
  length or baseline is ever invented.
- **The inpainter** (nunif `light_inpaint_v1`, MIT) is in the default pack so it is pre-fetched
  with the depth models, but the provider declares no capability for it: ABI v1 reserves
  `XRT_CONV_TASK_INPAINT` without code, and hole fill for composed views is the runtime's
  reference stage. A later ABI slot can expose it without another download.
- **Inference:** ONNX Runtime 1.24.4, loaded at run time by absolute path from the provider's
  own directory. On Windows the DirectML execution provider runs on the adapter
  `host->get_adapter_hint` names (the first hardware adapter otherwise); when DirectML cannot
  start, the capability moves to the CPU execution provider and re-describes itself. Linux and
  macOS use the CPU execution provider.
- **Transport:** every capability imports `XRT_CONV_MEM_CPU` (RGBA8 / BGRA8, UNORM or SRGB) and
  writes a CPU R32F target; the runtime copies GPU images for it. The temporal cache stays on
  the inference device between frames.

## Model packs and the model store

`packs/dxr-open-default.models.json` is a schema-2 pack (discovery spec §5): every blob by
sha256 + size, an SPDX licence and a `commercial` flag per model and per dependency, the pack's
own `commercial` flag (the AND over its models), and `defaults` per task and quality.

| Model | Task | Blob sha256 | Size | Prefetch |
|---|---|---|---|---|
| `vda-small-stream-518x294` | depth-video | `71e3ae9e96606afe905eb2f4bab71700e7b9a5b6167bf9d145851720b452a949` | 57 803 548 | yes |
| `vda-small-stream-364x210` | depth-video (low) | `2bcf27020b45bd25f3ed4e28b3fc328c33af934661c0ee2be73a427a30d4f184` | 57 506 238 | no |
| `da2-small` | depth-mono | `2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04` | 49 642 442 | yes |
| `light-inpaint-v1-512x288` | aux:inpaint | `c1cd5b9ac2a37840efa135641f4f7e5824ee8f6b067472c0f5037d9b2e8be011` | 5 167 981 | yes |
| `light-inpaint-v1-1024x576` | aux:inpaint (high) | `2af8952a4da6e8fde91da120329656ef885e16e159faa25b821902688658ad36` | 6 936 267 | no |

The pre-fetch set is 112.6 MB. Only the **Small** variants of the depth families are Apache-2.0
(Base and Large are non-commercial), so only they are listed.

**The provider never downloads and never opens a store path on its own.** It reads the pack
for identity, I/O and licence, and resolves blob bytes only through the host:

| When | The provider calls | It reports |
|---|---|---|
| `probe` | `model_resolve(ref)` (non-blocking) | `INSTALLED` (cold; the model downloads on first use if it is not in the store) or `DOWNLOADING` if the host is already fetching it |
| `warm`, blob in the store | `model_resolve` → `READY` + path | `WARMING` (session load, one warm-up inference) → `READY` |
| `warm`, blob missing | `model_request(ref)`, then `model_resolve` every 50 ms | `DOWNLOADING` with the host's progress → `WARMING` → `READY` |
| `warm`, download refused (policy, no store) | `model_request` returns false | `ABSENT` — the runtime falls through to the next provider; a later `warm` asks again |
| `warm`, the store failed | `model_resolve` → `FAILED` | `FAILED(reason)` (retryable for `NETWORK`) |

The ref is `"<pack>/<model>/<variant>"`, e.g. `dxr-open-default/da2-small/fp16`. The host
verifies the sha256 on first load (discovery spec §5.3).

`scripts/check_model_pack.py` validates a pack (schema, sha256 + size, the licence AND, the
pack flag, `defaults`, and that no `prefetch` blob is non-commercial — a signed installer must
never pre-fetch one); `--verify-store DIR` checks a model store against it.

## Licence classes

`commercial` = the AND over the model, every dependency, the pack's flag and the inference
runtime. A commercial-clean model on the CPU execution provider is `PERMISSIVE` (ONNX Runtime is
MIT); on DirectML it is `COMMERCIAL`, because DirectML ships under Microsoft's redistributable
licence (commercial use allowed, not an open licence). Anything else is `RESEARCH_ONLY`
(`commercial_ok = 0`) and is selected only after the user opts in with `AllowResearchOnly`.

## Build

Plain CMake and a C11 compiler; the build fetches ONNX Runtime (and DirectML on Windows) as
pinned release archives and checks their SHA-256 before extracting.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows, run this from a Visual Studio 2022 developer prompt. Output in `build/`:
`DisplayXR-OpenConversion.dll` (`.so` / `.dylib`), `onnxruntime.dll` (+ `DirectML.dll`;
`libonnxruntime.so.1` / `libonnxruntime.dylib` on POSIX) and `dxr-open-default.models.json` —
the files that ship together, flat in one folder. The provider exports exactly one symbol,
`xrtConversionPluginNegotiate`, links the C runtime statically and imports only `KERNEL32` and
`dxgi` (Windows). `-DOPENCONV_WERROR=ON` turns warnings into errors, as CI does.
`-DOPENCONV_DOWNLOAD_DIR=<dir>` keeps the fetched archives across build trees.

## Tests

`open-conversion-fakehost` loads the provider by absolute path, negotiates against a stub host
iface, walks `describe_cap` (including a truncated `struct_size`), validates every descriptor,
and plays the **model store** from a pack: `model_resolve` / `model_request` with a `file://`
blob base, `.part` + size + sha256 check + rename + `.ok` stamp last — the store's write
protocol, without network.

| CTest | What it proves | Needs models |
|---|---|---|
| `describe` | negotiate (and refusal of another ABI major), describe, cold `INSTALLED`, licence class | no |
| `absent-runtime` | ONNX Runtime missing → every capability `ABSENT` | no |
| `runtime-too-old` | a library that serves no API version we need → `FAILED(VERSION)`, never `FAILED(MODEL)` | no |
| `research-only` | a pack with a non-commercial dependency → `RESEARCH_ONLY`, `commercial_ok = 0` | no |
| `download-forbidden` | blob missing and the download refused → `ABSENT` after `warm` | no |
| `download-ready-mono` | `DOWNLOADING` (host progress) → `WARMING` → `READY`, then a CPU-EP stream and job on a synthetic scene | yes |
| `download-ready-video` | the same for the video model, four frames with a reset mid-stream | yes |
| `check-model-pack`, `check-model-pack-research` | the pack validator on both packs | no (Python) |
| `default-ep` (`-DOPENCONV_TEST_GPU=ON`) | both capabilities on the default execution provider (DirectML on Windows) | yes |

The synthetic scene is a perspective checkerboard floor under a sky. The checks are statistics,
not bytes (execution providers differ in the last bits): every texel finite, the reported range
equal to the written values, percentiles inside it, the per-row mean correlating with the row
index above 0.6 and the bottom third reading nearer than the top third.

The inference tests need the blobs as `<sha256>.onnx` in one directory:

```
python3 scripts/fetch_models.py packs/dxr-open-default.models.json models-cache --models vda-small-stream-518x294,da2-small
cmake -S . -B build -DOPENCONV_TEST_MODELS_DIR=$PWD/models-cache
```

Without them those two tests report *skipped*.

## Settings

Read with `host->get_setting(KEY)`: `DXR_CONV_OPEN_CONVERSION_<KEY>` in the hosting process's
environment, else `Settings\<KEY>` under the registry row (Windows) or the manifest's
`settings` object (POSIX).

| Key | Values | Effect |
|---|---|---|
| `ENABLE` | `0` | `probe` declines |
| `QUALITY` | `low`, `medium` (default), `high` | which model `defaults[task]` names (low = the 364-px video model) |
| `EP` | `auto` (default), `cpu` | `cpu` forces the CPU execution provider |
| `THREADS` | integer (default 4) | CPU intra-op threads; 0 = ONNX Runtime's default (every core) |
| `GRAPH_OPT` | `basic`, `extended` (default), `all` | ONNX Runtime graph optimisation (`all` fails to load the fp16 still-depth model on the CPU execution provider of 1.24.4) |
| `PACK` | absolute path | another pack manifest (default: `dxr-open-default.models.json` next to the binary) |
| `ONNXRUNTIME` | absolute path | another ONNX Runtime library (default: next to the binary) |

## Releases and installing

A `vX.Y.Z` tag builds and attaches (unsigned):

- `DisplayXROpenConversionSetup-<version>.exe` (Windows, DirectML + CPU): installs under
  `Program Files\DisplayXR\Plugins\OpenConversion`, writes the `ConversionProviders\open-conversion`
  row (`ProbeOrder` 200, `Packs` = the shipped pack manifest, `Version` last) and, because the
  build is unsigned, the provider DLL's SHA-256 under `HKLM\Software\DisplayXR\Trust\Allowlist`
  (ADR-054 N4b: the runtime's provenance gate enforces on Windows and refuses an unsigned,
  non-allowlisted component — 0.1.0 shipped without the row and was never loaded). It downloads
  no model; the runtime model store fetches blobs by URL and SHA-256 on first use. The uninstaller
  removes only this provider's row, its allowlist row and files, never the model store.
- `displayxr-open-conversion-<version>-linux-x86_64.tar.gz` (CPU): `install.sh` copies the payload
  to `${XDG_DATA_HOME:-~/.local/share}/DisplayXR/OpenConversion` and writes the JSON manifest.
  `uninstall.sh` removes both.

## Registration

Windows: `HKLM\Software\DisplayXR\ConversionProviders\open-conversion` with `Binary`,
`DisplayName`, `Vendor`, `ProbeOrder` = 200, `Packs` (the pack manifest's absolute path) and
`Version` written last — see `examples/settings/open-conversion.reg`. A hand registration of an
unsigned build also needs the DLL's SHA-256 as a value name under
`HKLM\Software\DisplayXR\Trust\Allowlist` (the runtime's `scripts\register_dev_plugin.bat conv
<dll>` writes both), or the runtime logs `provenance UNVERIFIED` and skips it. POSIX:
`200-open-conversion-conversion-provider.json` in a plug-in root — see `examples/manifest/`.
The provider is satellite-hosted by default (one `displayxr-conversion-host` process); updates
follow the runtime's installer contract (discovery spec §3.8: own
`Global\DisplayXR-ConvMaintenance-open-conversion`, close the host with Restart Manager, never
the service).

## Hosting rules this provider follows

No runtime device (it creates its own inference session on the hinted adapter); imports CPU
images only and exports nothing; no window, no process-state change, no `getenv`, no network
(the host downloads); ONNX Runtime by absolute path with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`;
files held open only while read; `destroy` may never be called and nothing needs flushing; GPU
loss is reported as `FAILED(DEVICE_LOST, retryable)` and the runtime re-warms. One call in
flight per provider (no capability opts into overlap); `get_provider_status`, `describe_cap`,
`get_generation`, `get_state` and `stream_get_output_desc` answer from memory.

## The ABI it builds against

`include/xrt/` is vendored **verbatim** from the runtime's `main` at `v2.42.0-68-g6cffa48b3`
(`src/xrt/include/xrt/xrt_conversion_plugin.h`, `xrt_results.h`, `xrt_conv_wire.h`):
conversion-provider ABI major `XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT` = **1**,
binary-identical to the `v2.42.0` release (the difference is comments only). The provider uses
`xrt_conversion_plugin.h` and `xrt_results.h`; `xrt_conv_wire.h` is the runtime's IPC wire,
vendored for reference and compile-checked only. Update by copying from a runtime release.

## Licence

[BSL-1.0](LICENSE), like the DisplayXR runtime. Third-party components and model licences:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
