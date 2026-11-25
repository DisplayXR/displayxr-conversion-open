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
| `cap_id` 3 | `SPLATS_MONO` | job | none in the default pack: a separate, research-only splat pack (e.g. SHARP, full or 4-bit, picked per machine) | 3DGS PLY ([below](#photo-to-splats-splats_mono-adr-057)) |

- **No views.** The provider declares no `VIEWS_*` capability: the runtime composes
  `VIEWS_MONO` from a depth capability and its reference view synthesis (discovery spec §1.1,
  §6.2). The splat capability is a separate job-lane task.
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
  start, the capability moves to the CPU execution provider and re-describes itself. On macOS
  the Core ML execution provider (part of the official osx-arm64 ONNX Runtime build) runs the
  video-depth model on the GPU, with the same fallback; the still-depth model stays on the CPU
  execution provider there (see [macOS: Core ML](#macos-core-ml)). Linux uses the CPU
  execution provider for depth. The splat capability runs on ONNX Runtime + **DirectML** on a
  discrete NVIDIA / AMD GPU (Windows), else on **OpenVINO** where the provider ships it (Linux
  x86_64, Windows x64: its GPU device when that is an Intel Arc / Xe-class GPU, else its CPU
  device), else on ONNX Runtime's CPU execution provider
  ([where splats run](#where-splats-mono-runs-ep-ov_device)).
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
| `warm`, blob in the store, not yet verified by this process | `model_resolve` → `INSTALLED`, `model_request(ref)`, then `model_resolve` every 50 ms while it reads `WARMING` (the store re-hashes it) | `WARMING` ("the model store is verifying it", the host's progress when it gives one) → session load → `READY`; `ABSENT` only when the store's state and progress stop moving for 2 min, or for the model's bytes at 8 MiB/s if longer |
| `warm`, blob missing | `model_request(ref)`, then `model_resolve` every 50 ms | `DOWNLOADING` with the host's progress → `WARMING` → `READY` |
| `warm`, download refused (policy, no store) | `model_request` returns false | `ABSENT` — the runtime falls through to the next provider; a later `warm` asks again |
| `warm`, the store failed | `model_resolve` → `FAILED` | `FAILED(reason)` (retryable for `NETWORK`) |

The ref is `"<pack>/<model>/<variant>"`, e.g. `dxr-open-default/da2-small/fp16`. The host
verifies the sha256 on first load (discovery spec §5.3).

`scripts/check_model_pack.py` validates a pack (schema, sha256 + size, the licence AND, the
pack flag, `defaults`, and that no `prefetch` blob is non-commercial — a signed installer must
never pre-fetch one); `--verify-store DIR` checks a model store against it.

### SHARP (image → Gaussian splats), built on your machine

`packs/dxr-open-sharp.models.json` lists three builds of Apple's SHARP single-image splat model
(task `splats-mono`), all exported to ONNX from the same official checkpoint:

- **`sharp-full-1536`** — the fp32 export as it comes out of PyTorch (numerically the PyTorch
  model: on ONNX Runtime's CPU every output is within 2e-4 relative L2 of PyTorch's and the median
  depth error is below 1e-4, `export_sharp.py --check`). 2.6 GB of weights, so it is
  **two files**: a 6 MB graph and one external-data file. The graph names that data file by
  the data's own SHA-256, which is exactly the name the model store gives it (`blobs/<sha256>`),
  so the two sit side by side in the store and the inference runtime finds the weights next to
  the graph; the pack lists both (roles `model` and `data`) and the store hands the model out
  only when both are present and verified. (ONNX's protobuf cannot hold 2.6 GB in one file.)
- **`sharp-int4-1536`** — the same export with its MatMul weights quantized to 4 bits; one
  0.58 GB file.
- **`sharp-fp16-1536`** — for **DirectML only** (its one variant lists `dml`): the same export
  with its static shape arithmetic folded (in Python, with ONNX's reference operators and only
  exactly-defined ones: integer arithmetic, IEEE `+ - * / sqrt`, data movement; no ONNX Runtime
  graph optimiser, whose saved graph differed between Windows and Linux), then converted to
  float16. Its graph inputs and outputs stay **float32** (the script checks every one against the
  fp32 model and refuses otherwise). The whole graph runs in fp16 except onnxconverter-common's
  default block list (in this graph: its two `Resize` nodes): the full-resolution convolutions of
  the decoders and the Gaussian heads are where the time and the memory are, and keeping them fp32
  (export_sharp.py 1.3.0) cost about 2.6× on DirectML and went over a 10 GB card's memory budget.
  Graph + one external-data file named by its SHA-256, like `full`, about 1.3 GB: half the GPU
  memory of `full`. Against the fp32 model on ONNX Runtime's CPU EP (one photo): median relative
  depth error 0.10 % (p95 1.4 %); per-attribute mean |difference| / mean |value|: positions
  0.78 %, colours 0.20 %, rotations 0.32 %, opacities 0.34 %, scales 2.1 %; no non-finite value.
  Measured on DirectML (RTX 3080, 10 GB, this recipe): about **1.2 s** of inference and **5.6 GB**
  of dedicated memory (within the budget), about 2 s for the whole job, where the fp32 model, over
  the card's memory budget, took 16 – 190 s.

The provider picks between them per machine (setting `MODEL`, [below](#which-sharp-build-model)).
SHARP's weights are licensed **for research only** (Apple Machine Learning Research Model
licence, `apple-amlr`), and the licence follows every derivative, so this repository never
ships, mirrors or links a weight file. The pack has no download URL; you build the blobs
yourself from the official checkpoint:

```
python3.12 -m venv sharp-venv
sharp-venv/bin/pip install -r scripts/requirements-export-sharp.txt
sharp-venv/bin/python scripts/export_sharp.py --accept-research-license [--variant int4|full|fp16|both]
```

`--variant` defaults to the build this machine's provider will pick by itself (the `MODEL=auto`
rule below) and says which and why; `both` builds `full` and `int4` from one export (one
checkpoint download serves both). `fp16` is never a default: build it on a DirectML machine
(a discrete NVIDIA / AMD card) when you want it; the provider then prefers it there.

The script downloads `sharp_2572gikvuh.pt` from the official `apple/Sharp` Hugging Face
repository at a pinned revision (or takes `--checkpoint FILE`), refuses any file whose size and
SHA-256 are not the official ones, checks out Apple's reference code `apple/ml-sharp` at a
pinned commit, exports the fp32 predictor (opset 18, standard operators, fixed 1536×1536
input); for `full` it re-saves that export as the graph + one data file, for `int4` it quantizes
the MatMul weights with ONNX Runtime's `MatMulNBitsQuantizer` (4-bit, block 32, symmetric; Conv
weights and all activations stay fp32); then it writes the result into the model store
(`blobs/<sha256>` + `.ok` stamp, the store's own protocol; the data file before the graph). It
takes about two minutes for both and about 8 GB of memory (7.3 GB peak here). With the pinned
package versions both outputs are byte-identical from run to run (checked by building each
twice), so the pack lists their SHA-256 like any other blob; the script compares what it
produced against the pack and says so if they differ.

| Model | Task | File | Blob sha256 | Size | Prefetch |
|---|---|---|---|---|---|
| `sharp-full-1536` | splats-mono | graph (`model`) | `cbe7db8a32a5e1d3af96fbd64841d0e77b8db79f3efba92fda3891e3edc75326` | 6 226 171 | never (built locally) |
| | | weights (`data`) | `95912399a850769c894d780c1761604ced31ce30e2464a8c3138e796f26e15d0` | 2 615 734 016 | |
| `sharp-int4-1536` | splats-mono | `model` | `01202f427e5f83ece061a5438e5b6670fbe9ffb26d30f8100bb285eb69f753f9` | 583 477 535 | never (built locally) |
| `sharp-fp16-1536` | splats-mono | graph (`model`) | `69a094fa7e2b2a479f02a6c77dda5edc7fdf6de647741055c533b5c171f534cc` | 3 676 447 | never (built locally) |
| | | weights (`data`) | `d3a96e008fdb4cc872c268b9541e905cb602353e8fb785c4a47ee54a29e92e05` | 1 308 059 008 | |

Each model's `local_build` block names the script, the version that introduced its recipe and
whether it is deterministic; `provenance` pins the parent checkpoint (`parent_sha256`,
`parent_size`). The version written into a blob's metadata is its recipe's, not the script's, so
an unrelated script change never moves a pinned hash (the int4 and full bytes are unchanged by
1.2.0, which added fp16, and 1.3.0 / 1.4.0, which rebuilt it). The fp16 recipe (1.4.0) is
byte-identical from run to run and across `PYTHONHASHSEED` values (checked on Linux); its 1.2.0
recipe went through ONNX Runtime's optimised-graph writer and gave other bytes on Windows. The capability is
`RESEARCH_ONLY` and is selected only with `AllowResearchOnly`. The model's tensors, shapes and
coordinate convention (projective means, unnormalised `wxyz` quaternions, linear-RGB colours)
are in the pack's `io` block.

## Licence classes

`commercial` = the AND over the model, every dependency and the inference runtime. It is decided
**per capability, from its own model**: the pack's `commercial` flag is the AND over all its
models, a summary `check_model_pack.py` verifies, and a research-only model lives in its own pack,
so it never changes another pack's class. A commercial-clean model on the CPU execution provider is
`PERMISSIVE` (ONNX Runtime is MIT); on DirectML it is `COMMERCIAL`, because DirectML ships under
Microsoft's redistributable licence (commercial use allowed, not an open licence). On the Core ML
execution provider it stays `PERMISSIVE`: the execution provider is part of ONNX Runtime (MIT) and
drives the operating system's own Core ML framework, which nothing redistributes. On OpenVINO it
stays `PERMISSIVE` too: OpenVINO and oneTBB are Apache-2.0 (hwloc, which oneTBB's NUMA binding
loads, is BSD-3-Clause), and the GPU device drives the system's own OpenCL driver, which nothing
redistributes. Anything else is `RESEARCH_ONLY` (`commercial_ok = 0`) and is selected only after
the user opts in with `AllowResearchOnly`: a SHARP pack (Apple's research-only model licence,
which follows derivatives) is `RESEARCH_ONLY` on every backend.

## Photo to splats (`SPLATS_MONO`, ADR-057)

`cap_id` 3 turns one photo into a 3D Gaussian splat on the **job lane** (one photo in, one
blob out, progress and cancel), for whatever model the packs name under `defaults["splats-mono"]`.
The default pack names none: a splat model is a separate, research-only pack (Apple's SHARP is
`packs/dxr-open-sharp.models.json`, model `sharp-int4-1536`), so without it the capability is
`ABSENT` and costs nothing. It is job-only and never prewarmed (the runtime's default `Prewarm`
tasks do not include it, and its warm runs no warm-up inference).

### Which SHARP build (`MODEL`)

When a pack names two splat models, `defaults["splats-mono"].high` (the full one) and `.low`
(the small one), setting **`MODEL`** picks between them; a third model named by `.medium` that has
a DirectML variant (the SHARP pack's `sharp-fp16-1536`) is a candidate on DirectML only:

| `MODEL` | Model |
|---|---|
| `auto` (default) | On **DirectML** (with at least 15 GiB of memory, below): **fp16** when its blob is in the store and the adapter has at least **6 GB** of dedicated memory (an estimate: half the fp32 model's measured use plus a desktop); else **full** with at least **9 GB** (a "10 GB" card reports about 10 050 MB) (measured, [below](#directml-measured-and-diagnosed)); else **int4** (which never runs on DirectML, so DirectML is not chosen). On any other GPU backend (OpenVINO's GPU device; Core ML where a pack lists it): **full** when the OS reports at least 15 GiB of memory (a "16 GB" machine: firmware and an integrated GPU keep a little). Otherwise **int4** (every CPU backend). |
| `full` | always the full model |
| `fp16` | the fp16 model on DirectML; the full model anywhere else |
| `int4` | always the small model |

`auto` chooses only among the models whose blobs are in the model store: when the preferred one
is absent and another is present (fp16, then full, then int4; fp16 only on DirectML) it uses that
one and says so (an INFO line and the
capability's ready detail, `… is not in the store; using …`); when neither is present it names
the preferred one, which reports that it needs its local build (`ABSENT`, as before). A forced
`MODEL` whose blob is absent fails with that reason (`MODEL=full, but it is not in the model
store …; not switching`) and never switches. The pick is made at probe and again at every warm,
so a build that finishes later (or a backend fallback, GPU → CPU) changes it on the next warm,
which re-describes the capability (`model_id` changes; a loaded session of the other model is
dropped). The build stamp of every result names the model that answered
(`<pack>/<model>@<version>` and its sha256). The rule is the same on every platform; memory comes
from `oc_system_ram_mb()` (`sysconf` / `sysctl hw.memsize` / `GlobalMemoryStatusEx`).

**The model contract** (pack `io` block, `splat_format: "sharp-projective"`): input `image`
`[1,3,H,W]` float in `[0,1]` (sRGB values; the whole picture resampled to the model's fixed grid,
aspect not kept) and `disparity_factor` `[1]` = focal px / input width; outputs `means`
`[1,N,3]` in the projective frame (`x_ndc·z, y_ndc·z, z`), `scales` `[1,N,3]`, `rotations`
`[1,N,4]` (w x y z, unnormalised), `colors` `[1,N,3]` (linear RGB) and `opacities` `[1,N]`. The
tensor names come from `io.tensors`, the grid from `io.input.resolution`, the count from
`io.output.count`. The model is one `.onnx` file, or a graph plus external-data files that the
graph names by their **sha256** (the store keeps every blob as `blobs/<sha256>`, side by side, so
only that name is found; the pack lists them as `data` files of the same variant).

**What a job does:**

1. **Preprocess** — the RGBA8 / BGRA8 photo is resampled into the grid with a separable
   triangle filter whose support widens with the downscale (an antialiased bilinear, like
   Pillow's `BILINEAR`), so a 6000-px photo does not alias into 1536 px.
2. **Focal** — `params.focal_px` when the job says it is given (`XRT_CONV_JOB_FOCAL_GIVEN`; it
   wins over everything); otherwise the model's own convention, a 30 mm full-frame equivalent
   (`30 · diag / 43.27` px), **labelled `ASSUMED`**. The provider receives decoded pixels and
   never reads EXIF.
3. **Inference** — on the backend [ranked below](#where-splats-mono-runs-ep-ov_device): DirectML,
   OpenVINO GPU, OpenVINO CPU or ONNX Runtime's CPU execution provider. The run is asynchronous
   and polls `is_cancelled` every 100 ms on OpenVINO; an ONNX Runtime run (DirectML, CPU) is one
   blocking call, so a cancel takes effect when it returns.
4. **Postprocess** — unproject through the input's intrinsics, `D = diag(w/2f, h/2f, 1)`:
   `mean' = D·mean`, `cov' = D·R·diag(s²)·Rᵀ·D`, re-diagonalised per Gaussian (the scale
   differs in x and y) into scales and a rotation; linear colours to sRGB degree-0 harmonics,
   opacity to its logit, scales to logs. Written straight into the sink as a binary
   little-endian **3DGS PLY** (`x y z f_dc_0..2 opacity scale_0..2 rot_0..3`, 56 bytes per
   Gaussian), the format every 3DGS viewer and the runtime's spec-3 path read. Any other
   requested format (SOG, say) is refused (`XRT_CONV_REFUSAL_UNSUPPORTED_INPUT`, reason
   `output format not supported: this provider writes 3DGS PLY only; pass an output path ending
   in .ply`; the job fails, the capability stays `READY`).

**Facts, never intent** (`xrt_conv_splat_info` + `xrt_conv_splat_result`, discovery spec §2.6):
OpenCV axes; `metric = 0` (a monocular model's scale through a focal that may be assumed is not
metres) and therefore no baseline, near, far or focus distance in metres; the intrinsics the cloud
was actually unprojected through (`fx = fy = f`, `cx = w/2`, `cy = h/2`, grid = the input's
`w × h`, `GIVEN` or `ASSUMED`); the splat count; and a **build stamp** with every result
(provider version + source revision, `<pack>/<model>@<version>`, the blob's sha256, the backend
and its version, host `LOCAL`). One **suggestion**, labelled `NEAREST_CLUMP`: the
opacity-weighted 10th percentile of depth over the central half of the frame, as a point in the
splat's own units. The provider never writes a camera block; auto-convergence stays runtime
policy.

### Where splats-mono runs (`EP`, `OV_DEVICE`)

`EP=auto` ranks the backends per machine (`src/oc_rank.c`, a pure function unit-tested on
described machines by the `splats-rank` test):

| Machine | Backend | `MODEL=auto` |
|---|---|---|
| Windows, a **discrete non-Intel** GPU (NVIDIA / AMD, ≥ 2 GB dedicated; the one with the most VRAM), ≥ 16 GB RAM, and ≥ 6 GB VRAM with `sharp-fp16-1536` built, or ≥ 9 GB VRAM | **DirectML** on that adapter (`onnxruntime-dml`) | fp16 when built, else full |
| otherwise, OpenVINO's GPU device is an **Intel Arc / Xe-class** GPU (discrete, named Arc, matrix units, GPU IP ≥ 12.55, or an Iris Xe) | OpenVINO GPU (`openvino-gpu`) | full with ≥ 16 GB RAM, else int4 |
| otherwise (an NVIDIA / AMD card that OpenVINO reaches through OpenCL, a UHD-class iGPU, no GPU) | OpenVINO CPU (`openvino-cpu`) | int4 |
| no OpenVINO (macOS) | ONNX Runtime CPU (`onnxruntime-cpu`) | int4 |

- **DirectML serves the full and fp16 models only.** The int4 build's 4-bit weights are ONNX
  Runtime's `MatMulNBits` contrib operator, which is not verified on the DirectML execution
  provider (an operator DirectML does not take would run on the CPU inside the same session, with a
  copy each way). The SHARP pack therefore lists `dml` for `sharp-full-1536` and
  `sharp-fp16-1536` only: `EP=auto` ranks DirectML first only when one of them is what would run
  there, and `MODEL=int4` with `EP=dml` runs the int4 build on the CPU, saying so. The ranking
  runs at probe: build `sharp-fp16-1536` and the next provider start (the service's next
  conversion-host spawn) takes it.
- **OpenVINO's GPU device is decided after OpenVINO loads** (its first warm lists the devices with
  `FULL_DEVICE_NAME`, `DEVICE_TYPE`, `DEVICE_ARCHITECTURE`, `GPU_EXECUTION_UNITS_COUNT`,
  `OPTIMIZATION_CAPABILITIES`). Before that, probe describes `openvino-gpu` provisionally; a GPU
  device the rule rejects moves the capability to OpenVINO's CPU device (re-described, and
  `MODEL=auto` re-picks int4). The EU count only separates an Iris Xe from a UHD part of the
  same generation: newer parts count differently (an Xe3 iGPU reports 32 EUs).
- **Explicit settings win.** `EP=dml` = DirectML on the discrete adapter (else the first hardware
  adapter); `EP=openvino` = OpenVINO, its device by `OV_DEVICE` (and by the Intel rule when
  `auto`); `EP=cpu` = ONNX Runtime's CPU EP. `OV_DEVICE=gpu` forces OpenVINO's (first) GPU device
  of any vendor and `cpu` its CPU device, whenever OpenVINO serves the capability; it does not by
  itself select OpenVINO over DirectML (use `EP=openvino` for that). `MODEL=full|int4` is honoured
  on every backend.
- **Fallbacks:** when the **fp16** model fails on DirectML for a reason of its own (it will not
  load, a wrong output type; not out of memory, not a lost device), only the fp16 model drops out
  for the process: DirectML keeps the full model when the adapter has the 9 GB it needs. Otherwise
  DirectML that cannot start, runs out of memory or fails a run drops out for the process and the
  capability re-ranks without it (OpenVINO, else ONNX Runtime CPU). On Windows the OpenVINO GPU
  device is a candidate only when a DXGI adapter is an Intel GPU that is not UHD / HD Graphics
  class: a machine whose only Intel adapter is a UHD 770 (or that has none) goes straight to
  OpenVINO's CPU device (`MODEL=auto`: int4), never a provisional GPU pick that would compile the
  full model first. OpenVINO GPU → OpenVINO CPU → ONNX Runtime CPU as before. Each step
  re-describes the capability.
- **No silent choice.** One line per ranking on stderr (the runtime's provider log), e.g.

  ```
  [open-conversion] DXGI adapter 1: 'NVIDIA GeForce RTX 3080' vendor 0x10de, 10240 MB dedicated, discrete
  [open-conversion] splats-mono: EP=auto, DirectML on 'NVIDIA GeForce RTX 3080' (discrete, 10240 MB) -> dml; model full (MODEL=auto, 65307 MB RAM)
  [open-conversion] OpenVINO GPU: Intel(R) UHD Graphics 630 (iGPU) (iGPU, vendor 0x8086, IP 9.5, 24 EUs); not used by EP=auto: 'Intel(R) UHD Graphics 630 (iGPU)' is a UHD-class iGPU
  [open-conversion] splats-mono: EP=dml, but this ONNX Runtime has no DirectML (Windows only); using onnxruntime-cpu
  [open-conversion] splats-mono: DirectML unavailable (<the error>)
  ```

  The capability's `backend` (`displayxr-cli conv list`) and every result's build stamp name the
  backend that answered; `luid` is the DirectML adapter's.

### DirectML: measured, and diagnosed

One measurement so far (a desktop RTX 3080, 10 GB, Windows, ONNX Runtime 1.24.4 + DirectML
1.15.4 as shipped): `sharp-full-1536` on a 3840×2160 photo (1 179 648 Gaussians) took **42.3 s**
for the first job and **35.2 s** for the next, and raised the card's dedicated memory by
**6.1 GB** over idle, to 8.9 of 10 GB with a desktop running. The same job runs in 6-13 s
end to end (2.3-2.7 s of it inference) on OpenVINO's GPU device on an Intel Arc iGPU, and the
model's authors report well under a second on a standard GPU. So DirectML was 10-30× slower than
it should be. What the graph rules out: the export has 5 314 nodes, 2 375 of them `Constant` and most of
the rest the traced attention's shape arithmetic (`Shape` → `Slice` → `Cast` → `Sqrt`, per
block); ONNX Runtime folds that at load (basic level: 2 504 nodes left, all standard ONNX
operators: Gemm / MatMul, Softmax, LayerNormalization, Erf, Conv, ConvTranspose,
InstanceNormalization, Resize linear, Slice / Reshape / Transpose / Concat / Split and a few
element-wise ops), every one of which the DirectML EP implements; no `NonZero`, `ScatterND`,
`GatherND`, `TopK`, `GridSample` or `Einsum`, no dynamic shape (the input is fixed), and no
tensor over 1.2 GB (DirectML's limit is 4 GB). So no CPU fallback is expected. The leading
explanation is memory: 6.1 GB on a 10 GB card that a desktop already uses is at or over the
budget Windows gives one process, and what does not fit is kept in system memory, read over
PCIe at a few percent of the card's bandwidth, which is the size of the slowdown seen. Hence
the floors (full fp32 ≥ 9 GB, fp16 ≥ 6 GB) and the fp16 build. Confirmed with the lines below:
the all-fp16 build (export_sharp.py 1.4.0) runs the same photo on the same card in about 1.2 s
of inference at 5.6 GB dedicated, within the budget (about 2 s for the whole job).

Every DirectML splat session and job now logs this process's memory on the adapter (DXGI
`QueryVideoMemoryInfo`: dedicated use and budget, shared use) after the load and before / after
each run, plus a warning when the run ends over the budget or grew shared memory by more than
256 MB:

```
[open-conversion] sharp-full-1536: loaded on DirectML in 5210 ms; GPU memory: 2731 MB dedicated (budget 6900 MB), 210 MB shared
[open-conversion] sharp-full-1536: GPU memory before the run: ...; after: ...
[open-conversion] sharp-full-1536: over the GPU memory budget: Windows keeps part of the model in system memory, which can make a run many times slower (the fp16 model needs half)
```

Setting **`DIAG_PROFILE=1`** adds, for the splat capability on ONNX Runtime (DirectML or CPU):
ONNX Runtime's node placement after its graph optimisation, counted per execution provider and
operator (`... N of M nodes on the CPU EP; DmlExecutionProvider: ... (DmlFusedNode k, MemcpyToHost
1, ...); CPUExecutionProvider: ...`); each job runs the model **twice** and logs both times (the
first pays DirectML's first-run allocation and compilation; `infer` is the second); and for the
first job a profiling JSON (`<scratch dir>/splats-profile_<time>.json`, the path is logged) with a
one-line summary of the second run: both run times and the top-10 operators by time, per
provider. On DirectML a node's time is its CPU-side recording unless the queue is full, so the
GPU's own time collects in the node that waits for it (`MemcpyToHost`): a large `MemcpyToHost`
with every node on DirectML says the GPU itself is slow (memory), CPU-placed nodes with big
times say a fallback.

## OpenVINO (the splat capability's GPU path)

On Linux x86_64 and Windows x64 the provider ships **OpenVINO 2026.4.1** (Apache-2.0) in an
`openvino/` folder next to its binary: the core, the C API, the ONNX and IR front ends, the CPU
and GPU device plug-ins and oneTBB (~140 MB). It is loaded at run time **by absolute path**
through its C API (`openvino/c/openvino.h`; on Linux each library is mapped in dependency order,
because the official libraries carry no `RUNPATH`), never linked, and only on the first warm that
needs it: `probe` checks that the files are there, nothing more.

- **Why the native C API, not ONNX Runtime's OpenVINO execution provider.** OpenVINO reads the
  4-bit ONNX (ONNX Runtime's `MatMulNBits` contrib operator) directly, so no conversion step is
  needed, and its own compiled-model cache (`CACHE_DIR`) turns a ~15 s GPU compile into a
  ~2 s import. The execution provider would add a second inference runtime on top: Microsoft
  does not ship it in the official ONNX Runtime packages (it comes as Intel's
  `onnxruntime-openvino` build, which bundles its own OpenVINO), so it would mean replacing the
  pinned ONNX Runtime with a different, larger build, for no extra coverage. Intel's own
  archives are versioned, hash-pinnable and redistributable (Apache-2.0), like ONNX Runtime's.
- **Device:** `OV_DEVICE` `auto` (default) = the GPU device when it is an Intel Arc / Xe-class GPU
  ([the rule](#where-splats-mono-runs-ep-ov_device)), else its CPU device; `gpu` / `cpu` force
  one. The GPU device drives the system's OpenCL driver (Linux:
  `libOpenCL.so.1` + Intel's compute runtime, e.g. Ubuntu `ocl-icd-libopencl1` +
  `intel-opencl-icd`); without it OpenVINO reports no GPU and the capability runs on its CPU
  device.
- **Fallback, re-described at each step:** OpenVINO GPU → OpenVINO CPU → ONNX Runtime CPU (when
  OpenVINO is missing, too old, or fails to load). A compile that fails moves the capability on
  and retries within the same `warm`; a run that fails after a good warm drops the session, moves
  the capability on and reports `FAILED(DEVICE_LOST, retryable)`, so the runtime re-warms there —
  the same pattern as DirectML and Core ML. Too old = below 2025.0 (`FAILED` is never claimed for
  a missing library: `ABSENT` moves to the next backend).
- **Compiled-model cache:** under the host's scratch directory (`get_path(XRT_CONV_PATH_SCRATCH)`)
  in `openvino-<version>/` (`OV_CACHE=0` disables it). OpenVINO names each blob by a hash of the
  model and the device configuration and recompiles when one does not import, so a blob left by a
  killed process is replaced rather than trusted; the scratch directory may be deleted at any time.
- **macOS** builds carry no OpenVINO: the splat capability runs on ONNX Runtime there.

Measured on a developer laptop with an Intel Panther Lake iGPU (Ubuntu 26.04, 30 GB, OpenVINO
2026.4.1), both SHARP builds, through the provider (fake host, the model store played per
protocol, blobs already in the store), two photos (6016 × 4000 and 4896 × 3264), focal given.
The GPU was otherwise idle: during the GPU runs its render engine was 52 – 78 % busy, all of it the
provider; every other client stayed under 1 %. Each configuration starts with an empty scratch
directory (first photo = first compile, second photo = cached compile). OpenVINO's CPU device runs
with the provider's default 4 threads (`THREADS`).

| OpenVINO | Model | Warm, first compile / cached | First job | Warm job (median) | Inference | Peak RSS, first / cached |
|---|---|---|---|---|---|---|
| GPU | `sharp-full-1536` | 10.6 s / 2.3 s | 2.9 s | 2.8 – 2.9 s | 2.4 s | 0.92 / 0.56 GB |
| GPU | `sharp-int4-1536` | 5.5 s / 1.9 s | 2.9 s | 2.9 – 3.1 s | 2.4 – 2.5 s | 1.63 / 0.60 GB |
| CPU | `sharp-full-1536` | 6.3 s / 3.0 s | 69 – 77 s | 59 – 64 s | 59 – 64 s | 9.8 / 8.8 GB |
| CPU | `sharp-int4-1536` | 4.4 s / 1.9 s | 51 – 53 s | 43 s | 43 s | 5.3 / 4.8 GB |

A job is preprocess + inference + PLY (the PLY write is 0.35 – 0.41 s); every result is
1 179 648 Gaussians, a 66 MB PLY. Blob sizes: full 2.62 GB (6 MB graph + 2.616 GB weights), int4
0.58 GB. On the iGPU the two builds run at the same speed: OpenVINO's GPU device computes in fp16
by default whatever the weights' precision, and the full build compiles with less process memory
(nothing to unpack). On the CPU the full build costs ~40 % more time and ~2× the memory.

Accuracy against the PyTorch fp32 checkpoint (Apple's reference code, same photos, same focal),
with the provider's PLY turned back into the model's projective tensors: depth = median / 95th
percentile relative error of every Gaussian's depth; views = PSNR / SSIM of a splat render from
the input camera and from a camera moved sideways by 10 % of the median depth.

| OpenVINO | Model | Depth, photo 1 | Depth, photo 2 | Input view, photo 1 / 2 | Side view, photo 1 / 2 |
|---|---|---|---|---|---|
| CPU | full | 0.04 % / 0.47 % | 0.05 % / 0.37 % | 53.7 dB 0.999 / 48.4 dB 0.998 | 46.5 dB 0.995 / 45.0 dB 0.994 |
| GPU | full | 0.16 % / 3.1 % | 0.31 % / 2.2 % | 38.0 dB 0.977 / 27.8 dB 0.900 | 38.3 dB 0.977 / 25.6 dB 0.853 |
| CPU | int4 | 1.0 % / 7.0 % | 1.1 % / 4.1 % | 46.3 dB 0.996 / 37.0 dB 0.981 | 29.0 dB 0.860 / 25.5 dB 0.755 |
| GPU | int4 | 1.2 % / 7.6 % | 1.4 % / 4.8 % | 37.7 dB 0.975 / 27.9 dB 0.891 | 28.5 dB 0.847 / 24.3 dB 0.693 |

The full build on the CPU is the reference model; its remaining 0.04 % is the provider's
antialiased resample into the 1536 × 1536 grid, not the model (the export itself matches PyTorch
to better than 1e-4 in depth). The GPU's fp16 compute costs the full build 0.2 – 0.3 % in depth; 4-bit weights cost
~1 %, and most visibly in views away from the input camera (side view SSIM 0.69 – 0.86 against
0.85 – 0.98 for full on the same device). Hence `MODEL=auto`: on a GPU the full build is as fast
and better, so it is the default there; on the CPU the int4 build is the faster, smaller one.

## Getting a splat model (`conv models fetch`, the export script)

Research-only weights are **never bundled, mirrored or pre-fetched** (ADR-057 §5), and the
installers download no model. A SHARP pack's blob is built on the user's machine from Apple's
official checkpoint by `scripts/export_sharp.py` (it downloads the checkpoint, verifies its
SHA-256, exports it, keeps it fp32, 4-bit-quantises it or converts it to fp16, and writes the bytes the pack
pins into the model store);
see that script for its requirements and the licence it asks you to accept. A pack whose model has
a downloadable source is fetched the usual way, `displayxr-cli conv models fetch <pack>
--model <name>` (the same action as the Dashboard's Download button). Either way the runtime
selects the capability only with `AllowResearchOnly` (`displayxr-cli conv run … --allow-research-only`
for one run). As of runtime v2.69.4 the runtime reads `AllowResearchOnly` only from the Windows
registry, so on Linux and macOS its model store refuses a research-only model (`fetch` reports
`policy`), and `conv run` hosts providers without a model store; both are runtime work, tracked
with this capability.

## Build

Plain CMake and a C11 compiler; the build fetches ONNX Runtime (the official NuGet packages:
`Microsoft.ML.OnnxRuntime.DirectML` on Windows, `Microsoft.ML.OnnxRuntime` elsewhere), DirectML
on Windows and OpenVINO on Linux x86_64 / Windows x64 (Intel's release archives) as pinned
archives and checks their SHA-256 before extracting. `-DOPENCONV_OPENVINO=OFF` builds without
OpenVINO.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows, run this from a Visual Studio 2022 developer prompt. Output in `build/`:
`DisplayXR-OpenConversion.dll` (`.so` / `.dylib`), `onnxruntime.dll` (+ `DirectML.dll`;
`libonnxruntime.so.1` / `libonnxruntime.dylib` on POSIX), `openvino/` (Linux x86_64, Windows
x64) and every `packs/*.models.json` — the files that ship together. The provider exports exactly one symbol,
`xrtConversionPluginNegotiate`, links the C runtime statically and imports only `KERNEL32` and
`dxgi` (Windows). `-DOPENCONV_WERROR=ON` turns warnings into errors, as CI does.
`-DOPENCONV_DOWNLOAD_DIR=<dir>` keeps the fetched archives across build trees.

### macOS: Core ML

The pinned `onnxruntime-osx-arm64-1.24.4.tgz` already carries the Core ML execution provider, so
macOS needs no other archive. The provider creates it with the ML Program format, compute units
from `COREML_UNITS` (default CPU + GPU), low-precision GPU accumulation allowed, and keeps the
compiled model under the host's scratch directory (`get_path(XRT_CONV_PATH_SCRATCH)`, in
`coreml-ort-<version>/`) when the host provides one. Core ML partitions the graph: what it cannot
take runs on the CPU execution provider inside the same session. A session that fails to start,
or of which Core ML takes no node, moves the capability to the CPU execution provider and
re-describes it; a Core ML run that fails drops the session, moves the capability to the CPU and
reports `FAILED(DEVICE_LOST, retryable)`, so the runtime re-warms it on the CPU. The temporal
cache works unchanged: Core ML reads and writes CPU memory, so the cache is bound there.

Which execution providers a blob is acceptable on is the pack's call (`variants[].ep`): the
default pack lists `coreml` for the two video-depth models only. Measured on a developer M1 Pro
(16 GB, macOS 26.3), `--bench 100` after four checked frames, latency of one `stream_convert`
(min / median), **on a shared machine under heavy unrelated load**: treat
the minimum as the better estimate of a quiet machine, and the ratios rather than the absolutes
(the experiment column ran at load 15 to 30).

| Model | CPU EP (4 threads) | Core ML EP (CPU + GPU) | Core ML, attention `Gather` rewritten (experiment) |
|---|---|---|---|
| `vda-small-stream-518x294` | 384 / 760 ms | 141 / 266 ms | 79 / 95 ms (10.5 fps) |
| `vda-small-stream-364x210` | 178 / 296 ms | 71 / 89 ms | 41 / 46 ms (21.8 fps) |
| `da2-small` (518 × 294 grid) | 320 / 578 ms | 374 / 431 ms | — |

| Model | CPU EP warm | Core ML warm, cached | Core ML warm, first (compile) | Peak RSS CPU / Core ML |
|---|---|---|---|---|
| `vda-small-stream-518x294` | 2.4 s | 10 – 17 s | 86 – 93 s (cache 415 MB) | 2.4 / 2.5 GB |
| `vda-small-stream-364x210` | 0.5 s | 11 – 12 s | 62 – 67 s | 1.3 / 1.5 GB |
| `da2-small` | 0.5 s | 5 – 16 s | 11 s | 0.5 / 2.6 GB |

- **Why not the 20 – 40 ms class with the shipped blobs:** Core ML takes 783 of the video model's
  827 nodes but in 13 partitions. Each ViT block splits q / k / v with a `Gather` on a scalar
  index, which the Core ML EP rejects ("Gather does not support scalar 'indices'"), so every
  block returns to the CPU. Rewriting those 37 nodes as `Slice` + `Squeeze` (same maths) gives
  one partition (869 of 876 nodes) and the third column — see
  [the follow-up](#follow-up-core-ml-friendly-video-depth-variants). The rest of a frame is the
  160 MB (518 px) temporal cache Core ML copies in and out of CPU memory.
- **The still-depth model stays on the CPU:** Core ML takes 719 of 836 nodes in 60 partitions
  (scalar `Gather`, dynamic `Reshape` and `Slice`, bicubic `Resize` are refused), it is no faster
  than the CPU EP, and it costs five times the memory.
- **Output difference vs the CPU EP** (four frames each, raw R32F, resampled to the request):
  video models max |Δ| 1.2e-5 (518 px) and 5.2e-5 (364 px) on a range of about 12, median
  ≤ 1.2e-6 — the GPU path computes in fp32, so there is no fp16 drift. The fp16 still-depth model
  on Core ML: max |Δ| 0.064 (0.7 % of range), median 0.0014 (0.015 %).
- **Not measured:** the Neural Engine (`COREML_UNITS=all` / `ane`). On the measuring machine the
  system's Neural Engine compiler service was busy with other work for hours and the first
  session never finished loading; such a load blocks `warm` and cannot be cancelled, which is why
  the default is CPU + GPU.
- **Without a scratch directory** every warm is a first warm (ONNX Runtime converts into a
  temporary directory and deletes it with the session). The runtime's hosts do not provide one
  yet: [displayxr-runtime#2194](https://github.com/DisplayXR/displayxr-runtime/issues/2194).
- **The cache is a transaction**, per the scratch-directory rule: a cold warm compiles into
  `coreml-ort-<version>/.tmp-<blob>-<pid>/`; once the warm-up inference succeeds a `.complete`
  stamp is written and the directory is renamed to `coreml-ort-<version>/<blob>/`. Only a stamped
  cache is loaded; one that fails to load is deleted and compiled afresh (the capability stays on
  Core ML), and temp directories of processes that died mid-compile are swept. Core ML keys its
  own device compile by path, so the first warm after a publish compiles once more; later warms
  take the cached time.

#### Follow-up: Core ML-friendly video-depth variants

Prepared, **not published**: `scripts/rewrite_video_depth_coreml.py` rebuilds the two
video-depth blobs with every scalar-index `Gather` replaced by `Slice` + `Squeeze`. It reads the
upstream blobs by sha256 from the pack, refuses any other bytes, changes nothing else (weights,
tensor names, shapes, the temporal cache) and with `--check` proves on the CPU execution provider
that the outputs are identical (bit-exact on both models, first and continued frame). The output
is deterministic for one `onnx` version; with onnx 1.19.1:

```
python3 scripts/fetch_models.py packs/dxr-open-default.models.json models-cache --models vda-small-stream-518x294,vda-small-stream-364x210
python3 scripts/rewrite_video_depth_coreml.py packs/dxr-open-default.models.json models-cache rewritten --check
```

| New model | From | Blob sha256 | Size |
|---|---|---|---|
| `vda-small-stream-coreml-518x294` 1.0.0 | `vda-small-stream-518x294` | `db51cb27484fcd302eba5ccc901c98a3fa7204b7a5b9979dafe839dd111386e7` | 57 819 066 |
| `vda-small-stream-coreml-364x210` 1.0.0 | `vda-small-stream-364x210` | `c76b9097897e9efc5021ff5010cc0292a10d0495eef89170a5a16061a7405b3b` | 57 521 756 |

Once the blobs are on the models repository's `blobs` release, the pack gains two entries, each a
copy of its source model with the new `name`, `version` 1.0.0, `provenance.converted_by` naming
the rewrite, and one variant `w16` with `ep` `["dml", "coreml", "cpu"]` and the blob above; and
`defaults.depth-video` points at them (the graph is the same maths, so every execution provider
can run it). The old entries stay, with `coreml` removed from their `ep`, so an older store keeps
working. Run `check_model_pack.py --verify-blobs DIR` before merging.

A further step, not prepared: the temporal cache is fp32 (`cache_dtype float32`, 160 MB at
518 px) and Core ML copies it in and out of CPU memory every frame. An export with an fp16 cache
halves that copy and the memory. It needs a new export (not a graph rewrite), the pack's
`cache_dtype` set to `float16`, and the provider's temporal run taught to create and bind
`FLOAT16` cache tensors (today it assumes `FLOAT`); its drift has to be measured over long
streams, since the error feeds back frame to frame.

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
| `default-ep` (`-DOPENCONV_TEST_GPU=ON`) | both capabilities on the default execution provider (DirectML on Windows, Core ML on macOS) | yes |
| `splats-tiny-model` | writes a tiny synthetic splat model with the real I/O contract, its pack and the expected output (`tests/make_tiny_splat_model.py`; needs the `onnx` Python package, skipped without it) | no |
| `splats-cpu` | the splat lane on ONNX Runtime's CPU EP: download → `READY`, three jobs (focal given: every sampled Gaussian equals an independent numpy unprojection; no focal: `ASSUMED` 30 mm equivalent; a SOG request: refused, the cap stays `READY`), PLY header + count, facts, build stamp | no |
| `splats-openvino-cpu` | the same on OpenVINO's CPU device (no GPU needed), backend `openvino-cpu` | no |
| `splats-rank` | the splat backend ranking on described machines (Intel UHD + RTX 3080 Laptop; RTX 3080 desktop, no iGPU; Intel Arc iGPU only; AMD APU + discrete Radeon; a 4 GB card; an 8 GB card; a 10 GB card reporting 10 240 and 10 067 MB; no GPU), with and without an fp16 model in the pack / in the store, and every explicit setting: backend, adapter, model, fallback line | no |
| `diag-profile-parse` | the `DIAG_PROFILE` parsers on text in ONNX Runtime's formats: the node placement report (per EP and operator, fused partitions counted as one operator) and the profiling JSON (the last `model_run`'s kernel times by operator and provider) | no |
| `splats-diag-profile` | `DIAG_PROFILE=1` on the CPU EP: the placement line, two runs per job and the profile summary line, end to end | no |
| `splats-pick-auto-cpu` | a three-model pack (`tiny-full` = high, `tiny-int4` = low, `tiny-fp16` = medium with a DirectML variant only) on the CPU: `MODEL=auto` picks `tiny-int4`, never `tiny-fp16` | no |
| `splats-pick-full-external-data` | `MODEL=full`: `tiny-full`, stored as a graph + an external-data file named by its sha256, downloads both files and runs (the full SHARP layout) | no |
| `splats-pick-fallback` | the store holds only `tiny-full`, downloads off: `auto` (int4 on a CPU) takes `tiny-full` and says so | no |
| `splats-pick-slow-verify` | the same store, re-verified by the host for 7 s (longer than the 5 s request grace; the runtime's store re-hashes a stored blob once per process): the provider waits through `WARMING`, then runs | no |
| `splats-pick-fp16-off-dml` | the pack also names `tiny-fp16` (`defaults.medium`, a DirectML variant only, the SHARP fp16 shape): `MODEL=fp16` off DirectML takes `tiny-full` | no |
| `splats-pick-forced-absent` | the same store, `MODEL=int4`: `ABSENT` with the reason, never switched to `tiny-full` | no |
| `splats-pick-full-openvino-cpu` | `MODEL=full` on OpenVINO's CPU device: OpenVINO finds the external data the same way | no |
| `splats-default-ep` (`-DOPENCONV_TEST_GPU=ON -DOPENCONV_TEST_SPLAT_MODEL=<.onnx> [-DOPENCONV_TEST_SPLAT_IMAGE=<.ppm>]`) | a real splat model you built yourself, on the default backend (OpenVINO GPU where there is one); nothing is fetched | yes (yours) |

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

For measurements the fake host takes `--scratch DIR` (served as the host's scratch directory),
`--bench N` (N more timed frames per stream after the checked ones: min / median / p90 / mean /
max) and `--dump DIR` (every checked depth map as raw R32F, to diff two execution providers).

## Settings

Read with `host->get_setting(KEY)`: `DXR_CONV_OPEN_CONVERSION_<KEY>` in the hosting process's
environment, else `Settings\<KEY>` under the registry row (Windows) or the manifest's
`settings` object (POSIX).

| Key | Values | Effect |
|---|---|---|
| `ENABLE` | `0` | `probe` declines |
| `QUALITY` | `low`, `medium` (default), `high` | which model `defaults[task]` names for the depth capabilities (low = the 364-px video model); the splat capability uses `MODEL` instead |
| `EP` | `auto` (default), `cpu`, `dml`, `coreml`, `openvino` | `cpu` forces the CPU execution provider; for depth `auto` = DirectML on Windows, Core ML on macOS (each falls back to the CPU); for the splat capability `auto` = DirectML on a discrete non-Intel GPU, else OpenVINO's Intel Arc / Xe GPU, else OpenVINO's CPU device, else ONNX Runtime's CPU EP ([details](#where-splats-mono-runs-ep-ov_device)); `openvino` forces OpenVINO for it (the depth capabilities never use OpenVINO) |
| `OV_DEVICE` | `auto` (default), `gpu`, `cpu` | OpenVINO's device whenever OpenVINO serves the splat capability (`auto` = an Intel Arc / Xe-class GPU, never an NVIDIA / AMD / UHD-class one, else the CPU; `gpu` = its GPU device, any vendor); it does not select OpenVINO over DirectML |
| `OV_CACHE` | `0` | no OpenVINO compiled-model cache, even when the host provides a scratch directory |
| `MODEL` | `auto` (default), `full`, `fp16`, `int4` | the splat model when a pack names two or three: `full` = `defaults["splats-mono"].high`, `int4` = `.low`, `fp16` = `.medium` on DirectML (elsewhere it means full); `auto` = on DirectML fp16 when built (≥ 6 GB VRAM), else full (≥ 10 GB VRAM); full on another GPU backend; ≥ 16 GB of memory for either; else int4; among the models in the store ([details](#which-sharp-build-model)) |
| `OPENVINO` | absolute path | another OpenVINO runtime folder (default: `openvino/` next to the binary) |
| `COREML_UNITS` | `gpu` (default), `all`, `ane`, `cpu` | macOS: the Core ML compute units (CPU + GPU, + the Neural Engine, CPU + Neural Engine, CPU only) |
| `COREML_CACHE` | `0` | macOS: no compiled-model cache, even when the host provides a scratch directory |
| `ORT_LOG` | `warning` (default), `info`, `verbose` | how much of ONNX Runtime's log is printed (`verbose` shows the Core ML node assignment) |
| `THREADS` | integer (default 4) | CPU intra-op threads; 0 = ONNX Runtime's default (every core) |
| `GRAPH_OPT` | `basic`, `extended` (default), `all` | ONNX Runtime graph optimisation (`all` fails to load the fp16 still-depth model on the CPU execution provider of 1.24.4) |
| `DIAG_PROFILE` | `1` | splat diagnostics on ONNX Runtime: node placement per execution provider, two runs per job with both times, and the first job's profile JSON in the scratch directory with a top-10 operator line ([details](#directml-measured-and-diagnosed)) |
| `PACK` | absolute paths, `;`-separated | the pack manifests (default: `dxr-open-default.models.json` and every other `*.models.json` next to the binary) |
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
  removes only this provider's row, its allowlist row and files, never the model store. It does
  **not** stop or start `displayxr-service`, by the runtime's installer contract (discovery spec
  §3.8): the provider runs in its own `displayxr-conversion-host`, which the service respawns from
  the new binary once the row's `Version` changes and the maintenance mutex is released
  (§3.8.1). A host still mapping the old DLL makes the file copy fail, which aborts the install
  (`AllowSkipFiles off`); closing it with Restart Manager first, as the Leia conversion
  installer does, is not done yet. If `displayxr-cli conv list` still shows the old version
  after an install, restarting the service (non-elevated: `explorer.exe "<runtime>\displayxr-service.exe"`)
  is the workaround; please report it with the service log.
- `displayxr-open-conversion-<version>-linux-x86_64.tar.gz` (CPU, + OpenVINO for the splat
  capability): `install.sh` copies the payload (including `openvino/`) to
  `${XDG_DATA_HOME:-~/.local/share}/DisplayXR/OpenConversion` and writes the JSON manifest, whose
  `packs` lists every pack shipped. OpenVINO's runtime is bundled (a pinned, hash-checked Intel
  archive built on Ubuntu 22.04, so it runs on 22.04 / 24.04 / 26.04), not taken from a distro
  or Intel apt package: neither Ubuntu nor Debian packages OpenVINO, and Intel's apt repository
  would make a user-level tarball need root and a third-party repository. What stays a system
  dependency is the GPU driver, which nothing should bundle: `install.sh` prints a note when no
  OpenCL loader is present (the capability then runs on OpenVINO's CPU device). `uninstall.sh`
  removes both.
- Neither package carries a splat model's weights; the Windows installer lists the splat pack in
  `Packs` when the build ships it.

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

`include/xrt/` is vendored **verbatim** from the runtime's `v2.69.4` release
(`src/xrt/include/xrt/xrt_conversion_plugin.h`, `xrt_component_settings.h`,
`xrt_component_license.h`, `xrt_results.h`, `xrt_conv_wire.h`): conversion-provider ABI major
`XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT` = **1**, with the ADR-057 splat result
(`xrt_conv_splat_result`, `XRT_CONVERSION_PLUGIN_HAS_SPLAT_RESULT`) and `describe_settings`
(`displayxr-cli conv options open-conversion` lists every key above). The provider uses
`xrt_conversion_plugin.h` (with the two component headers it includes) and `xrt_results.h`;
`xrt_conv_wire.h` is the runtime's IPC wire, vendored for reference and compile-checked only.
Update by copying from a runtime release.

## Licence

[BSL-1.0](LICENSE), like the DisplayXR runtime. Third-party components and model licences:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
