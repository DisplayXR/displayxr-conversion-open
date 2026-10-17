# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this repo is

The **DisplayXR open-model conversion provider** (`DisplayXR-OpenConversion`, id
`open-conversion`, ProbeOrder 200): the conversion provider every DisplayXR box gets without a
vendor module (ADR-053, runtime roadmap `docs/roadmap/conversion-providers.md` §9 P4). Two
depth capabilities — `DEPTH_VIDEO` (temporal, a small streaming video-depth model) and
`DEPTH_MONO` (a small still-depth model) — run through ONNX Runtime (DirectML on Windows, Core ML
on macOS for the models whose pack variant lists `coreml`, CPU everywhere). The runtime composes
views from them; this repo never declares `VIEWS_*`.

**Public repository.** No internal data anywhere (code, comments, commit messages, docs): no
dataset names or paths, no internal measurements, no vendor-module internals, no product or
title names. Models are named by their upstream project names only.

**Open only.** No proprietary or private dependency at build time. ONNX Runtime (MIT) and the
DirectML redistributable are fetched as pinned, hash-checked release archives; nothing is
committed. Never commit a model blob.

## Layout

- `src/open_conversion.c` — negotiate + the vtable: probe (settings, ONNX Runtime, the pack,
  the execution provider), describe, readiness, warm (store resolve / request → session →
  warm-up inference, DirectML / Core ML → CPU fallback), the stream lane and the `DEPTH_MONO` job lane.
- `src/oc_ort.{c,h}` — ONNX Runtime loaded at run time by absolute path (`OrtGetApiBase` only),
  sessions, the single-image run and the temporal run (IoBinding; the cache stays on device).
- `src/oc_pack.{c,h}`, `src/oc_json.{c,h}` — the schema-2 pack reader and a strict JSON parser.
- `src/oc_image.{c,h}` — resampling, normalisation, range statistics.
- `src/oc_platform.{c,h}` — mutex, sleep, clock, module directory, library loading.
- `tools/fakehost.c` (+ `tools/sha256.c`) — the fake host: drives the provider and plays the
  model store (file:// downloads, `.part` → verify → rename → `.ok` stamp last). CTest cases in
  `CMakeLists.txt`.
- `packs/dxr-open-default.models.json` — the default pack. `tests/packs/` — test packs.
- `scripts/check_model_pack.py` — pack validator (also the CI guard that no `prefetch` blob is
  non-commercial); `scripts/fetch_models.py` — dev / CI blob download with hash check;
  `scripts/rewrite_video_depth_coreml.py` — rebuilds the video-depth blobs for Core ML (scalar
  `Gather` → `Slice` + `Squeeze`), from the pack's blobs by sha256 (needs `onnx`).
- `cmake/OnnxRuntime.cmake` — the pinned archives and their SHA-256.
- `include/xrt/` — vendored **verbatim** from the runtime (tag in `README.md`). Never edit.

## The contract (source of truth: the runtime repository)

- `docs/specs/runtime/conversion-provider-discovery.md` — §2 ABI + threading + hosting rules,
  §3 discovery, §4.3 readiness, §5 model packs.
- `docs/guides/vendor-plugin-onboarding.md` §9 and the conversion-provider template repo.

Rules when touching provider code:
- Read and write a top-level struct only within its `struct_size` (`SET` / `COVERS`); embedded
  structs are frozen.
- The [C] slots (`get_provider_status`, `describe_cap`, `get_generation`, `get_state`,
  `stream_get_output_desc`) answer from memory under `lock`; never load or touch I/O there.
- Blob bytes only through `host->model_resolve` / `host->model_request`. The provider never
  downloads, never writes the store, never opens a path it did not get from the host.
- Configuration only via `host->get_setting`; never `getenv`. No windows, no process-state
  changes, no network.
- Honest units: affine disparity is reported as such, with the per-frame range; never invent
  intrinsics, a scale, a near / far or a baseline.
- Version gate: ONNX Runtime missing → `ABSENT`; present but older than the compiled API →
  `FAILED(VERSION)`, never `FAILED(MODEL)`.
- A session never outlives the streams' temporal bindings: `release_session()` drops them first.

## Building (Windows: from a VS 2022 developer prompt)

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOPENCONV_WERROR=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

`-DOPENCONV_TEST_MODELS_DIR=<dir>` (blobs as `<sha256>.onnx`, from `scripts/fetch_models.py`)
enables the inference tests; `-DOPENCONV_TEST_GPU=ON` adds the DirectML run.

## Conventions

- License every new file `BSL-1.0` with a `The DisplayXR Project` copyright header.
- Runtime style: tabs, 110 columns, `snake_case`, the `oc_` prefix.
- Bumping ONNX Runtime: change the version and every SHA-256 in `cmake/OnnxRuntime.cmake`
  (download each archive once and hash it), rebuild, run every test including the DirectML one.
- Changing a model: new `version` and blob; never re-point `name@version`. Re-run
  `scripts/check_model_pack.py --verify-blobs`.
- Commits: conventional-commit subjects.
