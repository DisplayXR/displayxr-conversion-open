// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Where the splat capability runs: the backend ranking as a pure
 *         function of the machine's devices and the settings (unit-tested
 *         in tests/rank_test.c, no GPU needed).
 *
 * EP=auto, in order:
 *   1. ONNX Runtime + DirectML on a DISCRETE, non-Intel adapter (Windows:
 *      the one with the most dedicated VRAM) - when the pack's full model has
 *      a DirectML variant and the full model would be picked there (RAM and
 *      VRAM floors, MODEL not int4). DirectML serves the full model only: the
 *      int4 model's MatMulNBits operator is not known to run on DirectML.
 *   2. OpenVINO's GPU device, only when it is an Intel Arc / Xe-class GPU
 *      (never an NVIDIA / AMD device reached through OpenCL, never a UHD-class
 *      iGPU). Before OpenVINO is loaded its devices are unknown: the choice is
 *      then provisional and re-ranked once warm has listed them.
 *   3. OpenVINO's CPU device.
 *   4. ONNX Runtime's CPU execution provider.
 * EP=dml | openvino | cpu | coreml, OV_DEVICE=gpu | cpu and MODEL=full | int4
 * are explicit and win; OV_DEVICE picks OpenVINO's device whenever OpenVINO
 * serves the capability, it does not by itself select OpenVINO over DirectML.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OC_RANK_MAX_ADAPTERS 8u
#define OC_RANK_MAX_OV_GPUS 4u

#define OC_VENDOR_INTEL 0x8086u
#define OC_VENDOR_NVIDIA 0x10deu
#define OC_VENDOR_AMD 0x1002u
#define OC_VENDOR_MICROSOFT 0x1414u //!< the Basic Render Driver (WARP)

/*! MODEL=auto picks the full splat model only with at least this much RAM: "16 GB" as
 *  the OS reports it (firmware and an integrated GPU reserve some of it, so a 16 GB
 *  machine reports a little under 16 GiB). */
#define OC_FULL_MODEL_MIN_RAM_MB (15ull * 1024u)
/*! ... and, on DirectML, at least this much dedicated VRAM: the fp32 weights are
 *  ~2.6 GB, plus the activations at 1536x1536. */
#define OC_FULL_MODEL_MIN_VRAM_MB (6ull * 1024u)
/*! A DXGI adapter with less dedicated memory than this is an integrated one (an
 *  APU's carve-out), not a discrete card. */
#define OC_DISCRETE_MIN_VRAM_MB 2048ull
/*! A Gen12 Xe-LP Intel GPU (architecture 12.0) with at least this many execution
 *  units is an Iris Xe (80-96); the UHD 7xx parts of the same generation have 16-32.
 *  Newer parts count differently (an Xe3 iGPU reports 32), so the EU count is only
 *  used for Xe-LP. */
#define OC_XE_MIN_EUS 80u
/*! Intel GPU IP 12.55 (Xe-HPG, the first Arc) and newer (Xe-LPG, Xe2, Xe3) is Arc-class. */
#define OC_ARC_MIN_ARCH_MAJOR 12u
#define OC_ARC_MIN_ARCH_MINOR 55u

/*! One DXGI adapter (Windows). */
struct oc_dxgi_adapter
{
	char name[128]; //!< DXGI_ADAPTER_DESC1.Description, UTF-8
	uint32_t vendor_id;
	uint64_t dedicated_mb; //!< DedicatedVideoMemory
	bool software;         //!< DXGI_ADAPTER_FLAG_SOFTWARE
	int index;             //!< the DXGI enumeration index (the DirectML device id)
	uint64_t luid;
};

enum oc_ov_gpu_type
{
	OC_OV_GPU_TYPE_UNKNOWN = 0,
	OC_OV_GPU_TYPE_INTEGRATED,
	OC_OV_GPU_TYPE_DISCRETE,
};

/*! One OpenVINO GPU device. */
struct oc_ov_gpu
{
	char device[16]; //!< "GPU", "GPU.0", "GPU.1"
	char name[128];  //!< FULL_DEVICE_NAME
	enum oc_ov_gpu_type type;
	uint32_t eu_count;   //!< GPU_EXECUTION_UNITS_COUNT; 0 = not reported
	uint32_t vendor_id;  //!< from DEVICE_ARCHITECTURE ("GPU: vendor=0x8086 arch=v30.1.0"); 0 = not reported
	uint32_t arch_major; //!< the GPU IP version from DEVICE_ARCHITECTURE; 0 = not reported
	uint32_t arch_minor;
	bool hw_matmul; //!< OPTIMIZATION_CAPABILITIES has GPU_HW_MATMUL (systolic / XMX units)
};

/*! Parse OpenVINO's GPU DEVICE_ARCHITECTURE ("GPU: vendor=0x8086 arch=v30.1.0") into
 *  @p g's vendor_id / arch_major / arch_minor; fields not found stay 0. */
void
oc_rank_parse_ov_arch(const char *arch, struct oc_ov_gpu *g);

enum oc_splat_backend
{
	OC_SB_ORT_CPU = 0,
	OC_SB_DML,
	OC_SB_OV_GPU,
	OC_SB_OV_CPU,
	OC_SB_COREML,
};

/*! Everything the ranking looks at. */
struct oc_splat_env
{
	const char *ep;        //!< setting EP: "auto" | "cpu" | "dml" | "coreml" | "openvino"
	const char *ov_device; //!< setting OV_DEVICE: "auto" | "gpu" | "cpu"
	const char *model;     //!< setting MODEL: "auto" | "full" | "int4"
	uint64_t ram_mb;       //!< 0 = unknown

	bool dml_available; //!< ONNX Runtime carries the DirectML EP (Windows)
	bool coreml_available;
	const struct oc_dxgi_adapter *adapters;
	uint32_t adapter_count;
	bool full_on_dml; //!< the pack's full splat model has a DirectML variant

	bool ov_present; //!< OpenVINO located (probe) / loaded (warm)
	bool ov_known;   //!< ov_gpus is the real device list (OpenVINO loaded)
	const struct oc_ov_gpu *ov_gpus;
	uint32_t ov_gpu_count;
};

struct oc_splat_choice
{
	enum oc_splat_backend backend;
	int adapter;        //!< OC_SB_DML: index into env->adapters
	int ov_gpu;         //!< OC_SB_OV_GPU: index into env->ov_gpus; -1 = OpenVINO's default "GPU"
	bool provisional;   //!< OC_SB_OV_GPU picked before OpenVINO listed its devices
	bool full;          //!< the model MODEL resolves to on this backend (true = full, false = int4)
	char why[256];      //!< why this backend, for the log line
	char fallback[200]; //!< non-empty: the requested backend is not available, and what replaced it
};

/*! The adapter is a discrete, non-Intel GPU (NVIDIA, AMD, ...): not software, not
 *  Microsoft's, not Intel, at least OC_DISCRETE_MIN_VRAM_MB of dedicated memory. */
bool
oc_rank_dxgi_discrete(const struct oc_dxgi_adapter *a);

/*! The OpenVINO GPU device is one OpenVINO's GPU plug-in is worth using on: an
 *  Intel GPU (by name, and by vendor id when reported) not named UHD / HD Graphics
 *  that is discrete, named Arc, has matrix units (GPU_HW_MATMUL), is GPU IP 12.55 or
 *  newer, or is a Gen12 Xe-LP part named Iris Xe / with >= OC_XE_MIN_EUS EUs.
 *  @p why (may be NULL) gets the reason it is not. */
bool
oc_rank_ov_gpu_ok(const struct oc_ov_gpu *g, char *why, size_t why_size);

/*! MODEL=auto: full on a GPU backend with enough RAM (and, on DirectML, enough
 *  VRAM: @p dml_vram_mb, ignored when @p dml is false); int4 otherwise. */
bool
oc_rank_auto_full(bool gpu, uint64_t ram_mb, bool dml, uint64_t dml_vram_mb);

/*! The ranking. Never fails: the last resort is ONNX Runtime's CPU EP. */
void
oc_rank_splat(const struct oc_splat_env *env, struct oc_splat_choice *out);

/*! "dml", "openvino-gpu", ... for the log line. */
const char *
oc_rank_backend_name(enum oc_splat_backend b);

#ifdef __cplusplus
}
#endif
