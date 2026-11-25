// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  The splat capability's backend ranking, see oc_rank.h.
 */

#include "oc_rank.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool
contains_ci(const char *hay, const char *needle)
{
	const size_t n = strlen(needle);
	for (const char *h = hay; *h != '\0'; h++) {
		size_t i = 0;
		while (i < n && h[i] != '\0' && tolower((unsigned char)h[i]) == tolower((unsigned char)needle[i])) {
			i++;
		}
		if (i == n) {
			return true;
		}
	}
	return false;
}

static bool
is(const char *setting, const char *value)
{
	return setting != NULL && strcmp(setting, value) == 0;
}

const char *
oc_rank_backend_name(enum oc_splat_backend b)
{
	switch (b) {
	case OC_SB_DML: return "dml";
	case OC_SB_OV_GPU: return "openvino-gpu";
	case OC_SB_OV_CPU: return "openvino-cpu";
	case OC_SB_COREML: return "coreml";
	default: return "onnxruntime-cpu";
	}
}

bool
oc_rank_dxgi_discrete(const struct oc_dxgi_adapter *a)
{
	return a != NULL && !a->software && a->vendor_id != OC_VENDOR_INTEL && a->vendor_id != OC_VENDOR_MICROSOFT &&
	       a->dedicated_mb >= OC_DISCRETE_MIN_VRAM_MB;
}

void
oc_rank_parse_ov_arch(const char *arch, struct oc_ov_gpu *g)
{
	if (arch == NULL) {
		return;
	}
	const char *v = strstr(arch, "vendor=");
	if (v != NULL) {
		g->vendor_id = (uint32_t)strtoul(v + 7, NULL, 0);
	}
	const char *a = strstr(arch, "arch=");
	if (a != NULL) {
		a += 5;
		if (*a == 'v' || *a == 'V') {
			a++;
		}
		char *end = NULL;
		g->arch_major = (uint32_t)strtoul(a, &end, 10);
		if (end != NULL && *end == '.') {
			g->arch_minor = (uint32_t)strtoul(end + 1, NULL, 10);
		}
	}
}

bool
oc_rank_ov_gpu_ok(const struct oc_ov_gpu *g, char *why, size_t why_size)
{
	char dummy[8];
	if (why == NULL) {
		why = dummy;
		why_size = sizeof(dummy);
	}
	why[0] = '\0';
	if (g == NULL) {
		snprintf(why, why_size, "no OpenVINO GPU device");
		return false;
	}
	if (!contains_ci(g->name, "intel") || (g->vendor_id != 0 && g->vendor_id != OC_VENDOR_INTEL)) {
		// OpenVINO's GPU plug-in reaching an NVIDIA / AMD card through OpenCL: measured
		// slower than the CPU.
		snprintf(why, why_size, "'%.80s' is not an Intel GPU", g->name);
		return false;
	}
	if (contains_ci(g->name, "UHD") || contains_ci(g->name, "HD Graphics")) {
		snprintf(why, why_size, "'%.80s' is a UHD-class iGPU", g->name);
		return false;
	}
	const bool arc_ip = g->arch_major > OC_ARC_MIN_ARCH_MAJOR ||
	                    (g->arch_major == OC_ARC_MIN_ARCH_MAJOR && g->arch_minor >= OC_ARC_MIN_ARCH_MINOR);
	const bool xe_lp = g->arch_major == 12u && !arc_ip;
	if (g->type == OC_OV_GPU_TYPE_DISCRETE || contains_ci(g->name, "Arc") || g->hw_matmul || arc_ip ||
	    contains_ci(g->name, "Iris Xe") || (xe_lp && g->eu_count >= OC_XE_MIN_EUS)) {
		return true;
	}
	snprintf(why, why_size, "'%.80s' is not an Arc / Xe-class GPU (IP %u.%u, %u EUs)", g->name, g->arch_major,
	         g->arch_minor, g->eu_count);
	return false;
}

bool
oc_rank_dxgi_rules_out_ov_gpu(const struct oc_dxgi_adapter *adapters, uint32_t count, char *why, size_t why_size)
{
	char dummy[8];
	if (why == NULL) {
		why = dummy;
		why_size = sizeof(dummy);
	}
	why[0] = '\0';
	if (adapters == NULL || count == 0) {
		return false;
	}
	for (uint32_t i = 0; i < count; i++) {
		const struct oc_dxgi_adapter *a = &adapters[i];
		if (!a->software && a->vendor_id == OC_VENDOR_INTEL && !contains_ci(a->name, "UHD") &&
		    !contains_ci(a->name, "HD Graphics")) {
			return false;
		}
	}
	snprintf(why, why_size, "no Intel Arc / Xe-class DXGI adapter, so no OpenVINO GPU device worth using");
	return true;
}

bool
oc_rank_auto_full(bool gpu, uint64_t ram_mb, bool dml, uint64_t dml_vram_mb)
{
	if (!gpu || ram_mb < OC_FULL_MODEL_MIN_RAM_MB) {
		return false;
	}
	return !dml || dml_vram_mb >= OC_FULL_MODEL_MIN_VRAM_MB;
}

enum oc_splat_model
oc_rank_auto_model(bool gpu, uint64_t ram_mb, bool dml, uint64_t dml_vram_mb, bool fp16_ready)
{
	if (!gpu || ram_mb < OC_FULL_MODEL_MIN_RAM_MB) {
		return OC_SM_INT4;
	}
	if (!dml) {
		return OC_SM_FULL;
	}
	if (fp16_ready && dml_vram_mb >= OC_FP16_MODEL_MIN_VRAM_MB) {
		return OC_SM_FP16;
	}
	return dml_vram_mb >= OC_FULL_MODEL_MIN_VRAM_MB ? OC_SM_FULL : OC_SM_INT4;
}

const char *
oc_rank_model_name(enum oc_splat_model m)
{
	switch (m) {
	case OC_SM_FULL: return "full";
	case OC_SM_FP16: return "fp16";
	default: return "int4";
	}
}

//! The discrete adapter with the most dedicated memory; -1 = none.
static int
best_discrete(const struct oc_splat_env *e)
{
	int best = -1;
	for (uint32_t i = 0; i < e->adapter_count && e->adapters != NULL; i++) {
		if (oc_rank_dxgi_discrete(&e->adapters[i]) &&
		    (best < 0 || e->adapters[i].dedicated_mb > e->adapters[best].dedicated_mb)) {
			best = (int)i;
		}
	}
	return best;
}

//! The first hardware adapter (EP=dml with no discrete one); -1 = none.
static int
first_hardware(const struct oc_splat_env *e)
{
	for (uint32_t i = 0; i < e->adapter_count && e->adapters != NULL; i++) {
		if (!e->adapters[i].software && e->adapters[i].vendor_id != OC_VENDOR_MICROSOFT) {
			return (int)i;
		}
	}
	return -1;
}

static void
dml_on(const struct oc_splat_env *e, struct oc_splat_choice *o, int a, const char *lead)
{
	const struct oc_dxgi_adapter *ad = &e->adapters[a];
	o->backend = OC_SB_DML;
	o->adapter = a;
	snprintf(o->why, sizeof(o->why), "%s, DirectML on '%.80s' (%s, %llu MB)", lead, ad->name,
	         oc_rank_dxgi_discrete(ad) ? "discrete" : "not discrete", (unsigned long long)ad->dedicated_mb);
}

/*! OpenVINO serves the capability: which device. @p lead says why OpenVINO. */
static void
ov_branch(const struct oc_splat_env *e, struct oc_splat_choice *o, const char *lead)
{
	if (is(e->ov_device, "cpu")) {
		o->backend = OC_SB_OV_CPU;
		snprintf(o->why, sizeof(o->why), "%.120s; OV_DEVICE=cpu", lead);
		return;
	}
	if (is(e->ov_device, "gpu")) {
		if (!e->ov_known || e->ov_gpu_count > 0) {
			// Explicit: any vendor, OpenVINO's default GPU device.
			o->backend = OC_SB_OV_GPU;
			o->ov_gpu = e->ov_known ? 0 : -1;
			o->provisional = !e->ov_known;
			snprintf(o->why, sizeof(o->why), "%.80s; OV_DEVICE=gpu%s%.80s%s", lead,
			         e->ov_known ? " on '" : "", e->ov_known ? e->ov_gpus[0].name : "",
			         e->ov_known ? "'" : "");
			return;
		}
		o->backend = OC_SB_OV_CPU;
		snprintf(o->fallback, sizeof(o->fallback),
		         "OV_DEVICE=gpu, but OpenVINO has no GPU device; using openvino-cpu");
		snprintf(o->why, sizeof(o->why), "%.120s; no OpenVINO GPU device", lead);
		return;
	}
	if (!e->ov_known) {
		char no_gpu[96];
		if (oc_rank_dxgi_rules_out_ov_gpu(e->adapters, e->adapter_count, no_gpu, sizeof(no_gpu))) {
			// Windows: the adapters already say OpenVINO's GPU device would be another
			// vendor's card or a UHD-class iGPU. Never a provisional GPU pick here: it
			// would compile the full model before the device check moves it away.
			o->backend = OC_SB_OV_CPU;
			snprintf(o->why, sizeof(o->why), "%.140s; %.100s", lead, no_gpu);
			return;
		}
		o->backend = OC_SB_OV_GPU;
		o->ov_gpu = -1;
		o->provisional = true;
		snprintf(o->why, sizeof(o->why), "%.120s; OpenVINO GPU pending its device check", lead);
		return;
	}
	int best = -1;
	char reject[160] = "no OpenVINO GPU device";
	for (uint32_t i = 0; i < e->ov_gpu_count && e->ov_gpus != NULL; i++) {
		char why[160];
		const struct oc_ov_gpu *g = &e->ov_gpus[i];
		if (!oc_rank_ov_gpu_ok(g, why, sizeof(why))) {
			snprintf(reject, sizeof(reject), "OpenVINO %.10s %.130s", g->device, why);
			continue;
		}
		const struct oc_ov_gpu *b = best >= 0 ? &e->ov_gpus[best] : NULL;
		if (b == NULL || (g->type == OC_OV_GPU_TYPE_DISCRETE && b->type != OC_OV_GPU_TYPE_DISCRETE) ||
		    (g->type == b->type && g->eu_count > b->eu_count)) {
			best = (int)i;
		}
	}
	if (best >= 0) {
		const struct oc_ov_gpu *g = &e->ov_gpus[best];
		o->backend = OC_SB_OV_GPU;
		o->ov_gpu = best;
		snprintf(o->why, sizeof(o->why), "%.100s; OpenVINO %.10s '%.80s' (Intel %s, IP %u.%u)", lead, g->device,
		         g->name,
		         g->type == OC_OV_GPU_TYPE_DISCRETE     ? "discrete"
		         : g->type == OC_OV_GPU_TYPE_INTEGRATED ? "integrated"
		                                                : "GPU",
		         g->arch_major, g->arch_minor);
		return;
	}
	o->backend = OC_SB_OV_CPU;
	snprintf(o->why, sizeof(o->why), "%.140s; %.110s", lead, reject);
}

void
oc_rank_splat(const struct oc_splat_env *e, struct oc_splat_choice *o)
{
	memset(o, 0, sizeof(*o));
	o->adapter = -1;
	o->ov_gpu = -1;
	o->backend = OC_SB_ORT_CPU;
	const bool force_full = is(e->model, "full");
	const bool force_fp16 = is(e->model, "fp16");
	const bool force_int4 = is(e->model, "int4");
	const int disc = e->dml_available ? best_discrete(e) : -1;
	const bool fp16_ok = e->fp16_on_dml && e->fp16_ready && !e->fp16_failed;

	if (is(e->ep, "cpu")) {
		snprintf(o->why, sizeof(o->why), "EP=cpu");
	} else if (is(e->ep, "dml")) {
		const int a = disc >= 0 ? disc : e->dml_available ? first_hardware(e) : -1;
		if (a >= 0) {
			dml_on(e, o, a, "EP=dml");
		} else {
			snprintf(o->fallback, sizeof(o->fallback), "EP=dml, but %s; using onnxruntime-cpu",
			         e->dml_available ? "no hardware DXGI adapter"
			                          : "this ONNX Runtime has no DirectML (Windows only)");
			snprintf(o->why, sizeof(o->why), "EP=dml unavailable");
		}
	} else if (is(e->ep, "coreml")) {
		if (e->coreml_available) {
			o->backend = OC_SB_COREML;
			snprintf(o->why, sizeof(o->why), "EP=coreml");
		} else {
			snprintf(o->fallback, sizeof(o->fallback),
			         "EP=coreml, but this ONNX Runtime has no Core ML; using onnxruntime-cpu");
			snprintf(o->why, sizeof(o->why), "EP=coreml unavailable");
		}
	} else if (is(e->ep, "openvino")) {
		if (e->ov_present) {
			ov_branch(e, o, "EP=openvino");
		} else {
			snprintf(o->fallback, sizeof(o->fallback),
			         "EP=openvino, but OpenVINO is not there; using onnxruntime-cpu");
			snprintf(o->why, sizeof(o->why), "EP=openvino unavailable");
		}
	} else {
		// auto. DirectML serves the full and fp16 models only (see oc_rank.h), so it
		// ranks first only when one of them is what would run there.
		const char *skip = NULL;
		char skip_buf[128];
		const uint64_t vram = disc >= 0 ? e->adapters[disc].dedicated_mb : 0;
		const enum oc_splat_model auto_model = oc_rank_auto_model(true, e->ram_mb, true, vram, fp16_ok);
		if (!e->dml_available) {
			skip = "no DirectML";
		} else if (disc < 0) {
			skip = "no discrete non-Intel GPU";
		} else if (force_int4) {
			skip = "MODEL=int4 does not run on DirectML";
		} else if (force_fp16 && !e->fp16_on_dml) {
			skip = "MODEL=fp16, but no pack has an fp16 model";
		} else if (force_full && !e->full_on_dml) {
			skip = "the full model has no DirectML variant";
		} else if (!force_full && !force_fp16 &&
		           (auto_model == OC_SM_INT4 || (auto_model == OC_SM_FULL && !e->full_on_dml))) {
			if (e->ram_mb < OC_FULL_MODEL_MIN_RAM_MB) {
				skip = "under 16 GB of RAM for the full model";
			} else {
				if (e->fp16_on_dml && e->fp16_ready && e->fp16_failed) {
					snprintf(skip_buf, sizeof(skip_buf),
					         "the fp16 model failed; %llu MB of VRAM, the full model needs %llu MB",
					         (unsigned long long)vram, (unsigned long long)OC_FULL_MODEL_MIN_VRAM_MB);
				} else if (fp16_ok) {
					snprintf(skip_buf, sizeof(skip_buf),
					         "%llu MB of VRAM: the full model needs %llu MB, the fp16 one %llu MB",
					         (unsigned long long)vram,
					         (unsigned long long)OC_FULL_MODEL_MIN_VRAM_MB,
					         (unsigned long long)OC_FP16_MODEL_MIN_VRAM_MB);
				} else {
					snprintf(skip_buf, sizeof(skip_buf),
					         "%llu MB of VRAM: the full model needs %llu MB%s",
					         (unsigned long long)vram,
					         (unsigned long long)OC_FULL_MODEL_MIN_VRAM_MB,
					         e->fp16_on_dml ? " and the fp16 model is not built" : "");
				}
				skip = skip_buf;
			}
		}
		if (skip == NULL) {
			dml_on(e, o, disc, "EP=auto");
		} else if (e->ov_present) {
			char lead[192];
			snprintf(lead, sizeof(lead), "EP=auto (DirectML: %s)", skip);
			ov_branch(e, o, lead);
		} else {
			snprintf(o->why, sizeof(o->why), "EP=auto (DirectML: %s; no OpenVINO)", skip);
		}
	}

	const bool gpu = o->backend == OC_SB_DML || o->backend == OC_SB_OV_GPU || o->backend == OC_SB_COREML;
	const bool on_dml = o->backend == OC_SB_DML;
	const uint64_t vram = on_dml ? e->adapters[o->adapter].dedicated_mb : 0;
	if (force_full) {
		o->model = OC_SM_FULL;
	} else if (force_int4) {
		o->model = OC_SM_INT4;
	} else if (force_fp16) {
		// fp16 has a DirectML variant only: anywhere else MODEL=fp16 means the full model.
		o->model = on_dml && e->fp16_on_dml && !e->fp16_failed ? OC_SM_FP16 : OC_SM_FULL;
	} else {
		o->model = oc_rank_auto_model(gpu, e->ram_mb, on_dml, vram, fp16_ok);
	}
}
