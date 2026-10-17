// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  DisplayXR open-model conversion provider (ADR-053, ProbeOrder 200).
 *
 * Two capabilities, both depth-only; the runtime composes views from them
 * with its reference view synthesis (discovery spec §1.1, §6.2):
 *
 *   cap 1  DEPTH_VIDEO  stream lane, TEMPORAL: a small streaming video-depth
 *                       model with an explicit per-stream temporal cache.
 *   cap 2  DEPTH_MONO   stream + job lanes: a small still-depth model.
 *
 * Inference runs through ONNX Runtime, loaded at run time from the provider's
 * own directory: the DirectML execution provider on Windows when a hardware
 * adapter is there, the Core ML execution provider on macOS, the CPU
 * execution provider everywhere else (and as the fallback when DirectML or
 * Core ML cannot start or a Core ML run fails). Both models report affine disparity
 * (larger = nearer, unknown scale and shift), never an invented metric depth.
 *
 * Models come from the provider's model pack (schema 2,
 * dxr-open-default.models.json next to the binary, or setting PACK). The
 * provider reads the pack for identity, I/O and licence only; blob bytes are
 * resolved exclusively through host->model_resolve / host->model_request,
 * and readiness follows the discovery spec §4.3 table: INSTALLED when cold,
 * DOWNLOADING while the host fetches, ABSENT when a download is refused,
 * WARMING while the session loads, READY.
 *
 * Contract: docs/specs/runtime/conversion-provider-discovery.md in the
 * runtime repository. ABI: include/xrt/xrt_conversion_plugin.h (vendored).
 *
 * Threading: the runtime makes ONE [S] call at a time per provider (no
 * capability opts into overlap), so sessions and streams need no lock of
 * their own. The [C] slots read the describe and readiness fields under
 * `lock`.
 */

#include "xrt/xrt_conversion_plugin.h"

#include "oc_image.h"
#include "oc_ort.h"
#include "oc_pack.h"
#include "oc_platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 *
 * Identity.
 *
 */

#define OC_ID "open-conversion"
#define OC_VERSION "0.1.0"
#define OC_PACK_FILE "dxr-open-default.models.json"

#define OC_CAP_VIDEO 1u //!< DEPTH_VIDEO
#define OC_CAP_MONO 2u  //!< DEPTH_MONO
#define OC_CAP_COUNT 2u

#define OC_MAX_STREAMS_VIDEO 2u //!< each holds a temporal cache of ~160 MB (518 px model)
#define OC_MAX_STREAMS_MONO 4u
#define OC_MAX_STREAMS (OC_MAX_STREAMS_VIDEO + OC_MAX_STREAMS_MONO)
#define OC_MAX_INPUT_EDGE 2048u //!< the runtime downscales above it; the models run at ~518 px

#define OC_OPT_DEFAULT OC_OPT_EXTENDED
#define OC_POLL_MS 50u            //!< warm() polls the host and is_cancelled this often (≤ 250 ms)
#define OC_REQUEST_GRACE_MS 5000u //!< ABSENT this long after an accepted request = refused

/*! xrt_plugin_platform_state values (xrt_plugin.h), restated so this file
 *  needs only the conversion header. */
#define OC_PLATFORM_READY 1u
#define OC_PLATFORM_ABSENT 2u
#define OC_PLATFORM_INCOMPATIBLE 5u

#define COVERS(PTR, TYPE, FIELD) XRT_CONV_COVERS(PTR, TYPE, FIELD)
#define SET(PTR, TYPE, FIELD, VALUE)                                                                                   \
	do {                                                                                                           \
		if (COVERS(PTR, TYPE, FIELD)) {                                                                        \
			(PTR)->FIELD = (VALUE);                                                                        \
		}                                                                                                      \
	} while (0)


/*
 *
 * State.
 *
 */

struct oc_cap
{
	uint32_t cap_id;
	uint32_t task;
	const char *task_name; //!< pack task key
	uint32_t max_streams;

	/* Describe fields: written in probe and on an EP fallback, under `lock`. */
	const struct oc_pack_model *model;     //!< NULL: the pack has no usable model for the task
	const struct oc_pack_variant *variant; //!< the blob variant for the EP below
	char ref[192];                         //!< "<pack>/<model>/<variant>"
	enum oc_ep ep;
	char backend[32];
	uint64_t device_luid;
	uint32_t licence_class;
	uint32_t commercial_ok;
	uint32_t depth_kind;
	uint32_t depth_measure;

	/* Readiness, under `lock`. */
	uint32_t state, reason, retryable;
	float progress;
	char detail[XRT_CONV_DETAIL_MAX];
	char version[48];

	/* [S]-only. */
	struct oc_session *session;
};

struct oc_stream
{
	bool used;
	uint64_t id;
	uint32_t cap_id;
	struct oc_temporal *temporal;
	uint64_t frames;
};

struct oc_provider
{
	oc_mutex lock;
	uint64_t generation;
	struct oc_cap caps[OC_CAP_COUNT];
	struct oc_stream streams[OC_MAX_STREAMS]; //!< table under `lock` ([C] output desc reads it)
	uint64_t next_stream;

	/* Configuration (probe). */
	char module_dir[1024];
	char ort_path[1100];
	char pack_path[1100];
	char quality[16];
	char ep_setting[16];
	int threads;
	enum oc_graph_opt opt;
	enum oc_ort_log ort_log;
	enum oc_coreml_units coreml_units;
	char coreml_cache[1100]; //!< under the host's scratch dir; empty = no compiled-model cache

	/* Dependencies. */
	struct oc_pack *pack; //!< NULL: missing / invalid
	char pack_error[160];
	struct oc_ort *ort; //!< NULL until loaded; never unloaded
	enum oc_ort_status ort_status;
	char ort_detail[128];
	int dml_device;
	uint64_t dml_luid;
	bool coreml_cache_ready; //!< the cache directory exists (created on first use)

	/* [S]-only scratch, grown on demand. */
	float *chw;
	size_t chw_cap;
	float *map;
	size_t map_cap;
};

static const struct xrt_conversion_plugin_host_iface *g_host = NULL;
static struct oc_provider *g_provider = NULL; //!< for get_provider_status (no instance)
static oc_mutex g_global_lock;
static bool g_global_lock_init = false;


/*
 *
 * Host services.
 *
 */

static bool
host_has(size_t end)
{
	return g_host != NULL && g_host->struct_size >= end;
}
#define HOST_HAS(FIELD)                                                                                                \
	(host_has(offsetof(struct xrt_conversion_plugin_host_iface, FIELD) +                                           \
	          sizeof(((struct xrt_conversion_plugin_host_iface *)0)->FIELD)) &&                                    \
	 g_host->FIELD != NULL)

/*! DXR_CONV_OPEN_CONVERSION_<KEY> in the hosting process, else Settings\<KEY>
 *  (registry) / the manifest "settings" object. Never getenv(). */
static bool
setting(const char *key, char *buf, uint32_t size)
{
	if (!HOST_HAS(get_setting)) {
		return false;
	}
	buf[0] = '\0';
	return g_host->get_setting(key, buf, size) && buf[0] != '\0';
}

static void
notify(uint32_t cap_id)
{
	if (HOST_HAS(notify_changed)) {
		g_host->notify_changed(cap_id);
	}
}

static bool
host_path(uint32_t kind, char *buf, uint32_t size)
{
	if (!HOST_HAS(get_path)) {
		return false;
	}
	buf[0] = '\0';
	return g_host->get_path(kind, buf, size) && buf[0] != '\0';
}

static bool
store_resolve(const char *ref, struct xrt_conv_model_location *loc)
{
	memset(loc, 0, sizeof(*loc));
	loc->struct_size = (uint32_t)sizeof(*loc);
	if (!HOST_HAS(model_resolve)) {
		return false;
	}
	return g_host->model_resolve(ref, loc);
}

static bool
store_request(const char *ref)
{
	return HOST_HAS(model_request) && g_host->model_request(ref);
}


/*
 *
 * Helpers.
 *
 */

static struct oc_cap *
cap_get(struct oc_provider *p, uint32_t cap_id)
{
	for (uint32_t i = 0; i < OC_CAP_COUNT; i++) {
		if (p->caps[i].cap_id == cap_id) {
			return &p->caps[i];
		}
	}
	return NULL;
}

/*! Set readiness; notify on a state change, or on a ≥ 5 % progress step. */
static void
set_state(struct oc_provider *p,
          struct oc_cap *c,
          uint32_t state,
          uint32_t reason,
          uint32_t retryable,
          float progress,
          const char *detail)
{
	oc_lock(&p->lock);
	const bool changed = c->state != state || c->reason != reason;
	const bool stepped = fabsf(progress - c->progress) >= 0.05f;
	c->state = state;
	c->reason = reason;
	c->retryable = retryable;
	c->progress = progress;
	if (detail != NULL) {
		oc_copy_str(c->detail, sizeof(c->detail), detail);
	}
	oc_unlock(&p->lock);
	if (changed || stepped) {
		notify(c->cap_id);
	}
}

static uint32_t
state_of(struct oc_provider *p, const struct oc_cap *c)
{
	oc_lock(&p->lock);
	const uint32_t s = c->state;
	oc_unlock(&p->lock);
	return s;
}

static bool
never_cancelled(void *ctx)
{
	(void)ctx;
	return false;
}

static uint32_t
depth_kind_from(const char *s)
{
	if (strcmp(s, "metric") == 0) {
		return XRT_CONV_DEPTH_METRIC;
	}
	if (strcmp(s, "metric-inverse") == 0) {
		return XRT_CONV_DEPTH_METRIC_INVERSE;
	}
	if (strcmp(s, "affine-inverse") == 0) {
		return XRT_CONV_DEPTH_AFFINE_INVERSE;
	}
	if (strcmp(s, "affine-disparity") == 0) {
		return XRT_CONV_DEPTH_AFFINE_DISPARITY;
	}
	if (strcmp(s, "affine") == 0) {
		return XRT_CONV_DEPTH_AFFINE;
	}
	return XRT_CONV_DEPTH_UNKNOWN;
}

static bool
grow(float **buf, size_t *cap, size_t need)
{
	if (*cap >= need) {
		return true;
	}
	float *n = (float *)realloc(*buf, need * sizeof(float));
	if (n == NULL) {
		return false;
	}
	*buf = n;
	*cap = need;
	return true;
}

static bool
is_colour_format(uint32_t f)
{
	return f == XRT_CONV_FMT_RGBA8_UNORM || f == XRT_CONV_FMT_RGBA8_SRGB || f == XRT_CONV_FMT_BGRA8_UNORM ||
	       f == XRT_CONV_FMT_BGRA8_SRGB;
}

static bool
is_bgr(uint32_t f)
{
	return f == XRT_CONV_FMT_BGRA8_UNORM || f == XRT_CONV_FMT_BGRA8_SRGB;
}

/*! Long edge of the depth output the provider writes for this cap's model. */
static uint32_t
output_long_edge(const struct oc_pack_model *m)
{
	if (m == NULL) {
		return 0;
	}
	if (m->native_long_edge > 0) {
		return m->native_long_edge;
	}
	return m->fixed ? (m->width > m->height ? m->width : m->height) : 518u;
}

/*! The model's inference grid for an input of w × h. */
static void
inference_grid(const struct oc_pack_model *m, uint32_t w, uint32_t h, uint32_t *gw, uint32_t *gh)
{
	if (m->fixed) {
		*gw = m->width;
		*gh = m->height;
		return;
	}
	oc_fit_long_edge(w, h, output_long_edge(m), m->multiple_of, false, gw, gh);
}

static const char *
ep_pack_name(enum oc_ep ep)
{
	return ep == OC_EP_DML ? "dml" : ep == OC_EP_COREML ? "coreml" : "cpu";
}

static const char *
ep_display_name(enum oc_ep ep)
{
	return ep == OC_EP_DML ? "DirectML" : ep == OC_EP_COREML ? "Core ML" : "CPU";
}

/*! Describe-time fields for a cap on an EP, from its pack model. Caller holds `lock` or is in probe. */
static void
bind_cap_locked(struct oc_provider *p, struct oc_cap *c, enum oc_ep ep)
{
	c->model = NULL;
	c->variant = NULL;
	c->ref[0] = '\0';
	const struct oc_pack_model *m =
	    p->pack != NULL ? oc_pack_default_model(p->pack, c->task_name, p->quality) : NULL;
	const struct oc_pack_variant *v = NULL;
	if (m != NULL) {
		v = oc_pack_variant_for_ep(m, ep_pack_name(ep));
		if (v == NULL && ep != OC_EP_CPU) {
			// No variant for the accelerator (the pack decides, per model, which
			// execution providers a blob is acceptable on): this cap runs on the CPU.
			ep = OC_EP_CPU;
			v = oc_pack_variant_for_ep(m, "cpu");
		}
	}
	c->ep = ep;
	oc_copy_str(c->backend, sizeof(c->backend),
	            ep == OC_EP_DML      ? "onnxruntime-dml"
	            : ep == OC_EP_COREML ? "onnxruntime-coreml"
	                                 : "onnxruntime-cpu");
	c->device_luid = ep == OC_EP_DML ? p->dml_luid : 0;
	if (m == NULL || v == NULL) {
		c->licence_class = XRT_CONV_LICENCE_UNKNOWN;
		c->commercial_ok = 0;
		c->depth_kind = XRT_CONV_DEPTH_AFFINE_DISPARITY; // what the default pack's models produce
		c->depth_measure = XRT_CONV_MEASURE_PLANAR;
		return;
	}
	c->model = m;
	c->variant = v;
	oc_pack_ref(p->pack, m, v, c->ref, sizeof(c->ref));
	c->depth_kind = depth_kind_from(m->depth_kind);
	c->depth_measure = strcmp(m->measure, "ray") == 0 ? XRT_CONV_MEASURE_RAY : XRT_CONV_MEASURE_PLANAR;
	// Commercial = AND over the model, its dependencies, the pack's own flag and
	// the inference runtime (ONNX Runtime: MIT). DirectML ships under Microsoft's
	// redistributable licence: commercial use is allowed, but it is not an open
	// licence, so a DirectML-backed capability is COMMERCIAL, a CPU one PERMISSIVE.
	// The Core ML execution provider is part of ONNX Runtime (MIT) and drives the
	// operating system's own Core ML framework, which nothing redistributes: a
	// Core ML-backed capability stays PERMISSIVE.
	const bool commercial = p->pack->commercial && m->commercial;
	c->commercial_ok = commercial ? 1u : 0u;
	c->licence_class = !commercial       ? XRT_CONV_LICENCE_RESEARCH_ONLY
	                   : ep == OC_EP_DML ? XRT_CONV_LICENCE_COMMERCIAL
	                                     : XRT_CONV_LICENCE_PERMISSIVE;
}

/*! Whether the cap's model fits the inference shape this provider implements. */
static bool
model_usable(const struct oc_cap *c, char *why, size_t why_size)
{
	const struct oc_pack_model *m = c->model;
	if (m == NULL) {
		snprintf(why, why_size, "the model pack has no %s model for this backend", c->task_name);
		return false;
	}
	if (depth_kind_from(m->depth_kind) == XRT_CONV_DEPTH_UNKNOWN || m->t_image[0] == '\0' ||
	    m->t_depth[0] == '\0') {
		snprintf(why, why_size, "pack model %s lacks a depth kind or tensor names", m->name);
		return false;
	}
	if (c->task == XRT_CONV_TASK_DEPTH_VIDEO && (!m->fixed || m->cache_rank == 0 || m->t_cache_in[0] == '\0' ||
	                                             m->t_cache_out[0] == '\0' || m->t_is_first[0] == '\0')) {
		snprintf(why, why_size, "pack model %s is not a fixed-size streaming model with a cache", m->name);
		return false;
	}
	return true;
}


/*
 *
 * Provider status, probe, describe, state.
 *
 */

static bool
oc_get_provider_status(struct xrt_conv_provider_status *out)
{
	if (out == NULL) {
		return false;
	}
	uint32_t state = OC_PLATFORM_READY;
	char hint[128] = "";
	oc_lock(&g_global_lock);
	struct oc_provider *p = g_provider;
	if (p != NULL) {
		// Probed: answer from memory.
		switch (p->ort_status) {
		case OC_ORT_OK: state = OC_PLATFORM_READY; break;
		case OC_ORT_ABSENT: state = OC_PLATFORM_ABSENT; break;
		default: state = OC_PLATFORM_INCOMPATIBLE; break;
		}
		oc_copy_str(hint, sizeof(hint), p->ort_detail);
	} else {
		// Not probed yet: the cheap check is whether the runtime library is there.
		char dir[1024], path[1100];
		if (oc_module_dir(dir, sizeof(dir))) {
			snprintf(path, sizeof(path), "%s%c%s", dir, OC_PATH_SEP, oc_ort_library_name());
			const bool there = oc_file_exists(path);
			state = there ? OC_PLATFORM_READY : OC_PLATFORM_ABSENT;
			snprintf(hint, sizeof(hint),
			         there ? "ONNX Runtime present (not loaded yet)"
			               : "ONNX Runtime not found next to the provider");
		}
	}
	oc_unlock(&g_global_lock);
	SET(out, struct xrt_conv_provider_status, state, state);
	SET(out, struct xrt_conv_provider_status, flags, 0u); // neither FALLBACK nor the runtime's REFERENCE
	if (COVERS(out, struct xrt_conv_provider_status, hint)) {
		oc_copy_str(out->hint, sizeof(out->hint), hint);
	}
	return true;
}

/*! Load ONNX Runtime and pick the execution provider. Idempotent once OK. */
static void
load_runtime(struct oc_provider *p)
{
	if (p->ort != NULL) {
		return;
	}
	struct oc_ort *ort = NULL;
	char detail[128];
	const enum oc_ort_status s = oc_ort_load(p->ort_path, p->ort_log, &ort, detail, sizeof(detail));
	oc_lock(&p->lock);
	p->ort_status = s;
	oc_copy_str(p->ort_detail, sizeof(p->ort_detail), detail);
	p->ort = ort;
	oc_unlock(&p->lock);
}

static enum oc_ep
choose_ep(struct oc_provider *p)
{
	if (strcmp(p->ep_setting, "cpu") == 0) {
		return OC_EP_CPU;
	}
	if (oc_ort_has_coreml(p->ort) && strcmp(p->ep_setting, "dml") != 0) {
		return OC_EP_COREML; // macOS: the Core ML EP is in the official osx-arm64 build
	}
	if (!oc_ort_has_dml(p->ort) || strcmp(p->ep_setting, "coreml") == 0) {
		return OC_EP_CPU;
	}
	struct xrt_conv_adapter_hint hint;
	memset(&hint, 0, sizeof(hint));
	hint.struct_size = (uint32_t)sizeof(hint);
	if (HOST_HAS(get_adapter_hint) && !g_host->get_adapter_hint(&hint)) {
		memset(&hint, 0, sizeof(hint));
	}
	int index = -1;
	uint64_t luid = 0;
	if (!oc_dml_pick_adapter(hint.preferred_luid, hint.avoid_luid, &index, &luid)) {
		return OC_EP_CPU; // no hardware adapter: DirectML on WARP is slower than the CPU EP
	}
	p->dml_device = index;
	p->dml_luid = luid;
	return OC_EP_DML;
}

/*! Readiness of a cold cap from the store, without loading anything. */
static void
initial_state(struct oc_provider *p, struct oc_cap *c)
{
	char why[XRT_CONV_DETAIL_MAX];
	if (p->ort_status == OC_ORT_ABSENT) {
		set_state(p, c, XRT_CONV_STATE_ABSENT, XRT_CONV_FAIL_NONE, 0, -1.0f, p->ort_detail);
		return;
	}
	if (p->ort_status == OC_ORT_TOO_OLD) {
		// Present but older than our floor: VERSION, never MODEL ("too old", not "missing").
		set_state(p, c, XRT_CONV_STATE_FAILED, XRT_CONV_FAIL_VERSION, 1, -1.0f, p->ort_detail);
		return;
	}
	if (p->ort_status != OC_ORT_OK) {
		set_state(p, c, XRT_CONV_STATE_FAILED, XRT_CONV_FAIL_LOAD, 0, -1.0f, p->ort_detail);
		return;
	}
	if (p->pack == NULL) {
		set_state(p, c, XRT_CONV_STATE_ABSENT, XRT_CONV_FAIL_NONE, 0, -1.0f, p->pack_error);
		return;
	}
	if (!model_usable(c, why, sizeof(why))) {
		set_state(p, c, XRT_CONV_STATE_ABSENT, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
		return;
	}
	struct xrt_conv_model_location loc;
	if (store_resolve(c->ref, &loc) && loc.state == XRT_CONV_STATE_DOWNLOADING) {
		snprintf(why, sizeof(why), "%s: downloading", c->model->name);
		set_state(p, c, XRT_CONV_STATE_DOWNLOADING, XRT_CONV_FAIL_NONE, 0, loc.progress, why);
		return;
	}
	const bool present = loc.state == XRT_CONV_STATE_READY;
	snprintf(why, sizeof(why), "%s: cold%s", c->model->name, present ? "" : "; the model downloads on first use");
	set_state(p, c, XRT_CONV_STATE_INSTALLED, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
}

static xrt_result_t
oc_probe(struct xrt_conversion_plugin_instance **out_inst)
{
	if (out_inst == NULL) {
		return XRT_ERROR_PROBER_NOT_SUPPORTED;
	}
	char buf[1024];
	if (setting("ENABLE", buf, sizeof(buf)) && strtoul(buf, NULL, 10) == 0) {
		return XRT_ERROR_PROBER_NOT_SUPPORTED; // clean decline
	}
	struct oc_provider *p = (struct oc_provider *)calloc(1, sizeof(*p));
	if (p == NULL) {
		return XRT_ERROR_ALLOCATION;
	}
	oc_mutex_init(&p->lock);
	p->generation = 1;
	p->dml_device = -1;

	// Settings, read once here (never per frame). No model load, no network.
	oc_copy_str(p->quality, sizeof(p->quality), "medium");
	if (setting("QUALITY", buf, sizeof(buf)) &&
	    (strcmp(buf, "low") == 0 || strcmp(buf, "medium") == 0 || strcmp(buf, "high") == 0)) {
		oc_copy_str(p->quality, sizeof(p->quality), buf);
	}
	oc_copy_str(p->ep_setting, sizeof(p->ep_setting), "auto");
	if (setting("EP", buf, sizeof(buf)) &&
	    (strcmp(buf, "cpu") == 0 || strcmp(buf, "dml") == 0 || strcmp(buf, "coreml") == 0)) {
		oc_copy_str(p->ep_setting, sizeof(p->ep_setting), buf);
	}
	p->coreml_units = OC_COREML_ALL;
	if (setting("COREML_UNITS", buf, sizeof(buf))) {
		p->coreml_units = strcmp(buf, "gpu") == 0   ? OC_COREML_GPU
		                  : strcmp(buf, "ane") == 0 ? OC_COREML_ANE
		                  : strcmp(buf, "cpu") == 0 ? OC_COREML_CPU
		                                            : OC_COREML_ALL;
	}
	// Core ML compiles the model on the first session; the compiled model is kept
	// in the host's per-provider scratch directory (never the model store), keyed
	// by the ONNX Runtime version below it. No scratch directory = no cache.
	p->coreml_cache[0] = '\0';
	if (!(setting("COREML_CACHE", buf, sizeof(buf)) && strtoul(buf, NULL, 10) == 0) &&
	    host_path(XRT_CONV_PATH_SCRATCH, buf, sizeof(buf))) {
		oc_copy_str(p->coreml_cache, sizeof(p->coreml_cache), buf);
	}
	p->ort_log = OC_ORT_LOG_WARNING;
	if (setting("ORT_LOG", buf, sizeof(buf))) {
		p->ort_log = strcmp(buf, "verbose") == 0 ? OC_ORT_LOG_VERBOSE
		             : strcmp(buf, "info") == 0  ? OC_ORT_LOG_INFO
		                                         : OC_ORT_LOG_WARNING;
	}
	p->threads = 4; // a background service: never take every core
	if (setting("THREADS", buf, sizeof(buf))) {
		p->threads = (int)strtol(buf, NULL, 10);
	}
	p->opt = OC_OPT_DEFAULT;
	if (setting("GRAPH_OPT", buf, sizeof(buf))) {
		p->opt = strcmp(buf, "all") == 0     ? OC_OPT_ALL
		         : strcmp(buf, "basic") == 0 ? OC_OPT_BASIC
		                                     : OC_OPT_EXTENDED;
	}
	if (!oc_module_dir(p->module_dir, sizeof(p->module_dir))) {
		p->module_dir[0] = '\0';
	}
	if (setting("ONNXRUNTIME", buf, sizeof(buf))) {
		oc_copy_str(p->ort_path, sizeof(p->ort_path), buf);
	} else {
		snprintf(p->ort_path, sizeof(p->ort_path), "%s%c%s", p->module_dir, OC_PATH_SEP, oc_ort_library_name());
	}
	if (setting("PACK", buf, sizeof(buf))) {
		oc_copy_str(p->pack_path, sizeof(p->pack_path), buf);
	} else {
		snprintf(p->pack_path, sizeof(p->pack_path), "%s%c%s", p->module_dir, OC_PATH_SEP, OC_PACK_FILE);
	}

	// The pack (a small JSON file, read and closed here).
	p->pack = (struct oc_pack *)calloc(1, sizeof(*p->pack));
	if (p->pack == NULL) {
		free(p);
		return XRT_ERROR_ALLOCATION;
	}
	char err[160] = "";
	if (!oc_pack_load(p->pack_path, p->pack, err, sizeof(err))) {
		snprintf(p->pack_error, sizeof(p->pack_error), "model pack unusable: %s", err);
		free(p->pack);
		p->pack = NULL;
	}

	// The inference runtime: mapping it is not a model load, and probe must be
	// able to say ABSENT / FAILED(VERSION) up front.
	load_runtime(p);
	const enum oc_ep ep = p->ort != NULL ? choose_ep(p) : OC_EP_CPU;

	struct oc_cap *v = &p->caps[0];
	v->cap_id = OC_CAP_VIDEO;
	v->task = XRT_CONV_TASK_DEPTH_VIDEO;
	v->task_name = "depth-video";
	v->max_streams = OC_MAX_STREAMS_VIDEO;
	struct oc_cap *m = &p->caps[1];
	m->cap_id = OC_CAP_MONO;
	m->task = XRT_CONV_TASK_DEPTH_MONO;
	m->task_name = "depth-mono";
	m->max_streams = OC_MAX_STREAMS_MONO;
	for (uint32_t i = 0; i < OC_CAP_COUNT; i++) {
		p->caps[i].state = XRT_CONV_STATE_UNKNOWN;
		p->caps[i].progress = -1.0f;
		bind_cap_locked(p, &p->caps[i], ep);
		initial_state(p, &p->caps[i]);
	}

	oc_lock(&g_global_lock);
	g_provider = p;
	oc_unlock(&g_global_lock);
	*out_inst = (struct xrt_conversion_plugin_instance *)p;
	return XRT_SUCCESS;
}

static void
fill_cap(struct oc_provider *p, const struct oc_cap *c, struct xrt_conv_capability *o)
{
#define C(FIELD, VALUE) SET(o, struct xrt_conv_capability, FIELD, VALUE)
#define S(FIELD, VALUE)                                                                                                \
	do {                                                                                                           \
		if (COVERS(o, struct xrt_conv_capability, FIELD)) {                                                    \
			oc_copy_str(o->FIELD, sizeof(o->FIELD), (VALUE));                                              \
		}                                                                                                      \
	} while (0)
	oc_lock(&p->lock);
	const bool video = c->task == XRT_CONV_TASK_DEPTH_VIDEO;
	C(cap_id, c->cap_id);
	C(task, c->task);
	C(lanes, video ? XRT_CONV_LANE_STREAM : (XRT_CONV_LANE_STREAM | XRT_CONV_LANE_JOB));
	C(input_kinds, XRT_CONV_INPUT_MONO);
	C(outputs, XRT_CONV_OUT_DEPTH);
	C(flags,
	  (video ? XRT_CONV_CAP_TEMPORAL : 0u) | XRT_CONV_CAP_USES_MODEL_STORE | XRT_CONV_CAP_NEEDS_NETWORK_TO_WARM);
	// CPU import on every lane: the runtime copies GPU inputs for us.
	C(import_memory_kinds, XRT_CONV_MEM_CPU);
	C(import_sync_kinds, 1u << XRT_CONV_SYNC_NONE);
	C(input_formats, (1u << XRT_CONV_FMT_RGBA8_UNORM) | (1u << XRT_CONV_FMT_RGBA8_SRGB) |
	                     (1u << XRT_CONV_FMT_BGRA8_UNORM) | (1u << XRT_CONV_FMT_BGRA8_SRGB));
	C(depth_output_format, XRT_CONV_FMT_R32_FLOAT);
	C(min_views, 0u);
	C(max_views, 0u);
	C(max_streams, c->max_streams);
	C(max_concurrent, 1u);
	C(max_input_edge, OC_MAX_INPUT_EDGE);
	C(preferred_input_edge, output_long_edge(c->model));
	C(depth_kind, c->depth_kind);
	C(depth_measure, c->depth_measure);
	C(intrinsics_source, XRT_CONV_INTRINSICS_NONE);
	C(splat_formats, 0u);
	C(licence_class, c->licence_class);
	C(commercial_ok, c->commercial_ok);
	S(licence_spdx, c->model != NULL ? c->model->spdx : "");
	S(model_id, c->model != NULL ? c->model->name : "");
	S(model_version, c->model != NULL ? c->model->version : "");
	S(backend, c->backend);
	S(display_name, video ? "Open video depth" : "Open still depth");
	C(device_luid, c->device_luid);
	// Unknown estimates stay 0; nothing here is invented.
	C(est_warmup_ms, 0u);
	C(est_latency_ms, 0u);
	C(est_vram_mb, 0u);
	C(est_ram_mb, 0u);
	oc_unlock(&p->lock);
#undef C
#undef S
}

static xrt_result_t
oc_describe_cap(struct xrt_conversion_plugin_instance *inst,
                uint32_t index,
                struct xrt_conv_capability *out,
                uint32_t *out_count)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	if (out_count != NULL) {
		*out_count = OC_CAP_COUNT;
	}
	if (out == NULL) {
		return out_count != NULL ? XRT_SUCCESS : XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	if (p == NULL || index >= OC_CAP_COUNT) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	fill_cap(p, &p->caps[index], out);
	return XRT_SUCCESS;
}

static uint64_t
oc_get_generation(struct xrt_conversion_plugin_instance *inst)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	if (p == NULL) {
		return 0;
	}
	oc_lock(&p->lock);
	const uint64_t g = p->generation;
	oc_unlock(&p->lock);
	return g;
}

static xrt_result_t
oc_get_state(struct xrt_conversion_plugin_instance *inst, uint32_t cap_id, struct xrt_conv_state *out)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	struct oc_cap *c = p != NULL ? cap_get(p, cap_id) : NULL;
	if (c == NULL || out == NULL) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	oc_lock(&p->lock);
	SET(out, struct xrt_conv_state, state, c->state);
	SET(out, struct xrt_conv_state, failure_reason, c->reason);
	SET(out, struct xrt_conv_state, retryable, c->retryable);
	SET(out, struct xrt_conv_state, progress, c->progress);
	SET(out, struct xrt_conv_state, eta_ns, 0ull);
	if (COVERS(out, struct xrt_conv_state, version)) {
		oc_copy_str(out->version, sizeof(out->version), c->version);
	}
	if (COVERS(out, struct xrt_conv_state, detail)) {
		oc_copy_str(out->detail, sizeof(out->detail), c->detail);
	}
	oc_unlock(&p->lock);
	return XRT_SUCCESS;
}


/*
 *
 * warm / evict.
 *
 */

/*! Run one inference on a grey frame so the first real frame pays no
 *  compile / allocation cost. */
static enum oc_run_status
warm_up_inference(struct oc_provider *p, struct oc_cap *c, char *err, size_t err_size)
{
	const struct oc_pack_model *m = c->model;
	uint32_t gw = 0, gh = 0;
	inference_grid(m, 1920, 1080, &gw, &gh);
	const size_t n = (size_t)gw * gh;
	if (!grow(&p->chw, &p->chw_cap, n * 3u) || !grow(&p->map, &p->map_cap, n * 4u)) {
		snprintf(err, err_size, "out of memory");
		return OC_RUN_OUT_OF_MEMORY;
	}
	for (size_t i = 0; i < n * 3u; i++) {
		p->chw[i] = m->normalize ? (0.5f - m->mean[i / n]) / m->std[i / n] : 0.5f;
	}
	uint32_t ow = 0, oh = 0;
	if (c->task == XRT_CONV_TASK_DEPTH_VIDEO) {
		struct oc_temporal *t = NULL;
		const struct oc_temporal_io io = {m->t_image,     m->t_cache_in,  m->t_is_first, m->t_depth,
		                                  m->t_cache_out, m->cache_shape, m->cache_rank};
		const enum oc_run_status r = oc_temporal_run(c->session, &t, &io, p->chw, gw, gh, true, p->map,
		                                             p->map_cap, &ow, &oh, err, err_size);
		oc_temporal_destroy(t);
		return r;
	}
	return oc_session_run_image(c->session, m->t_image, p->chw, gw, gh, m->t_depth, p->map, p->map_cap, &ow, &oh,
	                            err, err_size);
}

/*! Drop every stream's temporal state on @p cap_id. Called before the cap's
 *  session is destroyed (a binding never outlives its session); the streams
 *  survive and restart their history at the next frame. */
static void
drop_temporal(struct oc_provider *p, uint32_t cap_id)
{
	for (uint32_t i = 0; i < OC_MAX_STREAMS; i++) {
		oc_lock(&p->lock);
		struct oc_temporal *t = NULL;
		if (p->streams[i].used && p->streams[i].cap_id == cap_id) {
			t = p->streams[i].temporal;
			p->streams[i].temporal = NULL;
		}
		oc_unlock(&p->lock);
		oc_temporal_destroy(t);
	}
}

static void
release_session(struct oc_provider *p, struct oc_cap *c)
{
	if (c->session != NULL) {
		drop_temporal(p, c->cap_id);
		oc_session_destroy(c->session);
		c->session = NULL;
	}
}

/*! Move a cap to the CPU execution provider after DirectML / Core ML could not
 *  start or a Core ML run failed. Sticky for the process: later warms stay on
 *  the CPU. */
static void
fall_back_to_cpu(struct oc_provider *p, struct oc_cap *c, const char *why)
{
	const enum oc_ep from = c->ep;
	oc_lock(&p->lock);
	bind_cap_locked(p, c, OC_EP_CPU);
	p->generation++; // backend, device and licence class changed: re-describe
	oc_unlock(&p->lock);
	fprintf(stderr, "[open-conversion] %s: %s unavailable (%s); using the CPU\n", c->task_name,
	        ep_display_name(from), why);
	notify(0);
}

/*! The Core ML compiled-model cache directory for this ONNX Runtime, created
 *  on first use under the host's scratch directory; NULL = no cache. */
static const char *
coreml_cache_dir(struct oc_provider *p, char *buf, size_t size)
{
	if (p->coreml_cache[0] == '\0') {
		return NULL;
	}
	// Keyed by the ONNX Runtime version: the Core ML EP does not detect a cache
	// written by another version. Blob paths are content-addressed (sha256), so
	// a changed model is a new cache key on its own.
	snprintf(buf, size, "%s%ccoreml-ort-%s", p->coreml_cache, OC_PATH_SEP, oc_ort_version(p->ort));
	if (!p->coreml_cache_ready) {
		p->coreml_cache_ready = oc_make_dir(buf);
		if (!p->coreml_cache_ready) {
			fprintf(stderr, "[open-conversion] cannot create the Core ML cache %s; compiling every warm\n",
			        buf);
			p->coreml_cache[0] = '\0';
			return NULL;
		}
	}
	return buf;
}

static xrt_result_t
oc_warm(struct xrt_conversion_plugin_instance *inst, uint32_t cap_id, bool (*is_cancelled)(void *ctx), void *cancel_ctx)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	struct oc_cap *c = p != NULL ? cap_get(p, cap_id) : NULL;
	if (c == NULL) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	if (is_cancelled == NULL) {
		is_cancelled = never_cancelled;
	}
	if (state_of(p, c) == XRT_CONV_STATE_READY && c->session != NULL) {
		return XRT_SUCCESS;
	}

	// Dependencies first: a runtime installed since probe is picked up here.
	if (p->ort == NULL) {
		load_runtime(p);
		if (p->ort != NULL) {
			oc_lock(&p->lock);
			bind_cap_locked(p, c, choose_ep(p));
			p->generation++;
			oc_unlock(&p->lock);
		}
	}
	char why[XRT_CONV_DETAIL_MAX];
	if (p->ort == NULL || p->pack == NULL || !model_usable(c, why, sizeof(why))) {
		initial_state(p, c);
		return XRT_ERROR_FEATURE_NOT_SUPPORTED; // terminal for now; the runtime falls through
	}
	release_session(p, c); // a FAILED(DEVICE_LOST) session is unusable

	const struct oc_pack_model *m = c->model;
	struct xrt_conv_model_location loc;
	bool requested = false;
	uint64_t absent_since = 0;
	char err[XRT_CONV_DETAIL_MAX];
	enum oc_run_status r = OC_RUN_FAILED;

resolve:
	// 1. The model blob, through the host's store only (re-resolved after a
	//    backend fallback, whose variant may name another blob).
	for (;;) {
		if (is_cancelled(cancel_ctx)) {
			snprintf(why, sizeof(why), "%s: warm cancelled", m->name);
			set_state(p, c, XRT_CONV_STATE_INSTALLED, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
			return XRT_TIMEOUT;
		}
		const bool known = store_resolve(c->ref, &loc);
		if (known && loc.state == XRT_CONV_STATE_READY && loc.path[0] != '\0') {
			break;
		}
		if (known && loc.state == XRT_CONV_STATE_DOWNLOADING) {
			snprintf(why, sizeof(why), "%s: downloading", m->name);
			set_state(p, c, XRT_CONV_STATE_DOWNLOADING, XRT_CONV_FAIL_NONE, 0, loc.progress, why);
			absent_since = 0;
			oc_sleep_ms(OC_POLL_MS);
			continue;
		}
		if (known && loc.state == XRT_CONV_STATE_FAILED) {
			const uint32_t reason = loc.failure_reason != 0 ? loc.failure_reason : XRT_CONV_FAIL_MODEL;
			snprintf(why, sizeof(why), "%s: the model store could not provide the model", m->name);
			set_state(p, c, XRT_CONV_STATE_FAILED, reason, reason == XRT_CONV_FAIL_NETWORK ? 1u : 0u, -1.0f,
			          why);
			return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
		}
		// Not in the store: ask the host to fetch it (the host downloads,
		// verifies and stamps; this provider never touches the network).
		if (!requested) {
			requested = true;
			if (!store_request(c->ref)) {
				snprintf(why, sizeof(why), "%s: not in the model store and the download is not allowed",
				         m->name);
				set_state(p, c, XRT_CONV_STATE_ABSENT, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
				return XRT_ERROR_FEATURE_NOT_SUPPORTED;
			}
			snprintf(why, sizeof(why), "%s: download requested", m->name);
			set_state(p, c, XRT_CONV_STATE_DOWNLOADING, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
		}
		const uint64_t now = oc_now_ms();
		if (absent_since == 0) {
			absent_since = now;
		} else if (now - absent_since > OC_REQUEST_GRACE_MS) {
			// The host accepted the request but never started it (policy).
			snprintf(why, sizeof(why), "%s: the model store did not start the download", m->name);
			set_state(p, c, XRT_CONV_STATE_ABSENT, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
			return XRT_ERROR_FEATURE_NOT_SUPPORTED;
		}
		oc_sleep_ms(OC_POLL_MS);
	}

	// 2. Load. The blob path is the host's verified copy.
	snprintf(why, sizeof(why), "%s: loading (%s)", m->name, c->backend);
	set_state(p, c, XRT_CONV_STATE_WARMING, XRT_CONV_FAIL_NONE, 0, 0.1f, why);
	char cache[1200];
	const struct oc_ep_config epc = {
	    .dml_device = p->dml_device,
	    .coreml_units = p->coreml_units,
	    .coreml_cache_dir = c->ep == OC_EP_COREML ? coreml_cache_dir(p, cache, sizeof(cache)) : NULL,
	};
	const uint64_t load_t0 = oc_now_ms();
	r = oc_session_create(p->ort, loc.path, c->ep, &epc, p->threads, p->opt, &c->session, err, sizeof(err));
	const uint64_t load_ms = oc_now_ms() - load_t0;
	uint32_t taken = 0, total = 0, parts = 0;
	const bool reported = r == OC_RUN_OK && oc_session_ep_nodes(c->session, &taken, &total, &parts);
	if (r == OC_RUN_OK && c->ep == OC_EP_COREML) {
		if (reported) {
			fprintf(stderr,
			        "[open-conversion] %s: Core ML takes %u of %u nodes in %u partition(s); load %llu ms\n",
			        m->name, taken, total, parts, (unsigned long long)load_ms);
		}
		if (reported && taken == 0) {
			// Nothing ran on Core ML: the session is a CPU session in disguise.
			// Say so honestly (backend, licence) by moving to the CPU EP.
			snprintf(err, sizeof(err), "Core ML supports none of the model's operators");
			r = OC_RUN_FAILED;
		}
	}
	if (r == OC_RUN_OK && is_cancelled(cancel_ctx)) {
		release_session(p, c);
		snprintf(why, sizeof(why), "%s: warm cancelled", m->name);
		set_state(p, c, XRT_CONV_STATE_INSTALLED, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
		return XRT_TIMEOUT;
	}
	if (r == OC_RUN_OK) {
		snprintf(why, sizeof(why), "%s: first inference (%s)", m->name, c->backend);
		set_state(p, c, XRT_CONV_STATE_WARMING, XRT_CONV_FAIL_NONE, 0, 0.6f, why);
		r = warm_up_inference(p, c, err, sizeof(err));
	}
	if (r != OC_RUN_OK && r != OC_RUN_OUT_OF_MEMORY && c->ep != OC_EP_CPU) {
		// DirectML / Core ML could not start (driver, feature level, an operator,
		// a Core ML compile failure): the CPU execution provider serves the same model.
		release_session(p, c);
		fall_back_to_cpu(p, c, err);
		m = c->model;
		if (m != NULL && c->variant != NULL) {
			requested = false;
			absent_since = 0;
			goto resolve;
		}
		snprintf(err, sizeof(err), "no CPU variant in the model pack");
	}
	if (r != OC_RUN_OK) {
		release_session(p, c);
		const uint32_t reason = r == OC_RUN_OUT_OF_MEMORY ? XRT_CONV_FAIL_RESOURCES
		                        : r == OC_RUN_DEVICE_LOST ? XRT_CONV_FAIL_DEVICE_LOST
		                                                  : XRT_CONV_FAIL_MODEL;
		// Bounded: the detail is for a human and truncates by design (XRT_CONV_DETAIL_MAX).
		snprintf(why, sizeof(why), "%.40s: %.80s", m->name, err);
		set_state(p, c, XRT_CONV_STATE_FAILED, reason, reason == XRT_CONV_FAIL_DEVICE_LOST ? 1u : 0u, -1.0f,
		          why);
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}

	oc_lock(&p->lock);
	// 24 + 1 + 8 + 5 + 9 + NUL fits version[48].
	snprintf(c->version, sizeof(c->version), "%.24s@%.8s ort %.9s", m->name, m->version, oc_ort_version(p->ort));
	oc_unlock(&p->lock);
	if (c->ep == OC_EP_COREML && reported) {
		snprintf(why, sizeof(why), "%s: ready (%s, %u/%u nodes)", m->name, c->backend, taken, total);
	} else {
		snprintf(why, sizeof(why), "%s: ready (%s)", m->name, c->backend);
	}
	set_state(p, c, XRT_CONV_STATE_READY, XRT_CONV_FAIL_NONE, 0, 1.0f, why);
	return XRT_SUCCESS;
}

static xrt_result_t
oc_evict(struct xrt_conversion_plugin_instance *inst, uint32_t cap_id)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	struct oc_cap *c = p != NULL ? cap_get(p, cap_id) : NULL;
	if (c == NULL) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	const bool was_ready = state_of(p, c) == XRT_CONV_STATE_READY;
	release_session(p, c);
	if (was_ready) {
		char why[XRT_CONV_DETAIL_MAX];
		snprintf(why, sizeof(why), "%s: evicted", c->model != NULL ? c->model->name : c->task_name);
		set_state(p, c, XRT_CONV_STATE_EVICTED, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
	}
	return XRT_SUCCESS;
}


/*
 *
 * Inference on one colour image (both lanes).
 *
 */

/*! Run the cap's model on @p color and write the depth map into @p dst
 *  (w × h R32F, @p dst_stride bytes per row). Fills @p di. */
static xrt_result_t
infer(struct oc_provider *p,
      struct oc_cap *c,
      struct oc_stream *s,
      const struct xrt_conv_image *color,
      bool reset,
      float *dst,
      uint32_t dw,
      uint32_t dh,
      uint32_t dst_stride,
      uint64_t source_time_ns,
      struct xrt_conv_depth_info *di)
{
	const struct oc_pack_model *m = c->model;
	uint32_t gw = 0, gh = 0;
	inference_grid(m, color->width, color->height, &gw, &gh);
	const size_t n = (size_t)gw * gh;
	if (!grow(&p->chw, &p->chw_cap, n * 3u) || !grow(&p->map, &p->map_cap, n * 4u)) {
		return XRT_ERROR_ALLOCATION;
	}
	const uint32_t src_stride = color->row_stride != 0 ? color->row_stride : color->width * 4u;
	oc_prep_rgb((const uint8_t *)color->cpu_ptr, src_stride, color->width, color->height, is_bgr(color->format), gw,
	            gh, m->normalize ? m->mean : NULL, m->normalize ? m->std : NULL, p->chw);

	char err[XRT_CONV_DETAIL_MAX];
	uint32_t mw = 0, mh = 0;
	enum oc_run_status r;
	if (c->task == XRT_CONV_TASK_DEPTH_VIDEO && s != NULL) {
		const struct oc_temporal_io io = {m->t_image,     m->t_cache_in,  m->t_is_first, m->t_depth,
		                                  m->t_cache_out, m->cache_shape, m->cache_rank};
		r = oc_temporal_run(c->session, &s->temporal, &io, p->chw, gw, gh, reset || s->frames == 0, p->map,
		                    p->map_cap, &mw, &mh, err, sizeof(err));
	} else {
		r = oc_session_run_image(c->session, m->t_image, p->chw, gw, gh, m->t_depth, p->map, p->map_cap, &mw,
		                         &mh, err, sizeof(err));
	}
	if (r == OC_RUN_DEVICE_LOST) {
		// The runtime reads FAILED(DEVICE_LOST, retryable) and re-warms on its thread.
		char why[XRT_CONV_DETAIL_MAX];
		snprintf(why, sizeof(why), "%.40s: device lost: %.70s", m->name, err);
		set_state(p, c, XRT_CONV_STATE_FAILED, XRT_CONV_FAIL_DEVICE_LOST, 1, -1.0f, why);
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	if (r != OC_RUN_OK && r != OC_RUN_OUT_OF_MEMORY && c->ep == OC_EP_COREML) {
		// A Core ML run failed after a good warm: drop the session, move the cap
		// to the CPU EP (re-describe) and report the accelerator gone, as for GPU
		// loss: the runtime re-warms, and the re-warm builds a CPU session. No
		// silent retry inside a convert (discovery spec §2.4).
		char why[XRT_CONV_DETAIL_MAX];
		fprintf(stderr, "[open-conversion] %s: Core ML inference failed: %s\n", m->name, err);
		release_session(p, c);
		fall_back_to_cpu(p, c, err);
		snprintf(why, sizeof(why), "%.40s: Core ML run failed; re-warms on the CPU", m->name);
		set_state(p, c, XRT_CONV_STATE_FAILED, XRT_CONV_FAIL_DEVICE_LOST, 1, -1.0f, why);
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	if (r != OC_RUN_OK) {
		fprintf(stderr, "[open-conversion] %s: inference failed: %s\n", m->name, err);
		return r == OC_RUN_OUT_OF_MEMORY ? XRT_ERROR_ALLOCATION : XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}

	oc_resample_map(p->map, mw, mh, dst, dw, dh, dst_stride);
	float lo = 0.0f, hi = 0.0f, p05 = 0.0f, p95 = 0.0f;
	if (!oc_map_stats(dst, dw, dh, dst_stride, &lo, &hi, &p05, &p95)) {
		fprintf(stderr, "[open-conversion] %s: the model returned no finite sample\n", m->name);
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}

	// Honest units (frozen struct: every field set).
	memset(di, 0, sizeof(*di));
	di->kind = c->depth_kind; // affine disparity: larger = nearer, unknown scale + shift
	di->measure = c->depth_measure;
	di->intrinsics_source = XRT_CONV_INTRINSICS_NONE; // fx..cy stay 0: never invented
	di->value_scale = 1.0f;
	di->value_shift = 0.0f;
	di->near_value = 0.0f; // unknown: an affine reading has no depth to report
	di->far_value = 0.0f;
	di->invalid_sample = NAN; // every sample is valid
	di->range_min = lo;
	di->range_max = hi;
	di->range_p05 = p05;
	di->range_p95 = p95;
	di->source_width = color->width;
	di->source_height = color->height;
	di->temporal = c->task == XRT_CONV_TASK_DEPTH_VIDEO ? 1u : 0u;
	di->same_inference = 1;
	di->source_time_ns = source_time_ns;
	di->provider_frame_id = s != NULL ? s->frames : 0;
	return XRT_SUCCESS;
}


/*
 *
 * Stream lane.
 *
 */

static struct oc_stream *
stream_find_locked(struct oc_provider *p, uint64_t id)
{
	for (uint32_t i = 0; i < OC_MAX_STREAMS; i++) {
		if (p->streams[i].used && p->streams[i].id == id) {
			return &p->streams[i];
		}
	}
	return NULL;
}

static xrt_result_t
oc_stream_create(struct xrt_conversion_plugin_instance *inst,
                 uint32_t cap_id,
                 const struct xrt_conv_stream_info *info,
                 uint64_t *out_stream)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	struct oc_cap *c = p != NULL ? cap_get(p, cap_id) : NULL;
	if (c == NULL || info == NULL || out_stream == NULL || !COVERS(info, struct xrt_conv_stream_info, input_kind)) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	if (state_of(p, c) != XRT_CONV_STATE_READY || c->session == NULL) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED; // the runtime creates streams only on a READY cap
	}
	if ((info->input_kind & XRT_CONV_INPUT_MONO) == 0) {
		return XRT_ERROR_INPUT_UNSUPPORTED;
	}
	if (COVERS(info, struct xrt_conv_stream_info, outputs) && (info->outputs & ~XRT_CONV_OUT_DEPTH) != 0) {
		return XRT_ERROR_OUTPUT_UNSUPPORTED;
	}
	oc_lock(&p->lock);
	uint32_t live = 0;
	struct oc_stream *slot = NULL;
	for (uint32_t i = 0; i < OC_MAX_STREAMS; i++) {
		if (p->streams[i].used) {
			live += p->streams[i].cap_id == cap_id ? 1u : 0u;
		} else if (slot == NULL) {
			slot = &p->streams[i];
		}
	}
	if (slot == NULL || live >= c->max_streams) {
		oc_unlock(&p->lock);
		return XRT_ERROR_CLIENT_LIMIT_REACHED;
	}
	memset(slot, 0, sizeof(*slot));
	slot->used = true;
	slot->id = ++p->next_stream;
	slot->cap_id = cap_id;
	*out_stream = slot->id;
	oc_unlock(&p->lock);
	return XRT_SUCCESS;
}

static xrt_result_t
oc_stream_get_output_desc(struct xrt_conversion_plugin_instance *inst,
                          uint64_t stream,
                          uint32_t input_width,
                          uint32_t input_height,
                          struct xrt_conv_output_desc *out)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	if (p == NULL || out == NULL || input_width == 0 || input_height == 0) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	oc_lock(&p->lock);
	const struct oc_stream *s = stream_find_locked(p, stream);
	const struct oc_cap *c = s != NULL ? cap_get(p, s->cap_id) : NULL;
	const uint32_t edge = c != NULL ? output_long_edge(c->model) : 0u;
	oc_unlock(&p->lock);
	if (c == NULL || edge == 0) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	// The depth map keeps the input's aspect at the model's native long edge
	// (never above the input): the resolution the model actually resolves.
	uint32_t w = 0, h = 0;
	oc_fit_long_edge(input_width, input_height, edge, 1, true, &w, &h);
	SET(out, struct xrt_conv_output_desc, depth_width, w);
	SET(out, struct xrt_conv_output_desc, depth_height, h);
	SET(out, struct xrt_conv_output_desc, depth_format, XRT_CONV_FMT_R32_FLOAT);
	return XRT_SUCCESS;
}

static const struct xrt_conv_image *
find_color(const struct xrt_conv_image *images, const uint32_t *roles, uint32_t count)
{
	for (uint32_t i = 0; i < count && i < XRT_CONV_MAX_FRAME_IMAGES; i++) {
		if ((roles != NULL ? roles[i] : XRT_CONV_IMG_COLOR) == XRT_CONV_IMG_COLOR) {
			const struct xrt_conv_image *img = &images[i];
			if (img->memory_kind == XRT_CONV_MEM_CPU && img->cpu_ptr != NULL &&
			    is_colour_format(img->format) && img->width > 0 && img->height > 0) {
				return img;
			}
			return NULL;
		}
	}
	return NULL;
}

static xrt_result_t
oc_stream_convert(struct xrt_conversion_plugin_instance *inst,
                  uint64_t stream,
                  const struct xrt_conv_frame_in *in,
                  const struct xrt_conv_params *params,
                  struct xrt_conv_frame_out *out)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	if (p == NULL || in == NULL || out == NULL || !COVERS(in, struct xrt_conv_frame_in, images) ||
	    !COVERS(out, struct xrt_conv_frame_out, depth)) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	oc_lock(&p->lock);
	struct oc_stream *s = stream_find_locked(p, stream);
	oc_unlock(&p->lock);
	struct oc_cap *c = s != NULL ? cap_get(p, s->cap_id) : NULL;
	if (c == NULL) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	if (state_of(p, c) != XRT_CONV_STATE_READY || c->session == NULL) {
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE; // the runtime keeps the previous result
	}
	const struct xrt_conv_image *color = find_color(
	    in->images, COVERS(in, struct xrt_conv_frame_in, image_role) ? in->image_role : NULL, in->image_count);
	if (color == NULL) {
		return XRT_ERROR_INPUT_UNSUPPORTED;
	}
	const struct xrt_conv_image *dst = &out->depth;
	if (dst->memory_kind != XRT_CONV_MEM_CPU || dst->cpu_ptr == NULL || dst->format != XRT_CONV_FMT_R32_FLOAT ||
	    dst->width == 0 || dst->height == 0) {
		return XRT_ERROR_OUTPUT_UNSUPPORTED;
	}
	uint32_t frame_flags = 0;
	uint64_t source_time_ns = 0;
	if (params != NULL) {
		frame_flags = COVERS(params, struct xrt_conv_params, frame_flags) ? params->frame_flags : 0u;
		source_time_ns = COVERS(params, struct xrt_conv_params, source_time_ns) ? params->source_time_ns : 0u;
	}
	struct xrt_conv_depth_info di;
	const uint32_t dst_stride = dst->row_stride != 0 ? dst->row_stride : dst->width * 4u;
	// FRAME_RESET (cut / seek): drop temporal state BEFORE this frame.
	const xrt_result_t r = infer(p, c, s, color, (frame_flags & XRT_CONV_FRAME_RESET) != 0, (float *)dst->cpu_ptr,
	                             dst->width, dst->height, dst_stride, source_time_ns, &di);
	if (r != XRT_SUCCESS) {
		return r;
	}
	// CPU target, sync NONE: the write is complete when we return.
	SET(out, struct xrt_conv_frame_out, written, XRT_CONV_OUT_DEPTH);
	if (COVERS(out, struct xrt_conv_frame_out, depth_info)) {
		out->depth_info = di;
	}
	s->frames++;
	return XRT_SUCCESS;
}

static void
oc_stream_destroy(struct xrt_conversion_plugin_instance *inst, uint64_t stream)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	if (p == NULL) {
		return;
	}
	oc_lock(&p->lock);
	struct oc_stream *s = stream_find_locked(p, stream);
	struct oc_temporal *t = NULL;
	if (s != NULL) {
		t = s->temporal;
		memset(s, 0, sizeof(*s));
	}
	oc_unlock(&p->lock);
	oc_temporal_destroy(t);
}


/*
 *
 * Job lane (DEPTH_MONO): one image in, one RAW_IMAGE R32F blob out.
 *
 */

static xrt_result_t
oc_job_execute(struct xrt_conversion_plugin_instance *inst,
               uint32_t cap_id,
               const struct xrt_conv_job_in *in,
               const struct xrt_conv_params *params,
               const struct xrt_conv_job_control *ctl,
               struct xrt_conv_job_out *out)
{
	struct oc_provider *p = (struct oc_provider *)inst;
	struct oc_cap *c = p != NULL ? cap_get(p, cap_id) : NULL;
	if (c == NULL || c->task != XRT_CONV_TASK_DEPTH_MONO || in == NULL || ctl == NULL ||
	    !COVERS(in, struct xrt_conv_job_in, images) || !COVERS(ctl, struct xrt_conv_job_control, is_cancelled)) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	if (state_of(p, c) != XRT_CONV_STATE_READY || c->session == NULL) {
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	const struct xrt_conv_image *color = find_color(
	    in->images, COVERS(in, struct xrt_conv_job_in, image_role) ? in->image_role : NULL, in->image_count);
	if (color == NULL) {
		return XRT_ERROR_INPUT_UNSUPPORTED;
	}
	struct xrt_conv_blob_sink *sink = NULL;
	for (uint32_t i = 0; i < ctl->sink_count && ctl->sinks != NULL; i++) {
		if (ctl->sinks[i].role == XRT_CONV_OUT_DEPTH) {
			sink = &ctl->sinks[i];
			break;
		}
	}
	if (sink == NULL || sink->begin == NULL || sink->write == NULL) {
		return XRT_ERROR_OUTPUT_UNSUPPORTED;
	}
	bool (*cancelled)(void *) = ctl->is_cancelled != NULL ? ctl->is_cancelled : never_cancelled;
	if (ctl->progress != NULL) {
		ctl->progress(ctl->ctx, 0.0f, "infer");
	}
	if (cancelled(ctl->ctx)) {
		return XRT_SUCCESS; // cancelled: nothing written, the runtime reports it
	}
	uint32_t w = 0, h = 0;
	oc_fit_long_edge(color->width, color->height, output_long_edge(c->model), 1, true, &w, &h);
	float *img = (float *)malloc((size_t)w * h * sizeof(float));
	if (img == NULL) {
		return XRT_ERROR_ALLOCATION;
	}
	const uint64_t source_time_ns =
	    params != NULL && COVERS(params, struct xrt_conv_params, source_time_ns) ? params->source_time_ns : 0u;
	struct xrt_conv_depth_info di;
	xrt_result_t r = infer(p, c, NULL, color, true, img, w, h, w * 4u, source_time_ns, &di);
	if (r != XRT_SUCCESS || cancelled(ctl->ctx)) {
		free(img);
		return r;
	}
	if (ctl->progress != NULL) {
		ctl->progress(ctl->ctx, 0.9f, "write");
	}
	const uint64_t bytes = (uint64_t)w * h * sizeof(float);
	const bool ok = sink->begin(sink->ctx, XRT_CONV_BLOB_RAW_IMAGE, bytes) && sink->write(sink->ctx, img, bytes);
	free(img);
	if (!ok) {
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	if (out != NULL) {
		SET(out, struct xrt_conv_job_out, written, XRT_CONV_OUT_DEPTH);
		if (COVERS(out, struct xrt_conv_job_out, depth_info)) {
			// The blob is w × h: depth_info's source extent stays the input's.
			out->depth_info = di;
		}
	}
	if (ctl->progress != NULL) {
		ctl->progress(ctl->ctx, 1.0f, "done");
	}
	return XRT_SUCCESS;
}

static void
oc_destroy(struct xrt_conversion_plugin_instance *inst)
{
	// MAY NEVER BE CALLED (process exit, host kill): nothing here must be flushed.
	struct oc_provider *p = (struct oc_provider *)inst;
	if (p == NULL) {
		return;
	}
	oc_lock(&g_global_lock);
	if (g_provider == p) {
		g_provider = NULL;
	}
	oc_unlock(&g_global_lock);
	for (uint32_t i = 0; i < OC_MAX_STREAMS; i++) {
		oc_temporal_destroy(p->streams[i].temporal);
	}
	for (uint32_t i = 0; i < OC_CAP_COUNT; i++) {
		oc_session_destroy(p->caps[i].session);
	}
	// ONNX Runtime stays mapped and its environment alive: it does not support
	// being unloaded and re-initialised within one process.
	free(p->chw);
	free(p->map);
	free(p->pack);
	oc_mutex_destroy(&p->lock);
	free(p);
}


/*
 *
 * Negotiate: the only export.
 *
 */

static struct xrt_conversion_plugin_iface g_iface = {
    .struct_size = (uint32_t)sizeof(struct xrt_conversion_plugin_iface),
    .id = OC_ID,
    .display_name = "DisplayXR Open Conversion",
    .vendor = "DisplayXR",
    .version = OC_VERSION,
    .get_provider_status = oc_get_provider_status,
    .probe = oc_probe,
    .describe_cap = oc_describe_cap,
    .get_generation = oc_get_generation,
    .get_state = oc_get_state,
    .warm = oc_warm,
    .evict = oc_evict,
    .stream_create = oc_stream_create,
    .stream_get_output_desc = oc_stream_get_output_desc,
    .stream_convert = oc_stream_convert,
    .stream_destroy = oc_stream_destroy,
    .job_execute = oc_job_execute,
    .destroy = oc_destroy,
};

XRT_CONVERSION_PLUGIN_EXPORT xrt_result_t
xrtConversionPluginNegotiate(uint32_t runtime_api_version,
                             const struct xrt_conversion_plugin_host_iface *host,
                             struct xrt_conversion_plugin_iface **out_iface,
                             uint32_t *out_plugin_api_version)
{
	if (out_iface == NULL || out_plugin_api_version == NULL) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	*out_plugin_api_version = XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT;
	if (runtime_api_version != XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT) {
		*out_iface = NULL;
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	if (!g_global_lock_init) {
		// Negotiate runs once per process, on the loader's thread, before any other slot.
		oc_mutex_init(&g_global_lock);
		g_global_lock_init = true;
	}
	g_host = host; // valid until process end
	*out_iface = &g_iface;
	return XRT_SUCCESS;
}
