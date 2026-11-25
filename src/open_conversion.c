// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  DisplayXR open-model conversion provider (ADR-053, ProbeOrder 200).
 *
 * Three capabilities. Two are depth-only, and the runtime composes views from
 * them with its reference view synthesis (discovery spec §1.1, §6.2); the
 * third turns one photo into a 3D Gaussian splat (ADR-057):
 *
 *   cap 1  DEPTH_VIDEO  stream lane, TEMPORAL: a small streaming video-depth
 *                       model with an explicit per-stream temporal cache.
 *   cap 2  DEPTH_MONO   stream + job lanes: a small still-depth model.
 *   cap 3  SPLATS_MONO  job lane: photo -> 3DGS PLY, from whatever model the
 *                       packs name for "splats-mono" (none in the default
 *                       pack: a splat model is a separate, user-installed pack).
 *                       Setting MODEL picks between the pack's full model
 *                       (defaults.high), its small one (defaults.low) and,
 *                       on DirectML, its fp16 one (defaults.medium, when it
 *                       has a DirectML variant): auto = full on a GPU backend
 *                       with >= 16 GB of RAM; on DirectML fp16 when it is in
 *                       the store (>= 6 GB VRAM), else full with >= 10 GB VRAM;
 *                       among the models in the store (oc_rank.h).
 *
 * The splat capability runs where oc_rank.h ranks it: ONNX Runtime + DirectML
 * on a discrete non-Intel GPU, else OpenVINO (oc_ov.h) on an Intel Arc / Xe GPU,
 * else OpenVINO's CPU device, else ONNX Runtime's CPU execution provider.
 * Everything else runs through ONNX Runtime, loaded at
 * run time from the provider's own directory: the DirectML execution provider on Windows when a hardware
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
#include "oc_ov.h"
#include "oc_pack.h"
#include "oc_platform.h"
#include "oc_rank.h"
#include "oc_splat.h"

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
#define OC_VERSION "0.2.0"
#ifndef OC_BUILD_COMMIT
#define OC_BUILD_COMMIT "unknown" //!< the build's source revision (CMake passes the git describe)
#endif
#define OC_PACK_FILE "dxr-open-default.models.json"
#define OC_PACK_SUFFIX ".models.json"
#define OC_MAX_PACKS 4u
#define OC_PACK_LIST_SEP ';' //!< setting PACK: absolute paths separated by ';'

#define OC_CAP_VIDEO 1u  //!< DEPTH_VIDEO
#define OC_CAP_MONO 2u   //!< DEPTH_MONO
#define OC_CAP_SPLATS 3u //!< SPLATS_MONO
#define OC_CAP_COUNT 3u
#define OC_MAX_SPLAT_INPUT_EDGE 8192u //!< the runtime downscales above it; the splat model resamples to its grid
#define OC_ASSUMED_FOCAL_MM 30.0      //!< no focal given: a 30 mm full-frame equivalent, labelled ASSUMED
#define OC_SPLAT_FULL 0u              //!< splat_cand index: defaults["splats-mono"].high
#define OC_SPLAT_SMALL 1u             //!< splat_cand index: defaults["splats-mono"].low
#define OC_SPLAT_FP16 2u              //!< splat_cand index: .medium, when it is a third model with a DirectML variant
#define OC_SPLAT_CANDS 3u

#define OC_MAX_STREAMS_VIDEO 2u //!< each holds a temporal cache of ~160 MB (518 px model)
#define OC_MAX_STREAMS_MONO 4u
#define OC_MAX_STREAMS (OC_MAX_STREAMS_VIDEO + OC_MAX_STREAMS_MONO)
#define OC_MAX_INPUT_EDGE 2048u //!< the runtime downscales above it; the models run at ~518 px

#define OC_OPT_DEFAULT OC_OPT_EXTENDED
#define OC_POLL_MS 50u            //!< warm() polls the host and is_cancelled this often (≤ 250 ms)
#define OC_REQUEST_GRACE_MS 5000u //!< ABSENT this long after an accepted request = refused
/*! The store verifying a model (WARMING: it re-hashes a stored blob once per process)
 *  is progress, not a refusal. warm() gives up only when the store's state and
 *  progress stop moving for the longer of this ... */
#define OC_STORE_STALL_MIN_MS 120000u
//! ... and the model's bytes at this hashing floor (a slow disk), so a big model never times out early.
#define OC_STORE_VERIFY_FLOOR_BPS (8ull << 20)

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
	const struct oc_pack *pack;            //!< the pack that names model
	const struct oc_pack_model *model;     //!< NULL: no pack has a usable model for the task
	const struct oc_pack_variant *variant; //!< the blob variant for the EP below
	char ref[192];                         //!< "<pack>/<model>/<variant>"
	enum oc_ep ep;
	char ov_device[8];   //!< OC_EP_OPENVINO: "GPU" or "CPU"
	bool ov_provisional; //!< SPLATS_MONO on OpenVINO GPU, picked before OpenVINO listed its devices
	int dml_device;      //!< SPLATS_MONO on DirectML: its own adapter (oc_rank.h), not the depth caps'
	uint64_t dml_luid;
	uint64_t dml_vram_mb;
	bool dml_failed;    //!< SPLATS_MONO: DirectML failed in this process; ranked without it from now on
	bool ov_gpu_failed; //!< SPLATS_MONO: OpenVINO's GPU device failed; ranked without it from now on
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

	/* SPLATS_MONO: why the bound model was picked (MODEL), under `lock`. */
	bool pick_forced;    //!< MODEL=full / int4: never switched to the other model
	char pick_note[128]; //!< non-empty when the preferred model was absent and the other one used

	/* [S]-only. */
	struct oc_session *session;       //!< ONNX Runtime
	struct oc_ov_session *ov_session; //!< OpenVINO (the splat capability)
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
	char quality[16];
	char ep_setting[16];
	char ov_device_setting[8];   //!< "auto", "gpu", "cpu"
	char splat_model_setting[8]; //!< MODEL: "auto", "full", "fp16", "int4"
	uint64_t ram_mb;             //!< physical memory (oc_system_ram_mb), 0 = unknown
	char ov_dir[1100];           //!< the OpenVINO runtime folder
	char scratch[1100];          //!< the host's per-provider scratch dir; "" = none
	int threads;
	enum oc_graph_opt opt;
	enum oc_ort_log ort_log;
	enum oc_coreml_units coreml_units;
	char coreml_cache[1100]; //!< under the host's scratch dir; empty = no compiled-model cache
	bool ov_cache;           //!< keep OpenVINO's compiled models under the scratch dir
	bool diag_profile;       //!< DIAG_PROFILE: node placement + a profiled second run per splat job

	/* Dependencies. */
	struct oc_pack *packs[OC_MAX_PACKS]; //!< the default pack first, then the others by name
	uint32_t pack_count;                 //!< 0: none usable
	char pack_error[200];
	struct oc_ort *ort; //!< NULL until loaded; never unloaded
	enum oc_ort_status ort_status;
	char ort_detail[128];
	int dml_device;
	uint64_t dml_luid;
	bool coreml_cache_ready; //!< the cache directory exists and was swept (first use)
	enum oc_ep default_ep;   //!< the ONNX Runtime EP the depth capabilities use (choose_ep)

	/* OpenVINO: located in probe (a file check), loaded on the first warm that needs it. */
	bool ov_present;
	struct oc_ov *ov; //!< NULL until loaded; never unloaded
	enum oc_ov_status ov_status;
	char ov_detail[128];
	bool ov_devices_known;
	struct oc_ov_gpu ov_gpus[OC_RANK_MAX_OV_GPUS];
	uint32_t ov_gpu_count;
	char ov_gpu_device[16]; //!< the OpenVINO GPU device SPLATS_MONO compiles for ("GPU", "GPU.1")

	/* DXGI adapters (Windows; listed once in probe), for SPLATS_MONO's ranking. */
	struct oc_dxgi_adapter adapters[OC_RANK_MAX_ADAPTERS];
	uint32_t adapter_count;

	/* SPLATS_MONO's candidate models and which of their variants' blobs are in
	 * the store (refresh_splat_presence, outside `lock`; read under it). */
	const struct oc_pack *splat_pack[OC_SPLAT_CANDS];
	const struct oc_pack_model *splat_cand[OC_SPLAT_CANDS]; //!< [OC_SPLAT_FULL], [_SMALL], [_FP16] (may be NULL)
	bool splat_on_disk[OC_SPLAT_CANDS][OC_PACK_MAX_VARIANTS];

	/* [S]-only scratch, grown on demand. */
	float *chw;
	size_t chw_cap;
	float *map;
	size_t map_cap;
	float *gauss; //!< the splat model's five outputs, back to back
	size_t gauss_cap;
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

//! The bytes the store hands out for a variant: its model file and every data file.
static uint64_t
variant_bytes(const struct oc_pack_variant *v)
{
	uint64_t n = v != NULL ? v->file.size : 0;
	for (uint32_t i = 0; v != NULL && i < v->data_count; i++) {
		n += v->data[i].size;
	}
	return n;
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
	return ep == OC_EP_DML ? "dml" : ep == OC_EP_COREML ? "coreml" : ep == OC_EP_OPENVINO ? "openvino" : "cpu";
}

static const char *
ep_display_name(enum oc_ep ep)
{
	return ep == OC_EP_DML        ? "DirectML"
	       : ep == OC_EP_COREML   ? "Core ML"
	       : ep == OC_EP_OPENVINO ? "OpenVINO"
	                              : "CPU";
}

static bool
is_splats(const struct oc_cap *c)
{
	return c->task == XRT_CONV_TASK_SPLATS_MONO;
}

static bool
has_session(const struct oc_cap *c)
{
	return c->session != NULL || c->ov_session != NULL;
}

/*! The model `defaults[task][quality]` names, in the first pack that names one. */
static const struct oc_pack_model *
default_model(const struct oc_provider *p, const char *task, const struct oc_pack **out_pack)
{
	for (uint32_t i = 0; i < p->pack_count; i++) {
		const struct oc_pack_model *m = oc_pack_default_model(p->packs[i], task, p->quality);
		if (m != NULL) {
			*out_pack = p->packs[i];
			return m;
		}
	}
	*out_pack = NULL;
	return NULL;
}

/*! The model `defaults[task][tier]` names, in the first pack that names one. */
static const struct oc_pack_model *
tier_model(const struct oc_provider *p, const char *task, const char *tier, const struct oc_pack **out_pack)
{
	for (uint32_t i = 0; i < p->pack_count; i++) {
		const struct oc_pack_model *m = oc_pack_default_model(p->packs[i], task, tier);
		if (m != NULL) {
			*out_pack = p->packs[i];
			return m;
		}
	}
	*out_pack = NULL;
	return NULL;
}

/*! Does @p ep (with the cap's OpenVINO device) run on a GPU? */
static bool
backend_is_gpu(const struct oc_provider *p, const struct oc_cap *c, enum oc_ep ep)
{
	(void)p;
	return ep == OC_EP_DML || ep == OC_EP_COREML || (ep == OC_EP_OPENVINO && strcmp(c->ov_device, "GPU") == 0);
}

/*! The variant bind_cap_locked would take for @p ep: the accelerator's, else the CPU one. */
static const struct oc_pack_variant *
variant_for(const struct oc_pack_model *m, enum oc_ep ep)
{
	const struct oc_pack_variant *v = oc_pack_variant_for_ep(m, ep_pack_name(ep));
	return v != NULL ? v : oc_pack_variant_for_ep(m, "cpu");
}

/*! Is the blob of candidate @p k's variant for @p ep in the store (last refresh)? */
static bool
splat_present_locked(const struct oc_provider *p, uint32_t k, enum oc_ep ep)
{
	const struct oc_pack_model *m = p->splat_cand[k];
	const struct oc_pack_variant *v = m != NULL ? variant_for(m, ep) : NULL;
	return v != NULL && p->splat_on_disk[k][v - m->variants];
}

//! The candidate index of @p m (OC_SPLAT_*).
static uint32_t
splat_index(enum oc_splat_model m)
{
	return m == OC_SM_FULL ? OC_SPLAT_FULL : m == OC_SM_FP16 ? OC_SPLAT_FP16 : OC_SPLAT_SMALL;
}

/*! Is the fp16 candidate's DirectML blob in the store (last refresh)? Caller holds `lock`. */
static bool
splat_fp16_ready_locked(const struct oc_provider *p)
{
	return p->splat_cand[OC_SPLAT_FP16] != NULL && splat_present_locked(p, OC_SPLAT_FP16, OC_EP_DML);
}

/*!
 * SPLATS_MONO's model (setting MODEL): "full" = the model defaults["splats-mono"].high
 * names, "int4" = the one .low names, "fp16" = the one .medium names when it is a third
 * model with a DirectML variant (DirectML only; elsewhere fp16 means full). auto =
 * oc_rank_auto_model: on DirectML the fp16 model when its blob is in the store and the
 * adapter has >= 6 GB, else full with >= 10 GB; on another GPU backend (OpenVINO's GPU
 * device, Core ML) full; both with >= 16 GB of RAM; else int4. Only among the models whose
 * blob is in the store: the preferred one absent = the next present one (fp16 only on
 * DirectML; said in pick_note and the INFO line). None present = the preferred one, which
 * then reports what it needs (a local build, a download) the usual way. A forced MODEL never
 * switches. Caller holds `lock` or is in probe.
 */
static const struct oc_pack_model *
pick_splat_model_locked(struct oc_provider *p, struct oc_cap *c, enum oc_ep ep, const struct oc_pack **out_pack)
{
	c->pick_note[0] = '\0';
	c->pick_forced = false;
	const struct oc_pack_model *full = p->splat_cand[OC_SPLAT_FULL];
	const struct oc_pack_model *small = p->splat_cand[OC_SPLAT_SMALL];
	const bool fp16_here = ep == OC_EP_DML && p->splat_cand[OC_SPLAT_FP16] != NULL;
	if ((full == NULL || small == NULL || full == small) && !fp16_here) {
		// One model (or none) for the task: nothing to choose.
		const uint32_t k = full != NULL ? OC_SPLAT_FULL : OC_SPLAT_SMALL;
		*out_pack = p->splat_pack[k];
		return p->splat_cand[k];
	}
	enum oc_splat_model want;
	if (strcmp(p->splat_model_setting, "full") == 0) {
		want = OC_SM_FULL;
		c->pick_forced = true;
	} else if (strcmp(p->splat_model_setting, "int4") == 0) {
		want = OC_SM_INT4;
		c->pick_forced = true;
	} else if (strcmp(p->splat_model_setting, "fp16") == 0) {
		want = fp16_here ? OC_SM_FP16 : OC_SM_FULL;
		c->pick_forced = true;
	} else {
		// The backend the full model would really run on: a model without a variant for
		// the accelerator runs on the CPU (bind_cap_locked), and that is not a GPU.
		const enum oc_ep full_ep =
		    full != NULL && oc_pack_variant_for_ep(full, ep_pack_name(ep)) != NULL ? ep : OC_EP_CPU;
		const bool dml = ep == OC_EP_DML;
		want = oc_rank_auto_model(dml || backend_is_gpu(p, c, full_ep), p->ram_mb, dml, c->dml_vram_mb,
		                          fp16_here && splat_fp16_ready_locked(p));
		if (want == OC_SM_FULL && dml && full_ep != OC_EP_DML) {
			want = OC_SM_INT4; // no DirectML variant of the full model: it would run on the CPU
		}
	}
	uint32_t k = splat_index(want);
	if (p->splat_cand[k] == NULL) {
		k = full != NULL ? OC_SPLAT_FULL : OC_SPLAT_SMALL;
	}
	if (!c->pick_forced && !splat_present_locked(p, k, ep)) {
		// The next present model, in preference order (fp16 only on DirectML).
		static const uint32_t order[3] = {OC_SPLAT_FP16, OC_SPLAT_FULL, OC_SPLAT_SMALL};
		for (uint32_t i = 0; i < 3; i++) {
			const uint32_t o = order[i];
			if (o == k || p->splat_cand[o] == NULL || (o == OC_SPLAT_FP16 && !fp16_here) ||
			    !splat_present_locked(p, o, ep)) {
				continue;
			}
			snprintf(c->pick_note, sizeof(c->pick_note), "%.30s is not in the store; using %.30s",
			         p->splat_cand[k] != NULL ? p->splat_cand[k]->name : "the preferred model",
			         p->splat_cand[o]->name);
			k = o;
			break;
		}
	}
	*out_pack = p->splat_pack[k];
	return p->splat_cand[k];
}

/*! Re-read which splat candidate blobs are in the store (a stat each, through the
 *  host). Not under `lock`: the host is called. */
static void
refresh_splat_presence(struct oc_provider *p)
{
	bool on_disk[OC_SPLAT_CANDS][OC_PACK_MAX_VARIANTS];
	memset(on_disk, 0, sizeof(on_disk));
	for (uint32_t k = 0; k < OC_SPLAT_CANDS; k++) {
		const struct oc_pack_model *m = p->splat_cand[k];
		for (uint32_t i = 0; m != NULL && i < m->variant_count; i++) {
			char ref[192];
			oc_pack_ref(p->splat_pack[k], m, &m->variants[i], ref, sizeof(ref));
			struct xrt_conv_model_location loc;
			// On disk (verified or not) or on its way: anything but ABSENT / FAILED.
			on_disk[k][i] =
			    store_resolve(ref, &loc) &&
			    (loc.state == XRT_CONV_STATE_READY || loc.state == XRT_CONV_STATE_INSTALLED ||
			     loc.state == XRT_CONV_STATE_WARMING || loc.state == XRT_CONV_STATE_DOWNLOADING);
		}
	}
	oc_lock(&p->lock);
	memcpy(p->splat_on_disk, on_disk, sizeof(on_disk));
	oc_unlock(&p->lock);
}

/*! Describe-time fields for a cap on an EP, from its pack model. Caller holds `lock` or is in probe. */
static void
bind_cap_locked(struct oc_provider *p, struct oc_cap *c, enum oc_ep ep)
{
	const struct oc_pack_model *prev = c->model;
	c->model = NULL;
	c->variant = NULL;
	c->pack = NULL;
	c->ref[0] = '\0';
	if (ep == OC_EP_OPENVINO && (!is_splats(c) || !p->ov_present)) {
		// OpenVINO serves the splat capability only (the depth capabilities
		// keep their ONNX Runtime paths, including the temporal binding).
		ep = p->default_ep;
	}
	const struct oc_pack *pk = NULL;
	const struct oc_pack_model *m =
	    is_splats(c) ? pick_splat_model_locked(p, c, ep, &pk) : default_model(p, c->task_name, &pk);
	const struct oc_pack_variant *v = NULL;
	if (m != NULL) {
		v = oc_pack_variant_for_ep(m, ep_pack_name(ep));
		if (v == NULL && ep != OC_EP_CPU) {
			// No variant for the accelerator (the pack decides, per model, which
			// execution providers a blob is acceptable on): this cap runs on the CPU.
			if (is_splats(c) && m != prev) {
				fprintf(stderr,
				        "[open-conversion] splats-mono: %s has no %s variant in its pack; it runs on "
				        "onnxruntime-cpu\n",
				        m->name, ep_display_name(ep));
			}
			ep = OC_EP_CPU;
			v = oc_pack_variant_for_ep(m, "cpu");
		}
	}
	c->ep = ep;
	const bool ov_gpu = strcmp(c->ov_device, "GPU") == 0;
	oc_copy_str(c->backend, sizeof(c->backend),
	            ep == OC_EP_DML        ? "onnxruntime-dml"
	            : ep == OC_EP_COREML   ? "onnxruntime-coreml"
	            : ep == OC_EP_OPENVINO ? (ov_gpu ? "openvino-gpu" : "openvino-cpu")
	                                   : "onnxruntime-cpu");
	c->device_luid = ep != OC_EP_DML ? 0 : is_splats(c) ? c->dml_luid : p->dml_luid;
	if (is_splats(c) && m != NULL && m != prev) {
		fprintf(stderr, "[open-conversion] splats-mono: MODEL=%s -> %s (%s, %llu MB RAM)%s%s\n",
		        p->splat_model_setting, m->name, c->backend, (unsigned long long)p->ram_mb,
		        c->pick_note[0] != '\0' ? "; " : "", c->pick_note);
	}
	if (m == NULL || v == NULL) {
		c->licence_class = XRT_CONV_LICENCE_UNKNOWN;
		c->commercial_ok = 0;
		c->depth_kind = XRT_CONV_DEPTH_AFFINE_DISPARITY; // what the default pack's models produce
		c->depth_measure = XRT_CONV_MEASURE_PLANAR;
		return;
	}
	c->pack = pk;
	c->model = m;
	c->variant = v;
	oc_pack_ref(pk, m, v, c->ref, sizeof(c->ref));
	c->depth_kind = depth_kind_from(m->depth_kind);
	c->depth_measure = strcmp(m->measure, "ray") == 0 ? XRT_CONV_MEASURE_RAY : XRT_CONV_MEASURE_PLANAR;
	// Commercial = AND over the model, its dependencies and the inference runtime
	// (ONNX Runtime: MIT; OpenVINO: Apache-2.0). The pack's own flag is the AND
	// over all its models, a summary the validator checks; one capability's
	// class is its own model's, so a research-only pack never taints another
	// pack's models. DirectML ships under Microsoft's redistributable licence:
	// commercial use is allowed, but it is not an open licence, so a
	// DirectML-backed capability is COMMERCIAL, a CPU one PERMISSIVE. The Core ML
	// execution provider is part of ONNX Runtime (MIT) and drives the operating
	// system's own Core ML framework, which nothing redistributes: a Core ML-backed
	// capability stays PERMISSIVE. OpenVINO's GPU device drives the system's own
	// OpenCL driver, which nothing redistributes either: PERMISSIVE.
	const bool commercial = m->commercial;
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
		snprintf(why, why_size, "no model pack has a %s model for this backend", c->task_name);
		return false;
	}
	if (is_splats(c)) {
		if (!m->fixed || m->t_image[0] == '\0' || m->t_disparity_factor[0] == '\0' || m->t_means[0] == '\0' ||
		    m->t_scales[0] == '\0' || m->t_quats[0] == '\0' || m->t_colors[0] == '\0' ||
		    m->t_opacities[0] == '\0') {
			snprintf(why, why_size, "pack model %s lacks a fixed grid or the splat tensor names", m->name);
			return false;
		}
		if (strcmp(m->splat_format, "sharp-projective") != 0 || strcmp(m->coords, "opencv") != 0) {
			snprintf(why, why_size, "pack model %.40s: only splat_format sharp-projective, coords opencv",
			         m->name);
			return false;
		}
		return true;
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

/*!
 * The splat capability's backend (oc_rank.h): DirectML on a discrete non-Intel GPU,
 * else OpenVINO's GPU device when it is an Intel Arc / Xe-class GPU, else OpenVINO's
 * CPU device, else ONNX Runtime's CPU EP; explicit settings win. Sets the cap's
 * OpenVINO device and DirectML adapter and logs one line saying what it resolved to
 * and why (plus one when a requested backend is not available). DirectML is left
 * out once it failed in this process (sticky). Caller holds `lock` or is in probe.
 */
static enum oc_ep
rank_splat_locked(struct oc_provider *p, struct oc_cap *c)
{
	const bool no_dml = c->dml_failed;
	const struct oc_pack_model *full = p->splat_cand[OC_SPLAT_FULL];
	const struct oc_pack_model *fp16 = p->splat_cand[OC_SPLAT_FP16];
	const struct oc_splat_env env = {
	    .ep = no_dml ? "auto" : p->ep_setting,
	    .ov_device = p->ov_device_setting,
	    .model = p->splat_model_setting,
	    .ram_mb = p->ram_mb,
	    .dml_available = !no_dml && oc_ort_has_dml(p->ort),
	    .coreml_available = oc_ort_has_coreml(p->ort),
	    .adapters = p->adapters,
	    .adapter_count = p->adapter_count,
	    .full_on_dml = full != NULL && oc_pack_variant_for_ep(full, "dml") != NULL,
	    .fp16_on_dml = fp16 != NULL,
	    .fp16_ready = splat_fp16_ready_locked(p),
	    .ov_present = p->ov_present,
	    .ov_known = p->ov_devices_known || c->ov_gpu_failed,
	    .ov_gpus = p->ov_gpus,
	    .ov_gpu_count = c->ov_gpu_failed ? 0u : p->ov_gpu_count,
	};
	struct oc_splat_choice ch;
	oc_rank_splat(&env, &ch);
	if (ch.fallback[0] != '\0') {
		fprintf(stderr, "[open-conversion] splats-mono: %s\n", ch.fallback);
	}
	fprintf(stderr, "[open-conversion] splats-mono: %s%s -> %s; model %s (MODEL=%s, %llu MB RAM)\n",
	        no_dml ? "DirectML failed, re-ranked: " : "", ch.why, oc_rank_backend_name(ch.backend),
	        oc_rank_model_name(ch.model), p->splat_model_setting, (unsigned long long)p->ram_mb);
	c->ov_provisional = ch.provisional;
	c->dml_device = -1;
	c->dml_luid = 0;
	c->dml_vram_mb = 0;
	oc_copy_str(p->ov_gpu_device, sizeof(p->ov_gpu_device), ch.ov_gpu >= 0 ? p->ov_gpus[ch.ov_gpu].device : "GPU");
	switch (ch.backend) {
	case OC_SB_DML:
		c->dml_device = p->adapters[ch.adapter].index;
		c->dml_luid = p->adapters[ch.adapter].luid;
		c->dml_vram_mb = p->adapters[ch.adapter].dedicated_mb;
		return OC_EP_DML;
	case OC_SB_OV_GPU: oc_copy_str(c->ov_device, sizeof(c->ov_device), "GPU"); return OC_EP_OPENVINO;
	case OC_SB_OV_CPU: oc_copy_str(c->ov_device, sizeof(c->ov_device), "CPU"); return OC_EP_OPENVINO;
	case OC_SB_COREML: return OC_EP_COREML;
	default: return OC_EP_CPU;
	}
}

/*! Readiness of a cold cap from the store, without loading anything. */
static void
initial_state(struct oc_provider *p, struct oc_cap *c)
{
	char why[XRT_CONV_DETAIL_MAX];
	if (c->ep == OC_EP_OPENVINO) {
		// Bound to OpenVINO only while it is there (probe: the files; warm: the
		// load). A failed load already moved the capability to ONNX Runtime.
	} else if (p->ort_status == OC_ORT_ABSENT) {
		set_state(p, c, XRT_CONV_STATE_ABSENT, XRT_CONV_FAIL_NONE, 0, -1.0f, p->ort_detail);
		return;
	} else if (p->ort_status == OC_ORT_TOO_OLD) {
		// Present but older than our floor: VERSION, never MODEL ("too old", not "missing").
		set_state(p, c, XRT_CONV_STATE_FAILED, XRT_CONV_FAIL_VERSION, 1, -1.0f, p->ort_detail);
		return;
	} else if (p->ort_status != OC_ORT_OK) {
		set_state(p, c, XRT_CONV_STATE_FAILED, XRT_CONV_FAIL_LOAD, 0, -1.0f, p->ort_detail);
		return;
	}
	if (p->pack_count == 0) {
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

/*! Load one pack into the next slot; false (and a reason in pack_error) when unusable. */
static bool
load_pack(struct oc_provider *p, const char *path)
{
	if (p->pack_count >= OC_MAX_PACKS) {
		return false;
	}
	struct oc_pack *pk = (struct oc_pack *)calloc(1, sizeof(*pk));
	if (pk == NULL) {
		return false;
	}
	char err[160] = "";
	if (!oc_pack_load(path, pk, err, sizeof(err))) {
		snprintf(p->pack_error, sizeof(p->pack_error), "model pack unusable: %s", err);
		fprintf(stderr, "[open-conversion] %s\n", p->pack_error);
		free(pk);
		return false;
	}
	for (uint32_t i = 0; i < p->pack_count; i++) {
		if (strcmp(p->packs[i]->pack, pk->pack) == 0) {
			fprintf(stderr, "[open-conversion] pack %s listed twice (%s); the first wins\n", pk->pack,
			        path);
			free(pk);
			return false;
		}
	}
	p->packs[p->pack_count++] = pk;
	return true;
}

/*!
 * The packs: setting PACK (absolute paths separated by ';'), else every
 * "*.models.json" next to the binary, the default pack first. A research-only
 * model (a splat model, say) ships as its own pack, so the default pack stays
 * commercial-clean; the host's store serves every pack the registration lists.
 */
static void
load_packs(struct oc_provider *p)
{
	char buf[OC_MAX_PACKS * 1100];
	snprintf(p->pack_error, sizeof(p->pack_error), "no model pack found");
	if (setting("PACK", buf, sizeof(buf))) {
		char *item = buf;
		while (item != NULL && *item != '\0') {
			char *next = strchr(item, OC_PACK_LIST_SEP);
			if (next != NULL) {
				*next++ = '\0';
			}
			if (*item != '\0') {
				(void)load_pack(p, item);
			}
			item = next;
		}
		return;
	}
	char path[1200];
	snprintf(path, sizeof(path), "%.1000s%c%s", p->module_dir, OC_PATH_SEP, OC_PACK_FILE);
	(void)load_pack(p, path);
	static char names[16][128];
	const uint32_t n = oc_list_dir(p->module_dir, OC_PACK_SUFFIX, &names[0][0], sizeof(names[0]), 16);
	for (uint32_t i = 0; i < n; i++) {
		if (strcmp(names[i], OC_PACK_FILE) != 0) {
			snprintf(path, sizeof(path), "%.1000s%c%.127s", p->module_dir, OC_PATH_SEP, names[i]);
			(void)load_pack(p, path);
		}
	}
}

//! Is the OpenVINO runtime there (its C API library)? A file check only.
static bool
ov_located(struct oc_provider *p)
{
#if defined(_WIN32)
	const char *lib = "openvino_c.dll";
#elif defined(__APPLE__)
	const char *lib = "libopenvino_c.dylib";
#else
	const char *lib = "libopenvino_c.so." OC_OV_SOVERSION;
#endif
	char path[1300];
	snprintf(path, sizeof(path), "%s%c%s", p->ov_dir, OC_PATH_SEP, lib);
	const bool there = oc_file_exists(path);
	oc_lock(&p->lock);
	p->ov_status = there ? OC_OV_OK : OC_OV_ABSENT;
	snprintf(p->ov_detail, sizeof(p->ov_detail), there ? "OpenVINO present (not loaded yet)" : "no OpenVINO (%s)",
	         p->ov_dir);
	oc_unlock(&p->lock);
	return there;
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
	if (setting("EP", buf, sizeof(buf)) && (strcmp(buf, "cpu") == 0 || strcmp(buf, "dml") == 0 ||
	                                        strcmp(buf, "coreml") == 0 || strcmp(buf, "openvino") == 0)) {
		oc_copy_str(p->ep_setting, sizeof(p->ep_setting), buf);
	}
	oc_copy_str(p->ov_device_setting, sizeof(p->ov_device_setting), "auto");
	if (setting("OV_DEVICE", buf, sizeof(buf)) && (strcmp(buf, "gpu") == 0 || strcmp(buf, "cpu") == 0)) {
		oc_copy_str(p->ov_device_setting, sizeof(p->ov_device_setting), buf);
	}
	oc_copy_str(p->splat_model_setting, sizeof(p->splat_model_setting), "auto");
	if (setting("MODEL", buf, sizeof(buf)) &&
	    (strcmp(buf, "full") == 0 || strcmp(buf, "int4") == 0 || strcmp(buf, "fp16") == 0)) {
		oc_copy_str(p->splat_model_setting, sizeof(p->splat_model_setting), buf);
	}
	p->ram_mb = oc_system_ram_mb();
	// CPU + GPU by default: the measured configuration. "all" / "ane" let Core ML
	// use the Neural Engine, whose first-use compile runs in a system service
	// that a warm cannot cancel, and which computes in fp16.
	p->coreml_units = OC_COREML_GPU;
	if (setting("COREML_UNITS", buf, sizeof(buf))) {
		p->coreml_units = strcmp(buf, "all") == 0   ? OC_COREML_ALL
		                  : strcmp(buf, "ane") == 0 ? OC_COREML_ANE
		                  : strcmp(buf, "cpu") == 0 ? OC_COREML_CPU
		                                            : OC_COREML_GPU;
	}
	// Core ML compiles the model on the first session; the compiled model is kept
	// in the host's per-provider scratch directory (never the model store), keyed
	// by the ONNX Runtime version below it. No scratch directory = no cache.
	p->coreml_cache[0] = '\0';
	p->scratch[0] = '\0';
	if (host_path(XRT_CONV_PATH_SCRATCH, buf, sizeof(buf))) {
		oc_copy_str(p->scratch, sizeof(p->scratch), buf);
	}
	if (!(setting("COREML_CACHE", buf, sizeof(buf)) && strtoul(buf, NULL, 10) == 0)) {
		oc_copy_str(p->coreml_cache, sizeof(p->coreml_cache), p->scratch);
	}
	// OpenVINO compiles a model for its device on the first session; the compiled
	// blob is kept the same way (under the scratch dir, keyed by the OpenVINO
	// version below it), so a later warm imports it in seconds.
	p->ov_cache = !(setting("OV_CACHE", buf, sizeof(buf)) && strtoul(buf, NULL, 10) == 0);
	p->ort_log = OC_ORT_LOG_WARNING;
	if (setting("ORT_LOG", buf, sizeof(buf))) {
		p->ort_log = strcmp(buf, "verbose") == 0 ? OC_ORT_LOG_VERBOSE
		             : strcmp(buf, "info") == 0  ? OC_ORT_LOG_INFO
		                                         : OC_ORT_LOG_WARNING;
	}
	p->diag_profile = setting("DIAG_PROFILE", buf, sizeof(buf)) && strtoul(buf, NULL, 10) != 0;
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
	if (setting("OPENVINO", buf, sizeof(buf))) {
		oc_copy_str(p->ov_dir, sizeof(p->ov_dir), buf);
	} else {
		snprintf(p->ov_dir, sizeof(p->ov_dir), "%s%c%s", p->module_dir, OC_PATH_SEP, oc_ov_dir_name());
	}

	// The packs (small JSON files, read and closed here).
	load_packs(p);

	// The inference runtime: mapping it is not a model load, and probe must be
	// able to say ABSENT / FAILED(VERSION) up front. OpenVINO is only located
	// here (a file check): its first load and device query happen in warm.
	load_runtime(p);
	p->default_ep = p->ort != NULL ? choose_ep(p) : OC_EP_CPU;
	p->ov_present = ov_located(p);
	p->adapter_count = oc_dxgi_list_adapters(p->adapters, OC_RANK_MAX_ADAPTERS);
	for (uint32_t i = 0; i < p->adapter_count; i++) {
		const struct oc_dxgi_adapter *a = &p->adapters[i];
		fprintf(stderr, "[open-conversion] DXGI adapter %d: '%s' vendor 0x%04x, %llu MB dedicated%s%s\n",
		        a->index, a->name, a->vendor_id, (unsigned long long)a->dedicated_mb,
		        a->software ? ", software" : "", oc_rank_dxgi_discrete(a) ? ", discrete" : "");
	}
	p->splat_cand[OC_SPLAT_FULL] = tier_model(p, "splats-mono", "high", &p->splat_pack[OC_SPLAT_FULL]);
	p->splat_cand[OC_SPLAT_SMALL] = tier_model(p, "splats-mono", "low", &p->splat_pack[OC_SPLAT_SMALL]);
	// The fp16 model: .medium, when it is a third model and DirectML runs it (it has no other variant).
	p->splat_cand[OC_SPLAT_FP16] = tier_model(p, "splats-mono", "medium", &p->splat_pack[OC_SPLAT_FP16]);
	if (p->splat_cand[OC_SPLAT_FP16] == p->splat_cand[OC_SPLAT_FULL] ||
	    p->splat_cand[OC_SPLAT_FP16] == p->splat_cand[OC_SPLAT_SMALL] ||
	    (p->splat_cand[OC_SPLAT_FP16] != NULL &&
	     oc_pack_variant_for_ep(p->splat_cand[OC_SPLAT_FP16], "dml") == NULL)) {
		p->splat_cand[OC_SPLAT_FP16] = NULL;
		p->splat_pack[OC_SPLAT_FP16] = NULL;
	}
	refresh_splat_presence(p);

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
	struct oc_cap *g = &p->caps[2];
	g->cap_id = OC_CAP_SPLATS;
	g->task = XRT_CONV_TASK_SPLATS_MONO;
	g->task_name = "splats-mono";
	g->max_streams = 0; // job lane only
	for (uint32_t i = 0; i < OC_CAP_COUNT; i++) {
		struct oc_cap *c = &p->caps[i];
		c->state = XRT_CONV_STATE_UNKNOWN;
		c->progress = -1.0f;
		bind_cap_locked(p, c, is_splats(c) ? rank_splat_locked(p, c) : p->default_ep);
		initial_state(p, c);
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
	if (is_splats(c)) {
		const struct oc_pack_model *m = c->model;
		C(cap_id, c->cap_id);
		C(task, c->task);
		C(lanes, XRT_CONV_LANE_JOB);
		C(input_kinds, XRT_CONV_INPUT_MONO);
		C(outputs, XRT_CONV_OUT_SPLAT_BLOB);
		C(flags, XRT_CONV_CAP_USES_MODEL_STORE | XRT_CONV_CAP_NEEDS_NETWORK_TO_WARM);
		C(import_memory_kinds, XRT_CONV_MEM_CPU);
		C(import_sync_kinds, 1u << XRT_CONV_SYNC_NONE);
		C(input_formats, (1u << XRT_CONV_FMT_RGBA8_UNORM) | (1u << XRT_CONV_FMT_RGBA8_SRGB) |
		                     (1u << XRT_CONV_FMT_BGRA8_UNORM) | (1u << XRT_CONV_FMT_BGRA8_SRGB));
		C(depth_output_format, 0u);
		C(min_views, 0u);
		C(max_views, 0u);
		C(max_streams, 0u);
		C(max_concurrent, 1u);
		C(max_input_edge, OC_MAX_SPLAT_INPUT_EDGE);
		C(preferred_input_edge, m != NULL ? (m->width > m->height ? m->width : m->height) : 0u);
		C(depth_kind, XRT_CONV_DEPTH_UNKNOWN);
		C(depth_measure, 0u);
		// GIVEN when the job carries a focal length, else a labelled 30 mm equivalent.
		C(intrinsics_source, XRT_CONV_INTRINSICS_ASSUMED);
		C(splat_formats, 1u << XRT_CONV_BLOB_PLY_3DGS);
		C(licence_class, c->licence_class);
		C(commercial_ok, c->commercial_ok);
		S(licence_spdx, m != NULL ? m->spdx : "");
		S(model_id, m != NULL ? m->name : "");
		S(model_version, m != NULL ? m->version : "");
		S(backend, c->backend);
		S(display_name, "Open photo to splats");
		C(device_luid, 0ull);
		C(est_warmup_ms, 0u);
		C(est_latency_ms, 0u);
		C(est_vram_mb, 0u);
		C(est_ram_mb, 0u);
		oc_unlock(&p->lock);
		return;
	}
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
	if (c->ov_session != NULL) {
		oc_ov_session_destroy(c->ov_session);
		c->ov_session = NULL;
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

/*! The next backend after @p c's current one failed: OpenVINO GPU → OpenVINO
 *  CPU → ONNX Runtime CPU (re-described each step). Sticky for the process. */
static void
fall_back(struct oc_provider *p, struct oc_cap *c, const char *why)
{
	if (is_splats(c) && c->ep == OC_EP_DML) {
		// DirectML could not start or a run failed: the rest of the ranking without it
		// (OpenVINO, else ONNX Runtime's CPU EP), not straight to the slowest path.
		fprintf(stderr, "[open-conversion] splats-mono: DirectML unavailable (%s)\n", why);
		oc_lock(&p->lock);
		c->dml_failed = true;
		bind_cap_locked(p, c, rank_splat_locked(p, c));
		p->generation++;
		oc_unlock(&p->lock);
		notify(0);
		return;
	}
	if (c->ep == OC_EP_OPENVINO && strcmp(c->ov_device, "GPU") == 0) {
		oc_lock(&p->lock);
		oc_copy_str(c->ov_device, sizeof(c->ov_device), "CPU");
		c->ov_provisional = false;
		c->ov_gpu_failed = true;
		bind_cap_locked(p, c, OC_EP_OPENVINO);
		p->generation++;
		oc_unlock(&p->lock);
		fprintf(stderr,
		        "[open-conversion] %s: the OpenVINO GPU device is unavailable (%s); using its CPU device\n",
		        c->task_name, why);
		notify(0);
		return;
	}
	fall_back_to_cpu(p, c, why);
}

//! Is the runtime @p c is bound to loaded?
static bool
runtime_ready(const struct oc_provider *p, const struct oc_cap *c)
{
	return c->ep == OC_EP_OPENVINO ? p->ov != NULL : p->ort != NULL;
}

/*! Load OpenVINO and learn its devices (first warm that needs it). A failed
 *  load moves the capability to ONNX Runtime; no GPU device moves it to
 *  OpenVINO's CPU device. Both re-describe. */
static void
prepare_openvino(struct oc_provider *p, struct oc_cap *c)
{
	if (c->ep != OC_EP_OPENVINO) {
		return;
	}
	if (p->ov == NULL) {
		struct oc_ov *ov = NULL;
		char detail[128];
		const enum oc_ov_status s = oc_ov_load(p->ov_dir, &ov, detail, sizeof(detail));
		oc_lock(&p->lock);
		p->ov = ov;
		p->ov_status = s;
		oc_copy_str(p->ov_detail, sizeof(p->ov_detail), detail);
		oc_unlock(&p->lock);
		fprintf(stderr, "[open-conversion] %s\n", detail);
		if (ov == NULL) {
			p->ov_present = false; // not usable in this process
			fall_back_to_cpu(p, c, detail);
			return;
		}
	}
	if (!p->ov_devices_known) {
		struct oc_ov_gpu gpus[OC_RANK_MAX_OV_GPUS];
		const uint32_t n = oc_ov_list_gpus(p->ov, gpus, OC_RANK_MAX_OV_GPUS);
		oc_lock(&p->lock);
		memcpy(p->ov_gpus, gpus, sizeof(gpus));
		p->ov_gpu_count = n;
		p->ov_devices_known = true;
		oc_unlock(&p->lock);
		for (uint32_t i = 0; i < n; i++) {
			char why[160];
			const bool ok = oc_rank_ov_gpu_ok(&gpus[i], why, sizeof(why));
			fprintf(stderr,
			        "[open-conversion] OpenVINO %s: %s (%s, vendor 0x%04x, IP %u.%u, %u EUs%s)%s%s\n",
			        gpus[i].device, gpus[i].name,
			        gpus[i].type == OC_OV_GPU_TYPE_DISCRETE     ? "dGPU"
			        : gpus[i].type == OC_OV_GPU_TYPE_INTEGRATED ? "iGPU"
			                                                    : "type unknown",
			        gpus[i].vendor_id, gpus[i].arch_major, gpus[i].arch_minor, gpus[i].eu_count,
			        gpus[i].hw_matmul ? ", matrix units" : "",
			        ok ? "; used by EP=auto" : "; not used by EP=auto: ", ok ? "" : why);
		}
		if (n == 0) {
			fprintf(stderr, "[open-conversion] OpenVINO: no GPU device\n");
		}
	}
	if (is_splats(c) && c->ov_provisional) {
		// The GPU device was picked before OpenVINO listed its devices: rank again with
		// the real list (an NVIDIA / AMD / UHD-class device moves to the CPU device, and
		// MODEL=auto then picks the small model).
		oc_lock(&p->lock);
		const char before = c->ov_device[0];
		const struct oc_pack_model *mb = c->model;
		const enum oc_ep ep = rank_splat_locked(p, c);
		bind_cap_locked(p, c, ep);
		const bool changed = c->ov_device[0] != before || c->ep != OC_EP_OPENVINO || c->model != mb;
		if (changed) {
			p->generation++;
		}
		oc_unlock(&p->lock);
		if (changed) {
			notify(0);
		}
	}
	if (c->ep == OC_EP_OPENVINO && strcmp(c->ov_device, "GPU") == 0 && p->ov_gpu_count == 0) {
		fall_back(p, c, "no GPU device");
	}
}

//! OpenVINO's compiled-model cache dir (created on first use); NULL = none.
static const char *
ov_cache_dir(struct oc_provider *p, char *buf, size_t size)
{
	if (!p->ov_cache || p->scratch[0] == '\0' || p->ov == NULL) {
		return NULL;
	}
	// Keyed by the OpenVINO version (a blob from another version is not imported
	// anyway, but this keeps an upgrade from accumulating dead blobs in one dir).
	snprintf(buf, size, "%s%copenvino-%s", p->scratch, OC_PATH_SEP, oc_ov_version(p->ov));
	if (!oc_make_dir(buf)) {
		fprintf(stderr, "[open-conversion] cannot create the OpenVINO cache %s; compiling every warm\n", buf);
		return NULL;
	}
	return buf;
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
		// Temp dirs of processes that died mid-compile.
		oc_sweep_stale(buf, ".tmp-");
	}
	return buf;
}

/*!
 * The Core ML cache as a transaction (the scratch-dir rule: write to a temp
 * name, rename when complete). A model's cache is <base>/<blob file name>,
 * usable only once it holds the ".complete" stamp. A cold warm compiles into
 * <base>/.tmp-<blob>-<pid>; after the warm-up inference succeeds the stamp is
 * written and the directory renamed into place. A process killed mid-compile
 * leaves only a temp dir (swept later), never a half-written cache, and a
 * stamped cache that fails to load is dropped and compiled afresh rather than
 * moving the capability to the CPU.
 */
struct coreml_cache_txn
{
	char final_dir[1400];
	char tmp_dir[1400];
	bool fresh;    //!< the session compiled into tmp_dir
	bool used_hit; //!< the session loaded the stamped final_dir
};

static void
cache_stamp_path(const char *dir, char *buf, size_t size)
{
	snprintf(buf, size, "%s%c.complete", dir, OC_PATH_SEP);
}

/*! Create the cap's session; on Core ML through the cache transaction. */
static enum oc_run_status
create_session(
    struct oc_provider *p, struct oc_cap *c, const char *path, struct coreml_cache_txn *txn, char *err, size_t err_size)
{
	memset(txn, 0, sizeof(*txn));
	if (c->ep == OC_EP_OPENVINO) {
		// OpenVINO's cache (CACHE_DIR) names each blob by a hash of the model and
		// the device configuration, and recompiles when a cached blob does not
		// import: the transaction the Core ML cache needs is OpenVINO's own.
		char dir[1300];
		const char *cache = ov_cache_dir(p, dir, sizeof(dir));
		const bool cpu = strcmp(c->ov_device, "CPU") == 0;
		if (p->diag_profile && is_splats(c)) {
			fprintf(stderr,
			        "[open-conversion] diag: DIAG_PROFILE profiles ONNX Runtime sessions; OpenVINO "
			        "runs are timed only\n");
		}
		return oc_ov_session_create(p->ov, path, cpu ? "CPU" : p->ov_gpu_device, cache, cpu ? p->threads : 0,
		                            &c->ov_session, err, err_size)
		           ? OC_RUN_OK
		           : OC_RUN_FAILED;
	}
	char profile_prefix[1200];
	const bool diag = p->diag_profile && is_splats(c);
	if (diag && p->scratch[0] != '\0' && (oc_make_dir(p->scratch) || oc_file_exists(p->scratch))) {
		snprintf(profile_prefix, sizeof(profile_prefix), "%s%csplats-profile", p->scratch, OC_PATH_SEP);
	} else {
		profile_prefix[0] = '\0';
		if (diag) {
			fprintf(stderr,
			        "[open-conversion] diag: no scratch directory from the host; no profile file\n");
		}
	}
	struct oc_ep_config epc = {
	    .dml_device = is_splats(c) && c->ep == OC_EP_DML ? c->dml_device : p->dml_device,
	    .coreml_units = p->coreml_units,
	    .coreml_cache_dir = NULL,
	    .diag_placement = diag,
	    .profile_prefix = profile_prefix[0] != '\0' ? profile_prefix : NULL,
	};
	char base[1200];
	const char *b = c->ep == OC_EP_COREML ? coreml_cache_dir(p, base, sizeof(base)) : NULL;
	if (b == NULL) {
		return oc_session_create(p->ort, path, c->ep, &epc, p->threads, p->opt, &c->session, err, err_size);
	}
	const char *leaf = strrchr(path, OC_PATH_SEP);
	leaf = leaf != NULL ? leaf + 1 : path; // the store's blob file name (its sha256)
	snprintf(txn->final_dir, sizeof(txn->final_dir), "%s%c%.80s", b, OC_PATH_SEP, leaf);
	char stamp[1500];
	cache_stamp_path(txn->final_dir, stamp, sizeof(stamp));
	if (oc_file_exists(stamp)) {
		epc.coreml_cache_dir = txn->final_dir;
		const enum oc_run_status r =
		    oc_session_create(p->ort, path, c->ep, &epc, p->threads, p->opt, &c->session, err, err_size);
		if (r == OC_RUN_OK) {
			txn->used_hit = true;
			return r;
		}
		fprintf(stderr, "[open-conversion] %s: the Core ML cache did not load (%s); compiling afresh\n",
		        c->task_name, err);
	}
	// No complete cache: drop whatever is there and compile into a temp dir.
	(void)oc_remove_tree(txn->final_dir);
	snprintf(txn->tmp_dir, sizeof(txn->tmp_dir), "%s%c.tmp-%.80s-%llu", b, OC_PATH_SEP, leaf,
	         (unsigned long long)oc_process_id());
	(void)oc_remove_tree(txn->tmp_dir);
	if (!oc_make_dir(txn->tmp_dir)) {
		txn->tmp_dir[0] = '\0';
		return oc_session_create(p->ort, path, c->ep, &epc, p->threads, p->opt, &c->session, err, err_size);
	}
	epc.coreml_cache_dir = txn->tmp_dir;
	const enum oc_run_status r =
	    oc_session_create(p->ort, path, c->ep, &epc, p->threads, p->opt, &c->session, err, err_size);
	if (r == OC_RUN_OK) {
		txn->fresh = true;
	} else {
		(void)oc_remove_tree(txn->tmp_dir);
	}
	return r;
}

/*! After the warm: publish a fresh cache on success, drop a bad one on failure. */
static void
finish_cache(struct coreml_cache_txn *txn, bool ok)
{
	if (ok && txn->fresh) {
		char stamp[1500];
		cache_stamp_path(txn->tmp_dir, stamp, sizeof(stamp));
		// The live session keeps its open files across the rename (same inode).
		if (!oc_touch(stamp) || !oc_rename(txn->tmp_dir, txn->final_dir)) {
			// Another process published first, or the scratch dir refused:
			// the sweep removes this temp dir once this process is gone.
			fprintf(stderr, "[open-conversion] Core ML cache not published (%s)\n", txn->final_dir);
		}
	} else if (!ok && txn->fresh) {
		(void)oc_remove_tree(txn->tmp_dir);
	} else if (!ok && txn->used_hit) {
		(void)oc_remove_tree(txn->final_dir); // the next warm compiles afresh
	}
	txn->fresh = txn->used_hit = false;
}

/*! One line: this process's memory on the splat capability's DirectML adapter. */
static bool
splat_gpu_mem(const struct oc_cap *c, struct oc_gpu_mem *m, char *buf, size_t size)
{
	if (c->ep != OC_EP_DML || !oc_gpu_memory(c->dml_luid, m)) {
		buf[0] = '\0';
		return false;
	}
	snprintf(buf, size, "%llu MB dedicated (budget %llu MB), %llu MB shared", (unsigned long long)m->local_usage_mb,
	         (unsigned long long)m->local_budget_mb, (unsigned long long)m->nonlocal_usage_mb);
	return true;
}

/*! After a splat session loaded: its GPU memory and, with DIAG_PROFILE, where ONNX Runtime
 *  placed the nodes. */
static void
splat_log_load(struct oc_provider *p, struct oc_cap *c, uint64_t load_ms)
{
	struct oc_gpu_mem mem;
	char line[160];
	if (splat_gpu_mem(c, &mem, line, sizeof(line))) {
		fprintf(stderr, "[open-conversion] %s: loaded on DirectML in %llu ms; GPU memory: %s\n", c->model->name,
		        (unsigned long long)load_ms, line);
	}
	const struct oc_place_report *pl = oc_session_placement(c->session);
	if (!p->diag_profile || pl == NULL) {
		return;
	}
	char place[1024];
	oc_place_format(pl, 8, place, sizeof(place));
	const uint32_t total = oc_place_total(pl, "");
	const uint32_t cpu = oc_place_total(pl, "CPUExecutionProvider");
	fprintf(stderr,
	        "[open-conversion] diag: %s node placement (%s, after ONNX Runtime's graph optimisation): "
	        "%u of %u nodes on the CPU EP; %s\n",
	        c->model->name, c->backend, cpu, total, place);
}

/*! DIAG_PROFILE, after a splat job's second run: stop profiling, summarise the file. */
static void
splat_log_profile(struct oc_provider *p, struct oc_cap *c)
{
	char path[1300];
	if (!oc_session_end_profiling(c->session, path, sizeof(path))) {
		return; // profiled the first job only (or never: no scratch directory)
	}
	FILE *f = fopen(path, "rb");
	char *text = NULL;
	size_t len = 0;
	if (f != NULL) {
		if (fseek(f, 0, SEEK_END) == 0) {
			const long n = ftell(f);
			if (n > 0 && n < (64l << 20) && fseek(f, 0, SEEK_SET) == 0) {
				text = (char *)malloc((size_t)n);
				len = text != NULL ? fread(text, 1, (size_t)n, f) : 0;
			}
		}
		fclose(f);
	}
	struct oc_profile_summary *sum = (struct oc_profile_summary *)calloc(1, sizeof(*sum));
	char line[1400];
	if (sum != NULL && oc_profile_summarize(text, len, sum)) {
		oc_profile_format(sum, 10, line, sizeof(line));
	} else {
		snprintf(line, sizeof(line), "could not read it");
	}
	fprintf(stderr, "[open-conversion] diag: %s profile %s: %s\n", c->model->name, path, line);
	if (c->ep == OC_EP_DML) {
		fprintf(stderr,
		        "[open-conversion] diag: on DirectML a kernel's time is its CPU-side recording unless the "
		        "queue is full; the GPU's own time lands in the node that waits for it (MemcpyToHost)\n");
	}
	free(sum);
	free(text);
	(void)p;
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
	if (is_splats(c)) {
		// The store may have changed since the pick (a local build finished, a blob
		// was pruned): pick again, and re-describe if the model changes. A loaded
		// session of the old model is dropped and the new one warms below.
		refresh_splat_presence(p);
		oc_lock(&p->lock);
		const struct oc_pack_model *before = c->model;
		bind_cap_locked(p, c, c->ep);
		const bool changed = c->model != before;
		if (changed) {
			p->generation++;
		}
		oc_unlock(&p->lock);
		if (changed) {
			release_session(p, c);
			set_state(p, c, XRT_CONV_STATE_INSTALLED, XRT_CONV_FAIL_NONE, 0, -1.0f,
			          "splat model re-picked");
			notify(0);
		}
	}
	if (state_of(p, c) == XRT_CONV_STATE_READY && has_session(c)) {
		return XRT_SUCCESS;
	}

	// Dependencies first: a runtime installed since probe is picked up here.
	if (p->ort == NULL) {
		load_runtime(p);
		if (p->ort != NULL) {
			oc_lock(&p->lock);
			p->default_ep = choose_ep(p);
			if (is_splats(c)) {
				bind_cap_locked(p, c, rank_splat_locked(p, c));
			} else if (c->ep != OC_EP_OPENVINO) {
				bind_cap_locked(p, c, p->default_ep);
			}
			p->generation++;
			oc_unlock(&p->lock);
		}
	}
	prepare_openvino(p, c);
	char why[XRT_CONV_DETAIL_MAX];
	if (!runtime_ready(p, c) || p->pack_count == 0 || !model_usable(c, why, sizeof(why))) {
		initial_state(p, c);
		return XRT_ERROR_FEATURE_NOT_SUPPORTED; // terminal for now; the runtime falls through
	}
	release_session(p, c); // a FAILED(DEVICE_LOST) session is unusable

	const struct oc_pack_model *m = c->model;
	struct xrt_conv_model_location loc;
	bool requested = false;
	uint64_t absent_since = 0;
	uint64_t verify_t0 = 0, moved_at = 0; // the store's verify: start, last state / progress change
	uint32_t seen_state = UINT32_MAX;
	float seen_progress = -2.0f;
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
			if (verify_t0 != 0) {
				fprintf(stderr, "[open-conversion] %s: the model store verified it in %llu ms\n",
				        m->name, (unsigned long long)(oc_now_ms() - verify_t0));
			}
			break;
		}
		if (known && loc.state == XRT_CONV_STATE_WARMING) {
			// Every file is on disk and the store is hashing it: wait while it
			// moves (state or progress), fail only a store that has stalled.
			const uint64_t now = oc_now_ms();
			const uint64_t bytes = variant_bytes(c->variant);
			const uint64_t floor_ms = bytes * 1000ull / OC_STORE_VERIFY_FLOOR_BPS;
			const uint64_t stall_ms = floor_ms > OC_STORE_STALL_MIN_MS ? floor_ms : OC_STORE_STALL_MIN_MS;
			if (verify_t0 == 0) {
				verify_t0 = now;
				fprintf(stderr, "[open-conversion] %s: the model store is verifying it (%.2f GB)\n",
				        m->name, (double)bytes / 1e9);
			}
			if (loc.state != seen_state || loc.progress > seen_progress) {
				seen_state = loc.state;
				seen_progress = loc.progress;
				moved_at = now;
			} else if (now - moved_at > stall_ms) {
				snprintf(why, sizeof(why),
				         "%.40s: the model store stopped verifying it (no progress in %llu s)", m->name,
				         (unsigned long long)(stall_ms / 1000u));
				set_state(p, c, XRT_CONV_STATE_ABSENT, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
				return XRT_ERROR_FEATURE_NOT_SUPPORTED;
			}
			// Verify is the first stretch of the warm (loading starts at 0.1); the
			// host's progress when it gives one, else none.
			snprintf(why, sizeof(why), "%.40s: the model store is verifying it (%llu s)", m->name,
			         (unsigned long long)((now - verify_t0) / 1000u));
			set_state(p, c, XRT_CONV_STATE_WARMING, XRT_CONV_FAIL_NONE, 0,
			          loc.progress >= 0.0f ? 0.1f * (loc.progress > 1.0f ? 1.0f : loc.progress) : -1.0f,
			          why);
			absent_since = 0;
			oc_sleep_ms(OC_POLL_MS);
			continue;
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
				// ABSENT(MODEL) after a refused request: a locally built model that
				// nothing can fetch (spec §5.3, build-locally); else a policy.
				struct xrt_conv_model_location after;
				const bool build = store_resolve(c->ref, &after) &&
				                   after.state == XRT_CONV_STATE_ABSENT &&
				                   after.failure_reason == XRT_CONV_FAIL_MODEL;
				if (c->pick_forced) {
					// MODEL names this model: say so, never switch to the other one.
					snprintf(
					    why, sizeof(why),
					    "%.40s: MODEL=%s, but it is not in the model store (%s); not switching",
					    m->name, p->splat_model_setting,
					    build ? "build it locally" : "build or fetch it");
				} else if (build) {
					snprintf(why, sizeof(why), "%.40s: not in the model store; build it locally",
					         m->name);
				} else {
					snprintf(why, sizeof(why),
					         "%s: not in the model store and the download is not allowed", m->name);
				}
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
	struct coreml_cache_txn txn;
	const uint64_t load_t0 = oc_now_ms();
	r = create_session(p, c, loc.path, &txn, err, sizeof(err));
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
		finish_cache(&txn, false);
		snprintf(why, sizeof(why), "%s: warm cancelled", m->name);
		set_state(p, c, XRT_CONV_STATE_INSTALLED, XRT_CONV_FAIL_NONE, 0, -1.0f, why);
		return XRT_TIMEOUT;
	}
	if (r == OC_RUN_OK && !is_splats(c)) {
		// The splat capability is job-only and runs seconds per job: its first
		// job pays the first-inference cost instead of every warm.
		snprintf(why, sizeof(why), "%s: first inference (%s)", m->name, c->backend);
		set_state(p, c, XRT_CONV_STATE_WARMING, XRT_CONV_FAIL_NONE, 0, 0.6f, why);
		r = warm_up_inference(p, c, err, sizeof(err));
	}
	if (r == OC_RUN_OK && c->ep == OC_EP_OPENVINO) {
		fprintf(stderr, "[open-conversion] %s: OpenVINO %s compiled in %llu ms\n", m->name, c->ov_device,
		        (unsigned long long)load_ms);
	}
	if (r == OC_RUN_OK && is_splats(c) && c->session != NULL) {
		splat_log_load(p, c, load_ms);
	}
	if (r != OC_RUN_OK) {
		release_session(p, c); // before the cache dir it may have mapped is removed
	}
	finish_cache(&txn, r == OC_RUN_OK);
	// A splat model that does not fit the accelerator's memory falls back too (its
	// next backend is the CPU's RAM); a depth model out of memory is a resource failure.
	if (r != OC_RUN_OK && (r != OC_RUN_OUT_OF_MEMORY || is_splats(c)) && c->ep != OC_EP_CPU) {
		// DirectML / Core ML could not start (driver, feature level, an operator,
		// a Core ML compile failure): the CPU execution provider serves the same model.
		release_session(p, c);
		fall_back(p, c, err);
		prepare_openvino(p, c); // DirectML -> OpenVINO: load it and check its devices first
		m = c->model;
		if (!runtime_ready(p, c)) {
			initial_state(p, c); // e.g. OpenVINO gone and no ONNX Runtime either
			return XRT_ERROR_FEATURE_NOT_SUPPORTED;
		}
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
	// 24 + 1 + 8 + 4 + 9 + NUL fits version[48].
	if (c->ep == OC_EP_OPENVINO) {
		snprintf(c->version, sizeof(c->version), "%.24s@%.8s ov %.9s", m->name, m->version,
		         oc_ov_version(p->ov));
	} else {
		snprintf(c->version, sizeof(c->version), "%.24s@%.8s ort %.9s", m->name, m->version,
		         oc_ort_version(p->ort));
	}
	oc_unlock(&p->lock);
	if (c->ep == OC_EP_COREML && reported) {
		snprintf(why, sizeof(why), "%s: ready (%s, %u/%u nodes)", m->name, c->backend, taken, total);
	} else if (is_splats(c) && c->pick_note[0] != '\0') {
		snprintf(why, sizeof(why), "%.20s: ready (%.18s; %.76s)", m->name, c->backend, c->pick_note);
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
	if (state_of(p, c) != XRT_CONV_STATE_READY || c->session == NULL || is_splats(c)) {
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

/*
 *
 * Job lane (SPLATS_MONO): one photo in, one 3DGS PLY blob out, plus the facts.
 *
 */

struct splat_sink
{
	struct xrt_conv_blob_sink *sink;
	uint64_t written;
};

static bool
splat_sink_write(void *ctx, const void *bytes, uint64_t count)
{
	struct splat_sink *s = (struct splat_sink *)ctx;
	if (!s->sink->write(s->sink->ctx, bytes, count)) {
		return false;
	}
	s->written += count;
	return true;
}

//! A refused input: the job fails with no blob, the capability stays READY (ADR-057 §4).
static xrt_result_t
splat_refuse(struct xrt_conv_job_out *out, uint32_t code, const char *reason, xrt_result_t r)
{
	struct xrt_conv_splat_result *sr =
	    out != NULL && COVERS(out, struct xrt_conv_job_out, splat_result) ? out->splat_result : NULL;
	if (sr != NULL) {
		SET(sr, struct xrt_conv_splat_result, refusal_code, code);
		if (COVERS(sr, struct xrt_conv_splat_result, refusal_reason)) {
			oc_copy_str(sr->refusal_reason, sizeof(sr->refusal_reason), reason);
		}
	}
	fprintf(stderr, "[open-conversion] splats-mono: refused: %s\n", reason);
	return r;
}

//! The build stamp: which build answered (returned with every result, failed or not).
static void
splat_stamp(struct oc_provider *p, const struct oc_cap *c, struct xrt_conv_splat_result *sr)
{
	if (sr == NULL) {
		return;
	}
#define STAMP(FIELD, ...)                                                                                              \
	do {                                                                                                           \
		if (COVERS(sr, struct xrt_conv_splat_result, FIELD)) {                                                 \
			snprintf(sr->FIELD, sizeof(sr->FIELD), __VA_ARGS__);                                           \
		}                                                                                                      \
	} while (0)
	STAMP(build_provider, "%s %s %.36s", OC_ID, OC_VERSION, OC_BUILD_COMMIT);
	if (c->model != NULL && c->pack != NULL) {
		STAMP(build_model_id, "%.24s/%.24s@%.12s", c->pack->pack, c->model->name, c->model->version);
	}
	if (c->variant != NULL) {
		STAMP(build_model_sha256, "%.64s", c->variant->file.sha256);
	}
	STAMP(build_runtime, "%s", c->backend);
	STAMP(build_runtime_version, "%.30s", c->ep == OC_EP_OPENVINO ? oc_ov_version(p->ov) : oc_ort_version(p->ort));
	STAMP(build_host_id, "%s", "");
#undef STAMP
	SET(sr, struct xrt_conv_splat_result, host_kind, XRT_CONV_HOST_KIND_LOCAL);
	SET(sr, struct xrt_conv_splat_result, valid, XRT_CONV_SPLAT_VALID_BUILD);
}

static xrt_result_t
splat_job(struct oc_provider *p,
          struct oc_cap *c,
          const struct xrt_conv_job_in *in,
          const struct xrt_conv_params *params,
          const struct xrt_conv_job_control *ctl,
          struct xrt_conv_job_out *out)
{
	if (in == NULL || ctl == NULL || !COVERS(in, struct xrt_conv_job_in, images) ||
	    !COVERS(ctl, struct xrt_conv_job_control, is_cancelled)) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}
	struct xrt_conv_splat_result *sr =
	    out != NULL && COVERS(out, struct xrt_conv_job_out, splat_result) ? out->splat_result : NULL;
	splat_stamp(p, c, sr);
	if (state_of(p, c) != XRT_CONV_STATE_READY || !has_session(c) || c->model == NULL) {
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	const struct oc_pack_model *m = c->model;
	if (COVERS(in, struct xrt_conv_job_in, input_kind) && (in->input_kind & XRT_CONV_INPUT_MONO) == 0) {
		return splat_refuse(out, XRT_CONV_REFUSAL_UNSUPPORTED_INPUT, "one photo only (no stereo pair)",
		                    XRT_ERROR_INPUT_UNSUPPORTED);
	}
	const uint32_t requested =
	    COVERS(in, struct xrt_conv_job_in, requested_blob_format) ? in->requested_blob_format : 0u;
	if (requested != 0 && requested != XRT_CONV_BLOB_PLY_3DGS) {
		// The requested OUTPUT format, not the photo, is the problem: say what to ask for.
		// (No refusal code names an output format; UNSUPPORTED_INPUT is the job request's.)
		return splat_refuse(out, XRT_CONV_REFUSAL_UNSUPPORTED_INPUT,
		                    "output format not supported: this provider writes 3DGS PLY only; pass an output "
		                    "path ending in .ply",
		                    XRT_ERROR_OUTPUT_UNSUPPORTED);
	}
	const struct xrt_conv_image *color = find_color(
	    in->images, COVERS(in, struct xrt_conv_job_in, image_role) ? in->image_role : NULL, in->image_count);
	if (color == NULL) {
		return splat_refuse(out, XRT_CONV_REFUSAL_UNSUPPORTED_INPUT, "no CPU RGBA8 / BGRA8 colour image",
		                    XRT_ERROR_INPUT_UNSUPPORTED);
	}
	struct xrt_conv_blob_sink *sink = NULL;
	for (uint32_t i = 0; i < ctl->sink_count && ctl->sinks != NULL; i++) {
		if (ctl->sinks[i].role == XRT_CONV_OUT_SPLAT_BLOB) {
			sink = &ctl->sinks[i];
			break;
		}
	}
	if (sink == NULL || sink->begin == NULL || sink->write == NULL) {
		return XRT_ERROR_OUTPUT_UNSUPPORTED;
	}
	bool (*cancelled)(void *) = ctl->is_cancelled != NULL ? ctl->is_cancelled : never_cancelled;

	// Focal length, in the input's pixels: GIVEN wins; otherwise the model's own
	// convention (a 30 mm full-frame equivalent), labelled ASSUMED. Never EXIF
	// here: the provider receives decoded pixels.
	const uint32_t w = color->width, h = color->height;
	const uint32_t flags = COVERS(in, struct xrt_conv_job_in, job_flags) ? in->job_flags : 0u;
	const float given =
	    params != NULL && COVERS(params, struct xrt_conv_params, focal_px) ? params->focal_px : 0.0f;
	const bool focal_given =
	    given > 0.0f && isfinite(given) &&
	    ((flags & XRT_CONV_JOB_FOCAL_GIVEN) != 0 || !COVERS(in, struct xrt_conv_job_in, job_flags));
	const float focal =
	    focal_given
	        ? given
	        : (float)(OC_ASSUMED_FOCAL_MM * sqrt((double)w * w + (double)h * h) / sqrt(36.0 * 36.0 + 24.0 * 24.0));

	// 1. Preprocess: the whole picture into the model's grid.
	if (ctl->progress != NULL) {
		ctl->progress(ctl->ctx, 0.0f, "preprocess");
	}
	const uint32_t gw = m->width, gh = m->height;
	if (!grow(&p->chw, &p->chw_cap, (size_t)gw * gh * 3u)) {
		return XRT_ERROR_ALLOCATION;
	}
	const uint32_t stride = color->row_stride != 0 ? color->row_stride : w * 4u;
	if (!oc_resample_rgb_aa((const uint8_t *)color->cpu_ptr, stride, w, h, is_bgr(color->format), gw, gh, p->chw)) {
		return XRT_ERROR_ALLOCATION;
	}
	if (cancelled(ctl->ctx)) {
		return XRT_SUCCESS; // cancelled: nothing written, the runtime reports it
	}

	// 2. Inference.
	const size_t n_max = m->splat_count != 0 ? m->splat_count : (size_t)gw * gh;
	if (!grow(&p->gauss, &p->gauss_cap, n_max * 14u)) {
		return XRT_ERROR_ALLOCATION;
	}
	const float disparity_factor = focal / (float)w;
	const int64_t image_shape[4] = {1, 3, (int64_t)gh, (int64_t)gw};
	const int64_t factor_shape[1] = {1};
	const struct oc_tensor_in ins[2] = {
	    {m->t_image, p->chw, image_shape, 4},
	    {m->t_disparity_factor, &disparity_factor, factor_shape, 1},
	};
	float *g = p->gauss;
	struct oc_tensor_out outs[5] = {
	    {m->t_means, g, n_max * 3u, 0},
	    {m->t_scales, g + n_max * 3u, n_max * 3u, 0},
	    {m->t_quats, g + n_max * 6u, n_max * 4u, 0},
	    {m->t_colors, g + n_max * 10u, n_max * 3u, 0},
	    {m->t_opacities, g + n_max * 13u, n_max, 0},
	};
	if (ctl->progress != NULL) {
		ctl->progress(ctl->ctx, 0.05f, "infer");
	}
	char err[XRT_CONV_DETAIL_MAX] = "";
	uint64_t t0 = oc_now_ms();
	bool ran = false;
	if (c->ov_session != NULL) {
		const enum oc_ov_run r =
		    oc_ov_run(c->ov_session, ins, 2, outs, 5, cancelled, ctl->ctx, err, sizeof(err));
		if (r == OC_OV_RUN_CANCELLED) {
			return XRT_SUCCESS;
		}
		ran = r == OC_OV_RUN_OK;
	} else {
		struct oc_gpu_mem mem0, mem1;
		char before[160], after[160];
		const bool have_mem = splat_gpu_mem(c, &mem0, before, sizeof(before));
		enum oc_run_status r = oc_session_run_named(c->session, ins, 2, outs, 5, err, sizeof(err));
		if (r == OC_RUN_OK && p->diag_profile) {
			// A second run: the first pays DirectML's first-run allocation and compilation.
			const uint64_t first_ms = oc_now_ms() - t0;
			fprintf(stderr, "[open-conversion] diag: %s first run %llu ms (%s); running it again\n",
			        m->name, (unsigned long long)first_ms, c->backend);
			t0 = oc_now_ms();
			r = oc_session_run_named(c->session, ins, 2, outs, 5, err, sizeof(err));
			if (r == OC_RUN_OK) {
				splat_log_profile(p, c);
			}
		}
		if (have_mem && r == OC_RUN_OK && splat_gpu_mem(c, &mem1, after, sizeof(after))) {
			fprintf(stderr, "[open-conversion] %s: GPU memory before the run: %s; after: %s\n", m->name,
			        before, after);
			if (mem1.local_usage_mb > mem1.local_budget_mb ||
			    mem1.nonlocal_usage_mb > mem0.nonlocal_usage_mb + 256u) {
				fprintf(stderr,
				        "[open-conversion] %s: over the GPU memory budget: Windows keeps part of the "
				        "model in "
				        "system memory, which can make a run many times slower%s\n",
				        m->name,
				        c->model == p->splat_cand[OC_SPLAT_FP16] ? "" : " (the fp16 model needs half)");
			}
		}
		if (r == OC_RUN_OUT_OF_MEMORY && c->ep == OC_EP_CPU) {
			fprintf(stderr, "[open-conversion] %s: inference failed: %s\n", m->name, err);
			return XRT_ERROR_ALLOCATION;
		}
		ran = r == OC_RUN_OK;
	}
	const uint64_t infer_ms = oc_now_ms() - t0;
	if (!ran) {
		fprintf(stderr, "[open-conversion] %s: inference failed (%s): %s\n", m->name, c->backend, err);
		if (c->ep != OC_EP_CPU) {
			// An accelerator run failed after a good warm: drop the session, move the
			// cap to the next backend (re-describe) and report the accelerator gone;
			// the runtime re-warms there. No silent retry inside a job (§2.4).
			char why[XRT_CONV_DETAIL_MAX];
			release_session(p, c);
			fall_back(p, c, err);
			snprintf(why, sizeof(why), "%.40s: run failed; re-warms on %.24s", m->name, c->backend);
			set_state(p, c, XRT_CONV_STATE_FAILED, XRT_CONV_FAIL_DEVICE_LOST, 1, -1.0f, why);
		}
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	const size_t n = outs[4].count;
	if (n == 0 || outs[0].count != n * 3u || outs[1].count != n * 3u || outs[2].count != n * 4u ||
	    outs[3].count != n * 3u) {
		fprintf(stderr, "[open-conversion] %s: inconsistent Gaussian outputs (%zu / %zu / %zu / %zu / %zu)\n",
		        m->name, outs[0].count, outs[1].count, outs[2].count, outs[3].count, outs[4].count);
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	if (cancelled(ctl->ctx)) {
		return XRT_SUCCESS;
	}

	// 3. Postprocess straight into the sink.
	if (ctl->progress != NULL) {
		ctl->progress(ctl->ctx, 0.8f, "write");
	}
	const struct oc_splat_raw raw = {
	    .means = g,
	    .scales = g + n_max * 3u,
	    .quats = g + n_max * 6u,
	    .colors = g + n_max * 10u,
	    .opacities = g + n_max * 13u,
	    .count = n,
	    .colors_linear = strcmp(m->color_space, "srgb") != 0,
	};
	struct splat_sink ss = {sink, 0};
	struct oc_splat_stats st;
	const uint64_t t1 = oc_now_ms();
	if (!sink->begin(sink->ctx, XRT_CONV_BLOB_PLY_3DGS, oc_splat_ply_size(n))) {
		return XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	if (!oc_splat_write_ply(&raw, w, h, focal, splat_sink_write, &ss, cancelled, ctl->ctx, &st)) {
		return cancelled(ctl->ctx) ? XRT_SUCCESS : XRT_ERROR_OUTPUT_REQUEST_FAILURE;
	}
	fprintf(stderr,
	        "[open-conversion] %s: %zu Gaussians from %ux%u (focal %.1f px, %s) on %s: infer %llu ms, write %llu "
	        "ms%s\n",
	        m->name, n, w, h, (double)focal, focal_given ? "given" : "assumed", c->backend,
	        (unsigned long long)infer_ms, (unsigned long long)(oc_now_ms() - t1),
	        st.non_finite != 0 ? " (non-finite Gaussians made transparent)" : "");

	// 4. Facts (frozen splat_info: every field set) and the focus suggestion.
	if (out != NULL) {
		SET(out, struct xrt_conv_job_out, written, XRT_CONV_OUT_SPLAT_BLOB);
		if (COVERS(out, struct xrt_conv_job_out, splat_info)) {
			struct xrt_conv_splat_info *si = &out->splat_info;
			memset(si, 0, sizeof(*si));
			si->blob_format = XRT_CONV_BLOB_PLY_3DGS;
			si->coords = XRT_CONV_COORDS_OPENCV;
			si->color_space = XRT_CONV_CS_SRGB; // f_dc carries sRGB values, as 3DGS viewers expect
			si->metric = 0; // a monocular model's scale, through a focal that may be assumed: not metres
			si->intrinsics_source = focal_given ? XRT_CONV_INTRINSICS_GIVEN : XRT_CONV_INTRINSICS_ASSUMED;
			si->fx = focal;
			si->fy = focal;
			si->cx = (float)w * 0.5f;
			si->cy = (float)h * 0.5f;
			si->grid_width = w; // the cloud unprojects exactly through these intrinsics of the input
			si->grid_height = h;
			si->baseline_m = 0.0f;                         // unknown, never defaulted
			si->focus_z_m = si->near_m = si->far_m = 0.0f; // not metres: unknown
			si->splat_count = (uint64_t)n;
		}
	}
	if (sr != NULL && st.focus_valid && COVERS(sr, struct xrt_conv_splat_result, focus_source)) {
		sr->focus_point[0] = st.focus_point[0];
		sr->focus_point[1] = st.focus_point[1];
		sr->focus_point[2] = st.focus_point[2];
		sr->focus_source = XRT_CONV_FOCUS_SOURCE_NEAREST_CLUMP; // a suggestion, in the splat's own units
		sr->valid |= XRT_CONV_SPLAT_VALID_FOCUS;
	}
	if (ctl->progress != NULL) {
		ctl->progress(ctl->ctx, 1.0f, "done");
	}
	return XRT_SUCCESS;
}


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
	if (c != NULL && is_splats(c)) {
		return splat_job(p, c, in, params, ctl, out);
	}
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
		oc_ov_session_destroy(p->caps[i].ov_session);
	}
	// ONNX Runtime stays mapped and its environment alive: it does not support
	// being unloaded and re-initialised within one process.
	free(p->chw);
	free(p->map);
	free(p->gauss);
	for (uint32_t i = 0; i < p->pack_count; i++) {
		free(p->packs[i]);
	}
	oc_mutex_destroy(&p->lock);
	free(p);
}


/*
 *
 * Negotiate: the only export.
 *
 */

/*! The settings this provider reads (`displayxr-cli conv options open-conversion`). */
static const struct xrt_component_setting_desc g_settings[] = {
    {"ENABLE", XRT_COMPONENT_SETTING_TYPE_BOOL, "1", "0 = probe declines"},
    {"QUALITY", XRT_COMPONENT_SETTING_TYPE_ENUM, "medium", "low | medium | high: which model defaults[task] names"},
    {"EP", XRT_COMPONENT_SETTING_TYPE_ENUM, "auto",
     "auto | cpu | dml | coreml | openvino: the execution provider (openvino: the splat capability only; "
     "splats auto = DirectML on a discrete non-Intel GPU, else OpenVINO's Intel Arc / Xe GPU, else a CPU)"},
    {"OV_DEVICE", XRT_COMPONENT_SETTING_TYPE_ENUM, "auto",
     "auto | gpu | cpu: OpenVINO's device when OpenVINO serves splats (auto = an Intel Arc / Xe GPU, never an "
     "NVIDIA / AMD / UHD one, else CPU; gpu = any GPU device); does not select OpenVINO over DirectML"},
    {"OV_CACHE", XRT_COMPONENT_SETTING_TYPE_BOOL, "1", "0 = no OpenVINO compiled-model cache in the scratch dir"},
    {"MODEL", XRT_COMPONENT_SETTING_TYPE_ENUM, "auto",
     "auto | full | fp16 | int4: the splat model (full = defaults[splats-mono].high, int4 = .low, fp16 = .medium "
     "on DirectML only; auto = full on a GPU backend with >= 16 GB RAM, on DirectML fp16 when built (>= 6 GB "
     "VRAM) else full with >= 10 GB VRAM, else int4, among the models in the store; DirectML never runs int4)"},
    {"OPENVINO", XRT_COMPONENT_SETTING_TYPE_STRING, "", "absolute path of another OpenVINO runtime folder"},
    {"COREML_UNITS", XRT_COMPONENT_SETTING_TYPE_ENUM, "gpu", "gpu | all | ane | cpu: macOS Core ML compute units"},
    {"COREML_CACHE", XRT_COMPONENT_SETTING_TYPE_BOOL, "1", "0 = no Core ML compiled-model cache"},
    {"ORT_LOG", XRT_COMPONENT_SETTING_TYPE_ENUM, "warning", "warning | info | verbose"},
    {"THREADS", XRT_COMPONENT_SETTING_TYPE_INT, "4", "CPU inference threads; 0 = every core"},
    {"GRAPH_OPT", XRT_COMPONENT_SETTING_TYPE_ENUM, "extended", "basic | extended | all (ONNX Runtime)"},
    {"DIAG_PROFILE", XRT_COMPONENT_SETTING_TYPE_BOOL, "0",
     "1 = splat diagnostics: log ONNX Runtime's node placement, run each job's model twice and log both "
     "times, GPU memory and the second run's top-10 operators (profile JSON in the scratch dir)"},
    {"PACK", XRT_COMPONENT_SETTING_TYPE_STRING, "",
     "absolute pack manifest paths, ';'-separated (default: every *.models.json next to the binary)"},
    {"ONNXRUNTIME", XRT_COMPONENT_SETTING_TYPE_STRING, "", "absolute path of another ONNX Runtime library"},
};

static uint32_t
oc_describe_settings(const struct xrt_component_setting_desc **out_descs)
{
	if (out_descs != NULL) {
		*out_descs = g_settings;
	}
	return (uint32_t)(sizeof(g_settings) / sizeof(g_settings[0]));
}

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
    .describe_settings = oc_describe_settings,
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
