// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ONNX Runtime glue, see oc_ort.h.
 */

#include "oc_ort.h"
#include "oc_platform.h"

#include <onnxruntime_c_api.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define COBJMACROS
#include <dxgi.h>
#include <dml_provider_factory.h>
#endif

struct oc_ort
{
	void *lib;
	const OrtApi *api;
	OrtEnv *env;
	char version[32];
	bool has_dml;
};

struct oc_session
{
	struct oc_ort *ort;
	OrtSession *session;
	OrtMemoryInfo *cpu_mem;
	OrtMemoryInfo *device_mem; //!< where a temporal cache lives between frames
	enum oc_ep ep;
};

struct oc_temporal
{
	const OrtApi *api;
	OrtValue *cache;    //!< the previous frame's cache_out (on the session's device)
	OrtIoBinding *bind; //!< reused per frame
	OrtSession *session;
};

const char *
oc_ort_library_name(void)
{
#if defined(_WIN32)
	return "onnxruntime.dll";
#elif defined(__APPLE__)
	return "libonnxruntime.dylib";
#else
	return "libonnxruntime.so.1";
#endif
}

uint32_t
oc_ort_api_floor(void)
{
	return ORT_API_VERSION;
}

static void ORT_API_CALL
ort_log(void *param,
        OrtLoggingLevel severity,
        const char *category,
        const char *logid,
        const char *code_location,
        const char *message)
{
	(void)param;
	(void)category;
	(void)logid;
	(void)code_location;
	if (severity >= ORT_LOGGING_LEVEL_WARNING) {
		fprintf(stderr, "[open-conversion] onnxruntime: %s\n", message != NULL ? message : "");
	}
}

//! Consumes @p st. Returns true when there was no error.
static bool
status_ok(const OrtApi *api, OrtStatus *st, char *err, size_t err_size)
{
	if (st == NULL) {
		return true;
	}
	if (err != NULL && err_size > 0) {
		snprintf(err, err_size, "%s", api->GetErrorMessage(st));
	}
	api->ReleaseStatus(st);
	return false;
}

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

//! Classify an ONNX Runtime error message (it carries the HRESULT for DirectML).
static enum oc_run_status
classify(const char *msg)
{
	if (msg == NULL) {
		return OC_RUN_FAILED;
	}
	// DXGI_ERROR_DEVICE_REMOVED / _HUNG / _RESET.
	if (contains_ci(msg, "887a0005") || contains_ci(msg, "887a0006") || contains_ci(msg, "887a0007") ||
	    contains_ci(msg, "device removed") || contains_ci(msg, "device_removed")) {
		return OC_RUN_DEVICE_LOST;
	}
	// E_OUTOFMEMORY, allocator failures.
	if (contains_ci(msg, "8007000e") || contains_ci(msg, "out of memory") ||
	    contains_ci(msg, "failed to allocate") || contains_ci(msg, "bad allocation")) {
		return OC_RUN_OUT_OF_MEMORY;
	}
	return OC_RUN_FAILED;
}

#define TRY(EXPR)                                                                                                      \
	do {                                                                                                           \
		if (!status_ok(api, (EXPR), err, err_size)) {                                                          \
			goto fail;                                                                                     \
		}                                                                                                      \
	} while (0)

enum oc_ort_status
oc_ort_load(const char *lib_path, struct oc_ort **out, char *detail, size_t detail_size)
{
	*out = NULL;
	if (!oc_file_exists(lib_path)) {
		snprintf(detail, detail_size, "ONNX Runtime not found (%s)", lib_path != NULL ? lib_path : "-");
		return OC_ORT_ABSENT;
	}
	char err[160];
	void *lib = oc_lib_open(lib_path, err, sizeof(err));
	if (lib == NULL) {
		snprintf(detail, detail_size, "ONNX Runtime did not load: %s", err);
		return OC_ORT_BROKEN;
	}
	typedef const OrtApiBase *(ORT_API_CALL * get_base_fn)(void);
	get_base_fn get_base = (get_base_fn)oc_lib_sym(lib, "OrtGetApiBase");
	const OrtApiBase *base = get_base != NULL ? get_base() : NULL;
	if (base == NULL) {
		snprintf(detail, detail_size, "ONNX Runtime has no OrtGetApiBase");
		return OC_ORT_BROKEN; // the library stays mapped: unloading ONNX Runtime is not supported
	}
	const char *ver = base->GetVersionString != NULL ? base->GetVersionString() : "?";
	const OrtApi *api = base->GetApi(ORT_API_VERSION);
	if (api == NULL) {
		snprintf(detail, detail_size, "ONNX Runtime %s is older than the supported API %u", ver,
		         (unsigned)ORT_API_VERSION);
		return OC_ORT_TOO_OLD;
	}
	struct oc_ort *o = (struct oc_ort *)calloc(1, sizeof(*o));
	if (o == NULL) {
		snprintf(detail, detail_size, "out of memory");
		return OC_ORT_BROKEN;
	}
	o->lib = lib;
	o->api = api;
	oc_copy_str(o->version, sizeof(o->version), ver);
	OrtStatus *st =
	    api->CreateEnvWithCustomLogger(ort_log, NULL, ORT_LOGGING_LEVEL_WARNING, "open-conversion", &o->env);
	if (!status_ok(api, st, err, sizeof(err))) {
		snprintf(detail, detail_size, "ONNX Runtime %s: environment failed: %s", ver, err);
		free(o);
		return OC_ORT_BROKEN;
	}
	char **providers = NULL;
	int count = 0;
	if (api->GetAvailableProviders(&providers, &count) == NULL) {
		for (int i = 0; i < count; i++) {
			if (strcmp(providers[i], "DmlExecutionProvider") == 0) {
				o->has_dml = true;
			}
		}
		(void)status_ok(api, api->ReleaseAvailableProviders(providers, count), NULL, 0);
	}
	snprintf(detail, detail_size, "ONNX Runtime %s%s", ver, o->has_dml ? " (DirectML)" : "");
	*out = o;
	return OC_ORT_OK;
}

const char *
oc_ort_version(const struct oc_ort *ort)
{
	return ort != NULL ? ort->version : "";
}

bool
oc_ort_has_dml(const struct oc_ort *ort)
{
	return ort != NULL && ort->has_dml;
}

bool
oc_dml_pick_adapter(uint64_t preferred_luid, uint64_t avoid_luid, int *out_index, uint64_t *out_luid)
{
#ifdef _WIN32
	IDXGIFactory1 *f = NULL;
	if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory1, (void **)&f))) {
		return false;
	}
	int first = -1, pref = -1;
	uint64_t first_luid = 0, pref_luid = 0;
	IDXGIAdapter1 *a = NULL;
	for (UINT i = 0; IDXGIFactory1_EnumAdapters1(f, i, &a) == S_OK; i++) {
		DXGI_ADAPTER_DESC1 d;
		if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &d)) && (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
			const uint64_t luid =
			    ((uint64_t)(uint32_t)d.AdapterLuid.HighPart << 32) | d.AdapterLuid.LowPart;
			if (luid != avoid_luid || avoid_luid == 0) {
				if (first < 0) {
					first = (int)i;
					first_luid = luid;
				}
				if (preferred_luid != 0 && luid == preferred_luid) {
					pref = (int)i;
					pref_luid = luid;
				}
			}
		}
		IDXGIAdapter1_Release(a);
	}
	IDXGIFactory1_Release(f);
	if (pref >= 0) {
		*out_index = pref;
		*out_luid = pref_luid;
		return true;
	}
	if (first >= 0) {
		*out_index = first;
		*out_luid = first_luid;
		return true;
	}
	return false;
#else
	(void)preferred_luid;
	(void)avoid_luid;
	(void)out_index;
	(void)out_luid;
	return false;
#endif
}

enum oc_run_status
oc_session_create(struct oc_ort *ort,
                  const char *path,
                  enum oc_ep ep,
                  int dml_device,
                  int threads,
                  enum oc_graph_opt opt,
                  struct oc_session **out,
                  char *err,
                  size_t err_size)
{
	*out = NULL;
	const OrtApi *api = ort->api;
	OrtSessionOptions *opts = NULL;
	struct oc_session *s = (struct oc_session *)calloc(1, sizeof(*s));
	if (s == NULL) {
		snprintf(err, err_size, "out of memory");
		return OC_RUN_OUT_OF_MEMORY;
	}
	s->ort = ort;
	s->ep = ep;
	TRY(api->CreateSessionOptions(&opts));
	TRY(api->SetSessionGraphOptimizationLevel(opts, opt == OC_OPT_ALL        ? ORT_ENABLE_ALL
	                                                : opt == OC_OPT_EXTENDED ? ORT_ENABLE_EXTENDED
	                                                                         : ORT_ENABLE_BASIC));
	TRY(api->SetInterOpNumThreads(opts, 1));
	if (threads > 0) {
		TRY(api->SetIntraOpNumThreads(opts, threads));
	}
	if (ep == OC_EP_DML) {
#ifdef _WIN32
		// DirectML requires these two (onnxruntime DirectML EP documentation).
		TRY(api->DisableMemPattern(opts));
		TRY(api->SetSessionExecutionMode(opts, ORT_SEQUENTIAL));
		const OrtDmlApi *dml = NULL;
		TRY(api->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **)&dml));
		if (dml == NULL) {
			snprintf(err, err_size, "no DirectML API in this ONNX Runtime");
			goto fail;
		}
		TRY(dml->SessionOptionsAppendExecutionProvider_DML(opts, dml_device));
		TRY(api->CreateMemoryInfo("DML", OrtDeviceAllocator, dml_device, OrtMemTypeDefault, &s->device_mem));
#else
		(void)dml_device;
		snprintf(err, err_size, "DirectML is Windows-only");
		goto fail;
#endif
	}
	TRY(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s->cpu_mem));
#ifdef _WIN32
	{
		wchar_t wpath[1024];
		if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024) == 0) {
			snprintf(err, err_size, "bad model path");
			goto fail;
		}
		TRY(api->CreateSession(ort->env, wpath, opts, &s->session));
	}
#else
	TRY(api->CreateSession(ort->env, path, opts, &s->session));
#endif
	api->ReleaseSessionOptions(opts);
	*out = s;
	return OC_RUN_OK;

fail:
	if (opts != NULL) {
		api->ReleaseSessionOptions(opts);
	}
	oc_session_destroy(s);
	return classify(err);
}

void
oc_session_destroy(struct oc_session *s)
{
	if (s == NULL) {
		return;
	}
	const OrtApi *api = s->ort->api;
	if (s->session != NULL) {
		api->ReleaseSession(s->session);
	}
	if (s->cpu_mem != NULL) {
		api->ReleaseMemoryInfo(s->cpu_mem);
	}
	if (s->device_mem != NULL) {
		api->ReleaseMemoryInfo(s->device_mem);
	}
	free(s);
}

//! The last two dims of a float tensor; false unless it is [..., h, w] with leading 1s.
static bool
map_dims(const OrtApi *api, OrtValue *v, uint32_t *w, uint32_t *h, char *err, size_t err_size)
{
	OrtTensorTypeAndShapeInfo *info = NULL;
	if (!status_ok(api, api->GetTensorTypeAndShape(v, &info), err, err_size)) {
		return false;
	}
	size_t n = 0;
	int64_t dims[8] = {0};
	bool ok = status_ok(api, api->GetDimensionsCount(info, &n), err, err_size) && n >= 2 && n <= 8 &&
	          status_ok(api, api->GetDimensions(info, dims, n), err, err_size);
	ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ok = ok && status_ok(api, api->GetTensorElementType(info, &type), err, err_size);
	api->ReleaseTensorTypeAndShapeInfo(info);
	if (!ok) {
		return false;
	}
	for (size_t i = 0; i + 2 < n; i++) {
		if (dims[i] != 1) {
			snprintf(err, err_size, "output has a batch / channel dim of %lld", (long long)dims[i]);
			return false;
		}
	}
	if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dims[n - 1] <= 0 || dims[n - 2] <= 0) {
		snprintf(err, err_size, "output is not a float map");
		return false;
	}
	*w = (uint32_t)dims[n - 1];
	*h = (uint32_t)dims[n - 2];
	return true;
}

enum oc_run_status
oc_session_run_image(struct oc_session *s,
                     const char *in_name,
                     const float *chw,
                     uint32_t w,
                     uint32_t h,
                     const char *out_name,
                     float *out,
                     size_t out_capacity,
                     uint32_t *out_w,
                     uint32_t *out_h,
                     char *err,
                     size_t err_size)
{
	const OrtApi *api = s->ort->api;
	OrtValue *in = NULL, *res = NULL;
	const int64_t shape[4] = {1, 3, (int64_t)h, (int64_t)w};
	float *data = NULL;
	uint32_t mw = 0, mh = 0;
	TRY(api->CreateTensorWithDataAsOrtValue(s->cpu_mem, (void *)chw, (size_t)w * h * 3u * sizeof(float), shape, 4,
	                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
	const char *in_names[1] = {in_name};
	const char *out_names[1] = {out_name};
	const OrtValue *ins[1] = {in};
	TRY(api->Run(s->session, NULL, in_names, ins, 1, out_names, 1, &res));
	if (!map_dims(api, res, &mw, &mh, err, err_size)) {
		goto fail;
	}
	if ((size_t)mw * mh > out_capacity) {
		snprintf(err, err_size, "output %ux%u exceeds the buffer", mw, mh);
		goto fail;
	}
	TRY(api->GetTensorMutableData(res, (void **)&data));
	memcpy(out, data, (size_t)mw * mh * sizeof(float));
	*out_w = mw;
	*out_h = mh;
	api->ReleaseValue(res);
	api->ReleaseValue(in);
	return OC_RUN_OK;

fail:
	if (res != NULL) {
		api->ReleaseValue(res);
	}
	if (in != NULL) {
		api->ReleaseValue(in);
	}
	return classify(err);
}

enum oc_run_status
oc_temporal_run(struct oc_session *s,
                struct oc_temporal **state,
                const struct oc_temporal_io *io,
                const float *chw,
                uint32_t w,
                uint32_t h,
                bool first,
                float *out,
                size_t out_capacity,
                uint32_t *out_w,
                uint32_t *out_h,
                char *err,
                size_t err_size)
{
	const OrtApi *api = s->ort->api;
	OrtValue *in = NULL, *flag = NULL, *zero = NULL, *res = NULL;
	OrtValue **bound = NULL;
	size_t bound_count = 0;
	float *zero_buf = NULL;
	OrtAllocator *alloc = NULL;
	float *data = NULL;
	uint32_t mw = 0, mh = 0;
	float first_value = first ? 1.0f : 0.0f;

	struct oc_temporal *t = *state;
	if (t == NULL) {
		t = (struct oc_temporal *)calloc(1, sizeof(*t));
		if (t == NULL) {
			snprintf(err, err_size, "out of memory");
			return OC_RUN_OUT_OF_MEMORY;
		}
		t->api = api;
		t->session = s->session;
		if (!status_ok(api, api->CreateIoBinding(s->session, &t->bind), err, err_size)) {
			free(t);
			return classify(err);
		}
		*state = t;
	}

	const int64_t ishape[4] = {1, 3, (int64_t)h, (int64_t)w};
	const int64_t fshape[1] = {1};
	TRY(api->CreateTensorWithDataAsOrtValue(s->cpu_mem, (void *)chw, (size_t)w * h * 3u * sizeof(float), ishape, 4,
	                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
	TRY(api->CreateTensorWithDataAsOrtValue(s->cpu_mem, &first_value, sizeof(float), fshape, 1,
	                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &flag));
	const OrtValue *cache = t->cache;
	if (cache == NULL) {
		// No history yet: any finite cache of the right shape works, because the
		// model selects its fresh state when is_first is set. Zeros, freed after
		// this run (the cache_out of this frame replaces it).
		size_t n = 1;
		for (uint32_t i = 0; i < io->cache_rank; i++) {
			n *= (size_t)io->cache_shape[i];
		}
		zero_buf = (float *)calloc(n, sizeof(float));
		if (zero_buf == NULL) {
			snprintf(err, err_size, "out of memory (temporal cache)");
			goto fail;
		}
		TRY(api->CreateTensorWithDataAsOrtValue(s->cpu_mem, zero_buf, n * sizeof(float), io->cache_shape,
		                                        io->cache_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &zero));
		cache = zero;
		first_value = 1.0f; // a stream's first frame is always first
	}

	api->ClearBoundInputs(t->bind);
	api->ClearBoundOutputs(t->bind);
	TRY(api->BindInput(t->bind, io->image, in));
	TRY(api->BindInput(t->bind, io->cache_in, cache));
	TRY(api->BindInput(t->bind, io->is_first, flag));
	TRY(api->BindOutputToDevice(t->bind, io->depth, s->cpu_mem));
	// The cache stays on the session's device between frames (no PCIe round trip).
	TRY(api->BindOutputToDevice(t->bind, io->cache_out, s->device_mem != NULL ? s->device_mem : s->cpu_mem));
	TRY(api->RunWithBinding(t->session, NULL, t->bind));
	TRY(api->GetAllocatorWithDefaultOptions(&alloc));
	TRY(api->GetBoundOutputValues(t->bind, alloc, &bound, &bound_count));
	if (bound_count != 2) {
		snprintf(err, err_size, "expected 2 bound outputs, got %zu", bound_count);
		goto fail;
	}
	res = bound[0];
	if (!map_dims(api, res, &mw, &mh, err, err_size)) {
		goto fail;
	}
	if ((size_t)mw * mh > out_capacity) {
		snprintf(err, err_size, "output %ux%u exceeds the buffer", mw, mh);
		goto fail;
	}
	TRY(api->GetTensorMutableData(res, (void **)&data));
	memcpy(out, data, (size_t)mw * mh * sizeof(float));
	*out_w = mw;
	*out_h = mh;

	// Rotate the cache: this frame's cache_out is the next frame's cache_in.
	if (t->cache != NULL) {
		api->ReleaseValue(t->cache);
	}
	t->cache = bound[1];
	bound[1] = NULL;
	api->ReleaseValue(res);
	res = NULL;
	(void)status_ok(api, api->AllocatorFree(alloc, bound), NULL, 0);
	bound = NULL;
	// Drop the binding's references to this frame's inputs before freeing them.
	api->ClearBoundInputs(t->bind);
	api->ClearBoundOutputs(t->bind);
	api->ReleaseValue(in);
	api->ReleaseValue(flag);
	if (zero != NULL) {
		api->ReleaseValue(zero);
	}
	free(zero_buf);
	return OC_RUN_OK;

fail:
	if (bound != NULL) {
		for (size_t i = 0; i < bound_count; i++) {
			if (bound[i] != NULL) {
				api->ReleaseValue(bound[i]);
			}
		}
		(void)status_ok(api, api->AllocatorFree(alloc, bound), NULL, 0);
	} else if (res != NULL) {
		api->ReleaseValue(res);
	}
	if (t->bind != NULL) {
		api->ClearBoundInputs(t->bind);
		api->ClearBoundOutputs(t->bind);
	}
	if (in != NULL) {
		api->ReleaseValue(in);
	}
	if (flag != NULL) {
		api->ReleaseValue(flag);
	}
	if (zero != NULL) {
		api->ReleaseValue(zero);
	}
	free(zero_buf);
	return classify(err);
}

void
oc_temporal_destroy(struct oc_temporal *t)
{
	if (t == NULL) {
		return;
	}
	if (t->bind != NULL) {
		t->api->ReleaseIoBinding(t->bind);
	}
	if (t->cache != NULL) {
		t->api->ReleaseValue(t->cache);
	}
	free(t);
}
