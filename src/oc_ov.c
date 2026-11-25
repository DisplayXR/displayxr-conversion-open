// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  OpenVINO glue, see oc_ov.h.
 *
 * Only the C API's types come from `openvino/c/openvino.h`; every function is
 * resolved with oc_lib_sym from the library mapped by absolute path, so the
 * provider has no link-time dependency on OpenVINO.
 */

#include "oc_ov.h"
#include "oc_platform.h"

#include <stdio.h>

#ifdef OC_OV_DISABLED

/* Built without OpenVINO (macOS): it is never there, the splat capability runs
 * on ONNX Runtime. */

const char *
oc_ov_dir_name(void)
{
	return "openvino";
}

enum oc_ov_status
oc_ov_load(const char *dir, struct oc_ov **out, char *detail, size_t detail_size)
{
	(void)dir;
	*out = NULL;
	snprintf(detail, detail_size, "this build carries no OpenVINO back end");
	return OC_OV_ABSENT;
}

const char *
oc_ov_version(const struct oc_ov *ov)
{
	(void)ov;
	return "";
}

uint32_t
oc_ov_list_gpus(struct oc_ov *ov, struct oc_ov_gpu *out, uint32_t max)
{
	(void)ov;
	(void)out;
	(void)max;
	return 0;
}

bool
oc_ov_session_create(struct oc_ov *ov,
                     const char *path,
                     const char *device,
                     const char *cache_dir,
                     int threads,
                     struct oc_ov_session **out,
                     char *err,
                     size_t err_size)
{
	(void)ov;
	(void)path;
	(void)device;
	(void)cache_dir;
	(void)threads;
	*out = NULL;
	snprintf(err, err_size, "no OpenVINO back end");
	return false;
}

void
oc_ov_session_destroy(struct oc_ov_session *s)
{
	(void)s;
}

enum oc_ov_run
oc_ov_run(struct oc_ov_session *s,
          const struct oc_tensor_in *in,
          uint32_t in_count,
          struct oc_tensor_out *out,
          uint32_t out_count,
          bool (*is_cancelled)(void *ctx),
          void *cancel_ctx,
          char *err,
          size_t err_size)
{
	(void)s;
	(void)in;
	(void)in_count;
	(void)out;
	(void)out_count;
	(void)is_cancelled;
	(void)cancel_ctx;
	snprintf(err, err_size, "no OpenVINO back end");
	return OC_OV_RUN_FAILED;
}

#else // OpenVINO

#include <openvino/c/openvino.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OC_OV_POLL_MS 100

/* The C API, as function pointers (signatures from openvino/c/ov_*.h). */
typedef ov_status_e (*pfn_get_version)(ov_version_t *);
typedef void (*pfn_version_free)(ov_version_t *);
typedef ov_status_e (*pfn_core_create)(ov_core_t **);
typedef void (*pfn_core_free)(ov_core_t *);
typedef ov_status_e (*pfn_core_devices)(const ov_core_t *, ov_available_devices_t *);
typedef void (*pfn_devices_free)(ov_available_devices_t *);
typedef ov_status_e (*pfn_core_get_property)(const ov_core_t *, const char *, const char *, char **);
typedef void (*pfn_free)(const char *);
typedef ov_status_e (*pfn_compile_from_file)(
    const ov_core_t *, const char *, const char *, const size_t, ov_compiled_model_t **, ...);
typedef void (*pfn_compiled_free)(ov_compiled_model_t *);
typedef ov_status_e (*pfn_create_request)(const ov_compiled_model_t *, ov_infer_request_t **);
typedef void (*pfn_request_free)(ov_infer_request_t *);
typedef ov_status_e (*pfn_set_tensor)(ov_infer_request_t *, const char *, const ov_tensor_t *);
typedef ov_status_e (*pfn_get_tensor)(const ov_infer_request_t *, const char *, ov_tensor_t **);
typedef ov_status_e (*pfn_request_simple)(ov_infer_request_t *);
typedef ov_status_e (*pfn_wait_for)(ov_infer_request_t *, const int64_t);
typedef ov_status_e (*pfn_tensor_from_ptr)(const ov_element_type_e, const ov_shape_t, void *, ov_tensor_t **);
typedef ov_status_e (*pfn_tensor_data)(const ov_tensor_t *, void **);
typedef ov_status_e (*pfn_tensor_size)(const ov_tensor_t *, size_t *);
typedef ov_status_e (*pfn_tensor_type)(const ov_tensor_t *, ov_element_type_e *);
typedef void (*pfn_tensor_free)(ov_tensor_t *);
typedef ov_status_e (*pfn_shape_create)(const int64_t, const int64_t *, ov_shape_t *);
typedef ov_status_e (*pfn_shape_free)(ov_shape_t *);
typedef const char *(*pfn_last_err)(void);
typedef const char *(*pfn_error_info)(ov_status_e);

struct oc_ov
{
	void *libs[4]; //!< mapped in dependency order; never unmapped
	ov_core_t *core;
	char version[64];

	pfn_get_version get_version;
	pfn_version_free version_free;
	pfn_core_create core_create;
	pfn_core_free core_free;
	pfn_core_devices core_devices;
	pfn_devices_free devices_free;
	pfn_core_get_property core_get_property;
	pfn_free free_str;
	pfn_compile_from_file compile_from_file;
	pfn_compiled_free compiled_free;
	pfn_create_request create_request;
	pfn_request_free request_free;
	pfn_set_tensor set_tensor;
	pfn_get_tensor get_tensor;
	pfn_request_simple start_async;
	pfn_request_simple cancel;
	pfn_wait_for wait_for;
	pfn_tensor_from_ptr tensor_from_ptr;
	pfn_tensor_data tensor_data;
	pfn_tensor_size tensor_size;
	pfn_tensor_type tensor_type;
	pfn_tensor_free tensor_free;
	pfn_shape_create shape_create;
	pfn_shape_free shape_free;
	pfn_last_err last_err;     //!< optional
	pfn_error_info error_info; //!< optional
};

struct oc_ov_session
{
	struct oc_ov *ov;
	ov_compiled_model_t *compiled;
	ov_infer_request_t *request;
};

const char *
oc_ov_dir_name(void)
{
	return "openvino";
}

const char *
oc_ov_version(const struct oc_ov *ov)
{
	return ov != NULL ? ov->version : "";
}

//! A sentence for @p st: OpenVINO's last error message when it has one.
static void
ov_error(const struct oc_ov *ov, ov_status_e st, const char *what, char *err, size_t err_size)
{
	const char *msg = ov->last_err != NULL ? ov->last_err() : NULL;
	if (msg == NULL || msg[0] == '\0') {
		msg = ov->error_info != NULL ? ov->error_info(st) : "error";
	}
	// OpenVINO's messages carry a multi-line trace; keep the first line.
	char line[256];
	oc_copy_str(line, sizeof(line), msg != NULL ? msg : "error");
	for (char *c = line; *c != '\0'; c++) {
		if (*c == '\n' || *c == '\r') {
			*c = '\0';
			break;
		}
	}
	snprintf(err, err_size, "%s: %s (%d)", what, line, (int)st);
}

static bool
map_one(struct oc_ov *ov, size_t slot, const char *dir, const char *name, char *detail, size_t detail_size)
{
	char path[1400], err[200];
	snprintf(path, sizeof(path), "%s%c%s", dir, OC_PATH_SEP, name);
	ov->libs[slot] = oc_lib_open(path, err, sizeof(err));
	if (ov->libs[slot] == NULL) {
		snprintf(detail, detail_size, "OpenVINO did not load (%s): %s", name, err);
		return false;
	}
	return true;
}

enum oc_ov_status
oc_ov_load(const char *dir, struct oc_ov **out, char *detail, size_t detail_size)
{
	*out = NULL;
#if defined(_WIN32)
	static const char *const chain[] = {"openvino_c.dll"};
#elif defined(__APPLE__)
	static const char *const chain[] = {"libopenvino_c.dylib"};
#else
	// No RUNPATH in the official libraries: map each dependency by absolute path
	// first, so the next one's DT_NEEDED resolves to the copy already mapped.
	static const char *const chain[] = {"libtbb.so.12", "libopenvino.so." OC_OV_SOVERSION,
	                                    "libopenvino_c.so." OC_OV_SOVERSION};
#endif
	const size_t n_chain = sizeof(chain) / sizeof(chain[0]);
	char c_api[1400];
	snprintf(c_api, sizeof(c_api), "%s%c%s", dir != NULL ? dir : "", OC_PATH_SEP, chain[n_chain - 1]);
	if (dir == NULL || dir[0] == '\0' || !oc_file_exists(c_api)) {
		snprintf(detail, detail_size, "OpenVINO not found (%s)", c_api);
		return OC_OV_ABSENT;
	}
	struct oc_ov *ov = (struct oc_ov *)calloc(1, sizeof(*ov));
	if (ov == NULL) {
		snprintf(detail, detail_size, "out of memory");
		return OC_OV_BROKEN;
	}
	for (size_t i = 0; i < n_chain; i++) {
		if (!map_one(ov, i, dir, chain[i], detail, detail_size)) {
			free(ov); // mapped libraries stay mapped (never unloaded within a process)
			return OC_OV_BROKEN;
		}
	}
	void *lib = ov->libs[n_chain - 1];
#define SYM(FIELD, NAME, TYPE)                                                                                         \
	do {                                                                                                           \
		ov->FIELD = (TYPE)oc_lib_sym(lib, NAME);                                                               \
		if (ov->FIELD == NULL) {                                                                               \
			snprintf(detail, detail_size, "OpenVINO C API lacks %s", NAME);                                \
			free(ov);                                                                                      \
			return OC_OV_TOO_OLD;                                                                          \
		}                                                                                                      \
	} while (0)
	SYM(get_version, "ov_get_openvino_version", pfn_get_version);
	SYM(version_free, "ov_version_free", pfn_version_free);
	SYM(core_create, "ov_core_create", pfn_core_create);
	SYM(core_free, "ov_core_free", pfn_core_free);
	SYM(core_devices, "ov_core_get_available_devices", pfn_core_devices);
	SYM(devices_free, "ov_available_devices_free", pfn_devices_free);
	SYM(core_get_property, "ov_core_get_property", pfn_core_get_property);
	SYM(free_str, "ov_free", pfn_free);
	SYM(compile_from_file, "ov_core_compile_model_from_file", pfn_compile_from_file);
	SYM(compiled_free, "ov_compiled_model_free", pfn_compiled_free);
	SYM(create_request, "ov_compiled_model_create_infer_request", pfn_create_request);
	SYM(request_free, "ov_infer_request_free", pfn_request_free);
	SYM(set_tensor, "ov_infer_request_set_tensor", pfn_set_tensor);
	SYM(get_tensor, "ov_infer_request_get_tensor", pfn_get_tensor);
	SYM(start_async, "ov_infer_request_start_async", pfn_request_simple);
	SYM(cancel, "ov_infer_request_cancel", pfn_request_simple);
	SYM(wait_for, "ov_infer_request_wait_for", pfn_wait_for);
	SYM(tensor_from_ptr, "ov_tensor_create_from_host_ptr", pfn_tensor_from_ptr);
	SYM(tensor_data, "ov_tensor_data", pfn_tensor_data);
	SYM(tensor_size, "ov_tensor_get_size", pfn_tensor_size);
	SYM(tensor_type, "ov_tensor_get_element_type", pfn_tensor_type);
	SYM(tensor_free, "ov_tensor_free", pfn_tensor_free);
	SYM(shape_create, "ov_shape_create", pfn_shape_create);
	SYM(shape_free, "ov_shape_free", pfn_shape_free);
#undef SYM
	ov->last_err = (pfn_last_err)oc_lib_sym(lib, "ov_get_last_err_msg");
	ov->error_info = (pfn_error_info)oc_lib_sym(lib, "ov_get_error_info");

	ov_version_t v;
	memset(&v, 0, sizeof(v));
	if (ov->get_version(&v) != OK || v.buildNumber == NULL) {
		snprintf(detail, detail_size, "OpenVINO reports no version");
		free(ov);
		return OC_OV_BROKEN;
	}
	oc_copy_str(ov->version, sizeof(ov->version), v.buildNumber);
	ov->version_free(&v);
	int year = 0, release = 0;
	if (sscanf(ov->version, "%d.%d", &year, &release) != 2 ||
	    year * 100 + release < OC_OV_MIN_YEAR * 100 + OC_OV_MIN_RELEASE) {
		snprintf(detail, detail_size, "OpenVINO %.40s is older than %d.%d", ov->version, OC_OV_MIN_YEAR,
		         OC_OV_MIN_RELEASE);
		free(ov);
		return OC_OV_TOO_OLD;
	}
	const ov_status_e st = ov->core_create(&ov->core);
	if (st != OK || ov->core == NULL) {
		ov_error(ov, st, "OpenVINO core", detail, detail_size);
		free(ov);
		return OC_OV_BROKEN;
	}
	// The version up to its first '-' ("2026.4.1").
	char *dash = strchr(ov->version, '-');
	if (dash != NULL) {
		*dash = '\0';
	}
	snprintf(detail, detail_size, "OpenVINO %s", ov->version);
	*out = ov;
	return OC_OV_OK;
}

//! A string property of @p device into @p buf; false when not reported.
static bool
ov_prop(struct oc_ov *ov, const char *device, const char *key, char *buf, size_t size)
{
	char *v = NULL;
	buf[0] = '\0';
	if (ov->core_get_property(ov->core, device, key, &v) != OK || v == NULL) {
		return false;
	}
	oc_copy_str(buf, size, v);
	ov->free_str(v);
	return true;
}

uint32_t
oc_ov_list_gpus(struct oc_ov *ov, struct oc_ov_gpu *out, uint32_t max)
{
	if (ov == NULL) {
		return 0;
	}
	ov_available_devices_t d;
	memset(&d, 0, sizeof(d));
	if (ov->core_devices(ov->core, &d) != OK) {
		return 0;
	}
	uint32_t n = 0;
	for (size_t i = 0; i < d.size && n < max; i++) {
		if (d.devices[i] == NULL || strncmp(d.devices[i], "GPU", 3) != 0) {
			continue;
		}
		struct oc_ov_gpu *g = &out[n++];
		memset(g, 0, sizeof(*g));
		oc_copy_str(g->device, sizeof(g->device), d.devices[i]);
		if (!ov_prop(ov, d.devices[i], "FULL_DEVICE_NAME", g->name, sizeof(g->name))) {
			oc_copy_str(g->name, sizeof(g->name), d.devices[i]);
		}
		char buf[64];
		if (ov_prop(ov, d.devices[i], "DEVICE_TYPE", buf, sizeof(buf))) {
			// ov::device::Type: "integrated" / "discrete" (older releases: "INTEGRATED").
			g->type = (buf[0] == 'd' || buf[0] == 'D')   ? OC_OV_GPU_TYPE_DISCRETE
			          : (buf[0] == 'i' || buf[0] == 'I') ? OC_OV_GPU_TYPE_INTEGRATED
			                                             : OC_OV_GPU_TYPE_UNKNOWN;
		}
		if (ov_prop(ov, d.devices[i], "GPU_EXECUTION_UNITS_COUNT", buf, sizeof(buf))) {
			g->eu_count = (uint32_t)strtoul(buf, NULL, 10);
		}
		if (ov_prop(ov, d.devices[i], "DEVICE_ARCHITECTURE", buf, sizeof(buf))) {
			oc_rank_parse_ov_arch(buf, g);
		}
		char caps[256];
		if (ov_prop(ov, d.devices[i], "OPTIMIZATION_CAPABILITIES", caps, sizeof(caps))) {
			g->hw_matmul = strstr(caps, "GPU_HW_MATMUL") != NULL;
		}
	}
	ov->devices_free(&d);
	return n;
}

bool
oc_ov_session_create(struct oc_ov *ov,
                     const char *path,
                     const char *device,
                     const char *cache_dir,
                     int threads,
                     struct oc_ov_session **out,
                     char *err,
                     size_t err_size)
{
	*out = NULL;
	struct oc_ov_session *s = (struct oc_ov_session *)calloc(1, sizeof(*s));
	if (s == NULL) {
		snprintf(err, err_size, "out of memory");
		return false;
	}
	s->ov = ov;
	char nthreads[16];
	snprintf(nthreads, sizeof(nthreads), "%d", threads);
	const bool cpu = strcmp(device, "CPU") == 0;
	const bool cache = cache_dir != NULL && cache_dir[0] != '\0';
	// Properties for this compile only, as <key, value> string pairs; the count
	// is the number of arguments (twice the pairs).
	ov_status_e st;
	if (cache && cpu && threads > 0) {
		st = ov->compile_from_file(ov->core, path, device, 6, &s->compiled, "CACHE_DIR", cache_dir,
		                           "PERFORMANCE_HINT", "LATENCY", "INFERENCE_NUM_THREADS", nthreads);
	} else if (cache) {
		st = ov->compile_from_file(ov->core, path, device, 4, &s->compiled, "CACHE_DIR", cache_dir,
		                           "PERFORMANCE_HINT", "LATENCY");
	} else if (cpu && threads > 0) {
		st = ov->compile_from_file(ov->core, path, device, 4, &s->compiled, "PERFORMANCE_HINT", "LATENCY",
		                           "INFERENCE_NUM_THREADS", nthreads);
	} else {
		st = ov->compile_from_file(ov->core, path, device, 2, &s->compiled, "PERFORMANCE_HINT", "LATENCY");
	}
	if (st != OK || s->compiled == NULL) {
		ov_error(ov, st, "compile", err, err_size);
		oc_ov_session_destroy(s);
		return false;
	}
	st = ov->create_request(s->compiled, &s->request);
	if (st != OK || s->request == NULL) {
		ov_error(ov, st, "infer request", err, err_size);
		oc_ov_session_destroy(s);
		return false;
	}
	*out = s;
	return true;
}

void
oc_ov_session_destroy(struct oc_ov_session *s)
{
	if (s == NULL) {
		return;
	}
	if (s->request != NULL) {
		s->ov->request_free(s->request);
	}
	if (s->compiled != NULL) {
		s->ov->compiled_free(s->compiled);
	}
	free(s);
}

enum oc_ov_run
oc_ov_run(struct oc_ov_session *s,
          const struct oc_tensor_in *in,
          uint32_t in_count,
          struct oc_tensor_out *out,
          uint32_t out_count,
          bool (*is_cancelled)(void *ctx),
          void *cancel_ctx,
          char *err,
          size_t err_size)
{
	struct oc_ov *ov = s->ov;
	enum oc_ov_run result = OC_OV_RUN_FAILED;
	ov_tensor_t *tensors[8] = {0};
	ov_status_e st = OK;
	if (in_count > 8) {
		snprintf(err, err_size, "too many inputs");
		return OC_OV_RUN_FAILED;
	}
	for (uint32_t i = 0; i < in_count; i++) {
		ov_shape_t shape;
		memset(&shape, 0, sizeof(shape));
		st = ov->shape_create((int64_t)in[i].rank, in[i].shape, &shape);
		if (st != OK) {
			ov_error(ov, st, "shape", err, err_size);
			goto done;
		}
		// The host buffer is only read: OpenVINO wraps it without a copy.
		st = ov->tensor_from_ptr(F32, shape, (void *)in[i].data, &tensors[i]);
		ov->shape_free(&shape);
		if (st != OK) {
			ov_error(ov, st, "input tensor", err, err_size);
			goto done;
		}
		st = ov->set_tensor(s->request, in[i].name, tensors[i]);
		if (st != OK) {
			ov_error(ov, st, in[i].name, err, err_size);
			goto done;
		}
	}
	st = ov->start_async(s->request);
	if (st != OK) {
		ov_error(ov, st, "infer", err, err_size);
		goto done;
	}
	for (;;) {
		st = ov->wait_for(s->request, OC_OV_POLL_MS);
		if (st == OK) {
			break;
		}
		if (st != RESULT_NOT_READY) {
			ov_error(ov, st, "infer", err, err_size);
			goto done;
		}
		if (is_cancelled != NULL && is_cancelled(cancel_ctx)) {
			(void)ov->cancel(s->request);
			(void)ov->wait_for(s->request, 10000);
			snprintf(err, err_size, "cancelled");
			result = OC_OV_RUN_CANCELLED;
			goto done;
		}
	}
	for (uint32_t i = 0; i < out_count; i++) {
		ov_tensor_t *t = NULL;
		void *data = NULL;
		size_t count = 0;
		ov_element_type_e type = DYNAMIC;
		st = ov->get_tensor(s->request, out[i].name, &t);
		if (st == OK) {
			st = ov->tensor_type(t, &type);
		}
		if (st == OK) {
			st = ov->tensor_size(t, &count);
		}
		if (st == OK) {
			st = ov->tensor_data(t, &data);
		}
		if (st != OK || type != F32 || count > out[i].capacity || data == NULL) {
			if (st != OK) {
				ov_error(ov, st, out[i].name, err, err_size);
			} else {
				snprintf(err, err_size, "output %s: %zu elements of type %d (capacity %zu f32)",
				         out[i].name, count, (int)type, out[i].capacity);
			}
			if (t != NULL) {
				ov->tensor_free(t);
			}
			goto done;
		}
		memcpy(out[i].data, data, count * sizeof(float));
		out[i].count = count;
		ov->tensor_free(t);
	}
	result = OC_OV_RUN_OK;

done:
	for (uint32_t i = 0; i < in_count; i++) {
		if (tensors[i] != NULL) {
			ov->tensor_free(tensors[i]);
		}
	}
	return result;
}

#endif // OC_OV_DISABLED
