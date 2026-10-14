// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  A fake host for the open conversion provider: drives the provider
 *         binary through the runtime's call sequence with no runtime
 *         installed, plays the model store from a schema-2 pack, and checks
 *         the contract on the way.
 *
 *     open-conversion-fakehost <provider binary> [options]
 *
 *       --set KEY=VALUE        provider setting (also DXR_CONV_OPEN_CONVERSION_<KEY>)
 *       --pack FILE            the pack the store serves (also passed as setting PACK)
 *       --blob-base URL        override the pack's blobBaseUrl (file:// only)
 *       --store DIR            model-store root (blobs/<sha256>, <sha256>.ok stamp last)
 *       --clean-store          delete the pack's blobs and stamps from the store first
 *       --no-download          model_request refuses (download forbidden by policy)
 *       --chunk-ms N           delay per 4 MiB copied, so DOWNLOADING is observable
 *       --caps video,mono      the capabilities to warm and convert (default: both)
 *       --frames N             frames per stream (default 3; a reset mid-stream)
 *       --no-warm              describe and readiness only
 *       --expect ready|absent|version|policy
 *       --expect-licence permissive|commercial|research
 *       --expect-download      the provider must report DOWNLOADING before READY
 *       --warm-timeout-ms N
 *
 * Exit code: 0 = every check passed, 1 = a check failed, 2 = usage / load
 * error, 77 = skipped (a model blob the test needs is not available).
 */

#include "xrt/xrt_conversion_plugin.h"

#include "oc_json.h"
#include "oc_pack.h"
#include "oc_platform.h"
#include "sha256.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#else
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define SKIP_CODE 77


/*
 *
 * Reporting.
 *
 */

static int g_failures = 0;

static void
check(bool ok, const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	printf("[%s] ", ok ? "PASS" : "FAIL");
	vprintf(fmt, ap);
	printf("\n");
	va_end(ap);
	fflush(stdout);
	if (!ok) {
		g_failures++;
	}
}

static void
info(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	printf("       ");
	vprintf(fmt, ap);
	printf("\n");
	va_end(ap);
	fflush(stdout);
}


/*
 *
 * Portability: threads and files with UTF-8 paths.
 *
 */

typedef void (*thread_fn)(void *);

struct thread_start
{
	thread_fn fn;
	void *arg;
};

#ifdef _WIN32
static unsigned __stdcall thread_tramp(void *p)
{
	struct thread_start s = *(struct thread_start *)p;
	free(p);
	s.fn(s.arg);
	return 0;
}
typedef HANDLE thread_t;
#else
static void *
thread_tramp(void *p)
{
	struct thread_start s = *(struct thread_start *)p;
	free(p);
	s.fn(s.arg);
	return NULL;
}
typedef pthread_t thread_t;
#endif

static bool
thread_start(thread_t *t, thread_fn fn, void *arg)
{
	struct thread_start *s = (struct thread_start *)malloc(sizeof(*s));
	if (s == NULL) {
		return false;
	}
	s->fn = fn;
	s->arg = arg;
#ifdef _WIN32
	*t = (HANDLE)_beginthreadex(NULL, 0, thread_tramp, s, 0, NULL);
	return *t != NULL;
#else
	return pthread_create(t, NULL, thread_tramp, s) == 0;
#endif
}

static void
thread_join(thread_t t)
{
#ifdef _WIN32
	WaitForSingleObject(t, INFINITE);
	CloseHandle(t);
#else
	pthread_join(t, NULL);
#endif
}

static FILE *
fopen_utf8(const char *path, const char *mode)
{
#ifdef _WIN32
	wchar_t wp[1024], wm[8];
	if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 1024) == 0 ||
	    MultiByteToWideChar(CP_UTF8, 0, mode, -1, wm, 8) == 0) {
		return NULL;
	}
	return _wfopen(wp, wm);
#else
	return fopen(path, mode);
#endif
}

static bool
rename_utf8(const char *from, const char *to)
{
#ifdef _WIN32
	wchar_t wf[1024], wt[1024];
	return MultiByteToWideChar(CP_UTF8, 0, from, -1, wf, 1024) != 0 &&
	       MultiByteToWideChar(CP_UTF8, 0, to, -1, wt, 1024) != 0 &&
	       MoveFileExW(wf, wt, MOVEFILE_REPLACE_EXISTING) != 0;
#else
	return rename(from, to) == 0;
#endif
}

static void
remove_utf8(const char *path)
{
#ifdef _WIN32
	wchar_t w[1024];
	if (MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 1024) != 0) {
		_wremove(w);
	}
#else
	remove(path);
#endif
}

static void
mkdir_utf8(const char *path)
{
#ifdef _WIN32
	wchar_t w[1024];
	if (MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 1024) != 0) {
		_wmkdir(w);
	}
#else
	mkdir(path, 0755);
#endif
}

static int64_t
file_size(const char *path)
{
	FILE *f = fopen_utf8(path, "rb");
	if (f == NULL) {
		return -1;
	}
	int64_t n = -1;
#ifdef _WIN32
	if (_fseeki64(f, 0, SEEK_END) == 0) {
		n = _ftelli64(f);
	}
#else
	if (fseeko(f, 0, SEEK_END) == 0) {
		n = (int64_t)ftello(f);
	}
#endif
	fclose(f);
	return n;
}


/*
 *
 * The model store (the host's side of discovery spec §5, file:// only).
 *
 */

struct blob
{
	char sha256[65];
	char format[16];
	uint64_t size;
	char refs[4][192]; //!< "<pack>/<model>/<variant>" and "<pack>/<model>"

	/* Download state, under g_store.lock. */
	uint32_t state; //!< XRT_CONV_STATE_ABSENT / DOWNLOADING / READY / FAILED
	uint32_t failure_reason;
	float progress;
	bool thread_started;
	thread_t thread;
};

static struct
{
	oc_mutex lock;
	bool enabled; //!< --pack given
	bool no_download;
	bool clean; //!< --clean-store: start from an empty store
	uint32_t chunk_ms;
	char root[384]; //!< short enough that root/blobs/<sha256>.ok fits xrt_conv_model_location::path
	char blob_base[1024];
	struct blob blobs[OC_PACK_MAX_MODELS];
	uint32_t blob_count;
	uint32_t requests;
	uint32_t downloads_done;
} g_store;

static void
blob_path(const struct blob *b, char *buf, size_t n)
{
	snprintf(buf, n, "%s/blobs/%s", g_store.root, b->sha256);
}

static void
stamp_path(const struct blob *b, char *buf, size_t n)
{
	snprintf(buf, n, "%s/blobs/%s.ok", g_store.root, b->sha256);
}

//! Read check: the stamp and the size (the runtime's host re-hashes on first load).
static bool
blob_present(const struct blob *b)
{
	char p[1100], s[1100];
	blob_path(b, p, sizeof(p));
	stamp_path(b, s, sizeof(s));
	return oc_file_exists(s) && file_size(p) == (int64_t)b->size;
}

static struct blob *
find_blob(const char *ref)
{
	for (uint32_t i = 0; i < g_store.blob_count; i++) {
		struct blob *b = &g_store.blobs[i];
		if (strcmp(b->sha256, ref) == 0) {
			return b;
		}
		for (int r = 0; r < 4; r++) {
			if (b->refs[r][0] != '\0' && strcmp(b->refs[r], ref) == 0) {
				return b;
			}
		}
	}
	return NULL;
}

static bool
url_to_path(const char *url, char *out, size_t n)
{
	if (strncmp(url, "file://", 7) != 0) {
		return false;
	}
	const char *p = url + 7;
	// file:///C:/x -> C:/x ; file:///home/x -> /home/x
	if (p[0] == '/' && isalpha((unsigned char)p[1]) && p[2] == ':') {
		p++;
	}
	snprintf(out, n, "%s", p);
	return true;
}

static void
download_main(void *arg)
{
	struct blob *b = (struct blob *)arg;
	char url[1200], src_path[1200], dst[1100], part[1200], stamp[1100];
	snprintf(url, sizeof(url), "%s/%s.%s", g_store.blob_base, b->sha256, b->format);
	blob_path(b, dst, sizeof(dst));
	stamp_path(b, stamp, sizeof(stamp));
	snprintf(part, sizeof(part), "%s.%u.part", dst, (unsigned)(oc_now_ms() & 0xffffu));
	uint32_t reason = XRT_CONV_FAIL_NETWORK;
	FILE *in = url_to_path(url, src_path, sizeof(src_path)) ? fopen_utf8(src_path, "rb") : NULL;
	FILE *out = in != NULL ? fopen_utf8(part, "wb") : NULL;
	bool ok = in != NULL && out != NULL;
	if (ok) {
		struct sha256_ctx h;
		sha256_init(&h);
		uint8_t *buf = (uint8_t *)malloc(4u << 20);
		uint64_t done = 0;
		size_t n = 0;
		ok = buf != NULL;
		while (ok && (n = fread(buf, 1, 4u << 20, in)) > 0) {
			ok = fwrite(buf, 1, n, out) == n;
			sha256_update(&h, buf, n);
			done += n;
			oc_lock(&g_store.lock);
			b->progress = b->size > 0 ? (float)((double)done / (double)b->size) : -1.0f;
			if (b->progress > 0.99f) {
				b->progress = 0.99f;
			}
			oc_unlock(&g_store.lock);
			if (g_store.chunk_ms > 0) {
				oc_sleep_ms(g_store.chunk_ms);
			}
		}
		free(buf);
		char hex[65];
		sha256_final_hex(&h, hex);
		// Verify size, then sha256; only then rename and stamp (stamp last).
		if (ok && done != b->size) {
			info("store: %s size %llu != %llu", b->sha256, (unsigned long long)done,
			     (unsigned long long)b->size);
			ok = false;
			reason = XRT_CONV_FAIL_MODEL;
		} else if (ok && strcmp(hex, b->sha256) != 0) {
			info("store: %s hash mismatch (%s)", b->sha256, hex);
			ok = false;
			reason = XRT_CONV_FAIL_MODEL;
		}
	}
	if (in != NULL) {
		fclose(in);
	}
	if (out != NULL) {
		ok = fclose(out) == 0 && ok;
	}
	if (ok) {
		ok = rename_utf8(part, dst);
		FILE *s = ok ? fopen_utf8(stamp, "wb") : NULL;
		ok = s != NULL && fprintf(s, "%llu\n", (unsigned long long)b->size) > 0;
		if (s != NULL) {
			ok = fclose(s) == 0 && ok;
		}
	} else {
		remove_utf8(part);
	}
	oc_lock(&g_store.lock);
	b->state = ok ? XRT_CONV_STATE_READY : XRT_CONV_STATE_FAILED;
	b->failure_reason = ok ? XRT_CONV_FAIL_NONE : reason;
	b->progress = ok ? 1.0f : -1.0f;
	g_store.downloads_done += ok ? 1u : 0u;
	oc_unlock(&g_store.lock);
	info("store: %s %s", b->sha256, ok ? "downloaded, verified, stamped" : "download FAILED");
}

static bool
host_model_resolve(const char *ref, struct xrt_conv_model_location *out)
{
	if (!g_store.enabled || ref == NULL || out == NULL) {
		return false;
	}
	oc_lock(&g_store.lock);
	struct blob *b = find_blob(ref);
	if (b == NULL) {
		oc_unlock(&g_store.lock);
		return false;
	}
	if (b->state != XRT_CONV_STATE_DOWNLOADING && b->state != XRT_CONV_STATE_FAILED) {
		b->state = blob_present(b) ? XRT_CONV_STATE_READY : XRT_CONV_STATE_ABSENT; // a wiped store reads ABSENT
	}
	if (XRT_CONV_COVERS(out, struct xrt_conv_model_location, state)) {
		out->state = b->state;
	}
	if (XRT_CONV_COVERS(out, struct xrt_conv_model_location, progress)) {
		out->progress = b->state == XRT_CONV_STATE_DOWNLOADING ? b->progress : -1.0f;
	}
	if (XRT_CONV_COVERS(out, struct xrt_conv_model_location, failure_reason)) {
		out->failure_reason = b->failure_reason;
	}
	if (XRT_CONV_COVERS(out, struct xrt_conv_model_location, path)) {
		out->path[0] = '\0';
		if (b->state == XRT_CONV_STATE_READY) {
			blob_path(b, out->path, sizeof(out->path));
		}
	}
	oc_unlock(&g_store.lock);
	return true;
}

static bool
host_model_request(const char *ref)
{
	if (!g_store.enabled || ref == NULL) {
		return false;
	}
	oc_lock(&g_store.lock);
	g_store.requests++;
	struct blob *b = find_blob(ref);
	if (b == NULL || g_store.no_download) {
		oc_unlock(&g_store.lock);
		return false; // unknown ref, or forbidden by policy
	}
	if (blob_present(b) || b->state == XRT_CONV_STATE_DOWNLOADING) {
		oc_unlock(&g_store.lock);
		return true; // present, or one download shared by every request
	}
	if (b->thread_started) {
		thread_join(b->thread); // a finished, failed attempt
		b->thread_started = false;
	}
	b->state = XRT_CONV_STATE_DOWNLOADING;
	b->progress = 0.0f;
	b->failure_reason = 0;
	b->thread_started = thread_start(&b->thread, download_main, b);
	if (!b->thread_started) {
		b->state = XRT_CONV_STATE_FAILED;
		b->failure_reason = XRT_CONV_FAIL_RESOURCES;
	}
	oc_unlock(&g_store.lock);
	return true;
}

static bool
store_setup(const char *pack_path, const char *blob_base, const char *root)
{
	struct oc_pack *pack = (struct oc_pack *)calloc(1, sizeof(*pack));
	char err[200] = "";
	if (pack == NULL || !oc_pack_load(pack_path, pack, err, sizeof(err))) {
		fprintf(stderr, "store: %s\n", err);
		free(pack);
		return false;
	}
	snprintf(g_store.blob_base, sizeof(g_store.blob_base), "%s",
	         blob_base != NULL ? blob_base : pack->blob_base_url);
	if (strlen(root) >= sizeof(g_store.root)) {
		fprintf(stderr, "store: root path longer than %u bytes\n", (unsigned)sizeof(g_store.root) - 1);
		free(pack);
		return false;
	}
	oc_copy_str(g_store.root, sizeof(g_store.root), root);
	mkdir_utf8(g_store.root);
	char blobs[1100];
	snprintf(blobs, sizeof(blobs), "%s/blobs", g_store.root);
	mkdir_utf8(blobs);
	for (uint32_t i = 0; i < pack->model_count; i++) {
		const struct oc_pack_model *m = &pack->models[i];
		for (uint32_t v = 0; v < m->variant_count; v++) {
			const struct oc_pack_variant *var = &m->variants[v];
			struct blob *b = find_blob(var->file.sha256);
			if (b == NULL && g_store.blob_count < OC_PACK_MAX_MODELS) {
				b = &g_store.blobs[g_store.blob_count++];
				snprintf(b->sha256, sizeof(b->sha256), "%s", var->file.sha256);
				snprintf(b->format, sizeof(b->format), "%s", var->file.format);
				b->size = var->file.size;
				b->state = XRT_CONV_STATE_ABSENT;
			}
			if (b == NULL) {
				continue;
			}
			char refs[2][192];
			oc_pack_ref(pack, m, var, refs[0], sizeof(refs[0]));
			snprintf(refs[1], sizeof(refs[1]), "%s/%s", pack->pack, m->name); // "<pack>/<model>"
			for (int k = 0; k < (v == 0 ? 2 : 1); k++) {
				for (int r = 0; r < 4; r++) {
					if (b->refs[r][0] == '\0') {
						oc_copy_str(b->refs[r], sizeof(b->refs[r]), refs[k]);
						break;
					}
				}
			}
		}
	}
	free(pack);
	if (g_store.clean) {
		for (uint32_t i = 0; i < g_store.blob_count; i++) {
			char path[1100];
			stamp_path(&g_store.blobs[i], path,
			           sizeof(path)); // stamp first: never a stamp without its blob
			remove_utf8(path);
			blob_path(&g_store.blobs[i], path, sizeof(path));
			remove_utf8(path);
		}
	}
	g_store.enabled = true;
	return true;
}

//! True when the source blob for @p sha256 is reachable (file:// only).
static bool
source_available(const struct blob *b)
{
	char url[1200], path[1200];
	snprintf(url, sizeof(url), "%s/%s.%s", g_store.blob_base, b->sha256, b->format);
	return url_to_path(url, path, sizeof(path)) && file_size(path) == (int64_t)b->size;
}


/*
 *
 * The rest of the stub host iface.
 *
 */

#define MAX_SETTINGS 32

static struct
{
	char key[64];
	char value[1024];
} g_settings[MAX_SETTINGS];
static int g_setting_count = 0;
static char g_env_prefix[96] = "";
static volatile int g_notifications = 0;

static bool
host_get_path(uint32_t kind, char *buf, uint32_t buf_size)
{
	if (buf == NULL || buf_size == 0) {
		return false;
	}
	if (kind == XRT_CONV_PATH_MODEL_STORE && g_store.enabled) {
		snprintf(buf, buf_size, "%s", g_store.root);
		return true;
	}
	return false;
}

static bool
host_get_setting(const char *key, char *buf, uint32_t buf_size)
{
	if (key == NULL || buf == NULL || buf_size == 0) {
		return false;
	}
	char env_name[192];
	snprintf(env_name, sizeof(env_name), "%s%s", g_env_prefix, key);
	const char *e = getenv(env_name);
	if (e != NULL) {
		snprintf(buf, buf_size, "%s", e);
		return true;
	}
	for (int i = 0; i < g_setting_count; i++) {
		if (strcmp(g_settings[i].key, key) == 0) {
			snprintf(buf, buf_size, "%s", g_settings[i].value);
			return true;
		}
	}
	return false;
}

static bool
host_get_adapter_hint(struct xrt_conv_adapter_hint *inout)
{
	if (inout == NULL) {
		return false;
	}
	inout->flags = 0;
	inout->preferred_luid = 0;
	inout->avoid_luid = 0;
	return true;
}

static void
host_notify_changed(uint32_t cap_id)
{
	(void)cap_id;
	g_notifications++;
}

static struct xrt_conversion_plugin_host_iface g_host = {
    .struct_size = (uint32_t)sizeof(struct xrt_conversion_plugin_host_iface),
    .host_api_version = XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT,
    .host_flags = 2u, // bit1: headless / CI
    .get_path = host_get_path,
    .get_setting = host_get_setting,
    .get_adapter_hint = host_get_adapter_hint,
    .model_resolve = host_model_resolve,
    .model_request = host_model_request,
    .notify_changed = host_notify_changed,
};

static bool
never_cancelled(void *ctx)
{
	(void)ctx;
	return false;
}


/*
 *
 * Names, descriptor validation.
 *
 */

static const char *
task_name(uint32_t t)
{
	static const char *names[] = {"none",       "depth-mono",       "depth-video", "disparity-stereo",
	                              "views-mono", "views-from-depth", "splats-mono", "splats-stereo"};
	return t < sizeof(names) / sizeof(names[0]) ? names[t] : "?";
}

static const char *
state_name(uint32_t s)
{
	static const char *names[] = {"UNKNOWN", "READY",    "WARMING",   "ABSENT",      "INCOMPATIBLE", "UNLICENSED",
	                              "FAILED",  "DISABLED", "INSTALLED", "DOWNLOADING", "EVICTED"};
	return s < sizeof(names) / sizeof(names[0]) ? names[s] : "?";
}

static void
validate_cap(const struct xrt_conv_capability *c, uint32_t index)
{
	check(c->cap_id != 0, "cap %u: cap_id is non-zero (%u)", index, c->cap_id);
	check(c->task == XRT_CONV_TASK_DEPTH_MONO || c->task == XRT_CONV_TASK_DEPTH_VIDEO,
	      "cap %u: a depth task (%u = %s); views are composed by the runtime", index, c->task, task_name(c->task));
	check(c->outputs == XRT_CONV_OUT_DEPTH, "cap %u: outputs DEPTH only (0x%x)", index, c->outputs);
	check(c->lanes != 0 && (c->lanes & ~(XRT_CONV_LANE_STREAM | XRT_CONV_LANE_JOB)) == 0,
	      "cap %u: lanes set and known (0x%x)", index, c->lanes);
	check(c->max_views == 0 && c->min_views == 0, "cap %u: no views", index);
	if (c->task == XRT_CONV_TASK_DEPTH_VIDEO) {
		check((c->flags & XRT_CONV_CAP_TEMPORAL) != 0, "cap %u: DEPTH_VIDEO is TEMPORAL", index);
		check(c->lanes == XRT_CONV_LANE_STREAM, "cap %u: DEPTH_VIDEO is stream-only", index);
	}
	check((c->flags & XRT_CONV_CAP_USES_MODEL_STORE) != 0, "cap %u: USES_MODEL_STORE", index);
	check(c->licence_class <= XRT_CONV_LICENCE_RESEARCH_ONLY, "cap %u: licence class known (%u)", index,
	      c->licence_class);
	check((c->licence_class == XRT_CONV_LICENCE_RESEARCH_ONLY) == (c->commercial_ok == 0) ||
	          c->licence_class == XRT_CONV_LICENCE_UNKNOWN,
	      "cap %u: commercial_ok (%u) agrees with the licence class (%u)", index, c->commercial_ok,
	      c->licence_class);
	check((c->import_memory_kinds & XRT_CONV_MEM_CPU) != 0, "cap %u: imports XRT_CONV_MEM_CPU", index);
	check(c->depth_kind != XRT_CONV_DEPTH_UNKNOWN, "cap %u: declares its depth kind (%u)", index, c->depth_kind);
	check(c->intrinsics_source == XRT_CONV_INTRINSICS_NONE, "cap %u: claims no intrinsics", index);
	info("cap %u: id %u, %s, model '%s' v%s, backend '%s', licence %s (class %u), commercial_ok %u, luid 0x%llx",
	     index, c->cap_id, task_name(c->task), c->model_id, c->model_version, c->backend, c->licence_spdx,
	     c->licence_class, c->commercial_ok, (unsigned long long)c->device_luid);
}


/*
 *
 * Readiness watcher: polls get_state ([C]) while warm ([S]) runs.
 *
 */

static struct
{
	struct xrt_conversion_plugin_iface *iface;
	struct xrt_conversion_plugin_instance *inst;
	uint32_t cap_id;
	volatile bool stop;
	uint32_t seen; //!< 1u << state
	float max_download_progress;
} g_watch;

static void
watch_main(void *arg)
{
	(void)arg;
	while (!g_watch.stop) {
		struct xrt_conv_state st;
		memset(&st, 0, sizeof(st));
		st.struct_size = (uint32_t)sizeof(st);
		if (g_watch.iface->get_state(g_watch.inst, g_watch.cap_id, &st) == XRT_SUCCESS && st.state < 32) {
			g_watch.seen |= 1u << st.state;
			if (st.state == XRT_CONV_STATE_DOWNLOADING && st.progress > g_watch.max_download_progress) {
				g_watch.max_download_progress = st.progress;
			}
		}
		oc_sleep_ms(5);
	}
}


/*
 *
 * Test image: a perspective checkerboard floor under a sky. Any depth model
 * sees the floor recede towards the horizon: near at the bottom.
 *
 */

static uint8_t *
make_scene(uint32_t w, uint32_t h, float phase)
{
	uint8_t *img = (uint8_t *)malloc((size_t)w * h * 4u);
	if (img == NULL) {
		return NULL;
	}
	const float horizon = (float)h * 0.35f;
	const float f = (float)w * 0.8f;
	for (uint32_t y = 0; y < h; y++) {
		for (uint32_t x = 0; x < w; x++) {
			float r, g, b;
			if ((float)y < horizon) {
				const float t = (float)y / horizon;
				r = 0.45f + 0.3f * t;
				g = 0.6f + 0.25f * t;
				b = 0.9f;
			} else {
				const float dy = (float)y - horizon + 0.5f;
				const float z = f / dy; // camera 1 m above the floor
				const float xw = ((float)x - (float)w * 0.5f) * z / f;
				const int c = ((int)floorf(xw + phase) + (int)floorf(z)) & 1;
				const float v = 0.25f + 0.5f * (float)c;
				const float fog = expf(-z / 40.0f);
				r = v * fog + 0.6f * (1.0f - fog);
				g = v * fog + 0.7f * (1.0f - fog);
				b = v * 0.9f * fog + 0.9f * (1.0f - fog);
			}
			uint8_t *px = img + ((size_t)y * w + x) * 4u;
			px[0] = (uint8_t)(r * 255.0f + 0.5f);
			px[1] = (uint8_t)(g * 255.0f + 0.5f);
			px[2] = (uint8_t)(b * 255.0f + 0.5f);
			px[3] = 255;
		}
	}
	return img;
}

static void
cpu_image(struct xrt_conv_image *img, void *ptr, uint32_t w, uint32_t h, uint32_t format, uint32_t bpp, uint64_t key)
{
	memset(img, 0, sizeof(*img));
	img->memory_kind = XRT_CONV_MEM_CPU;
	img->import_key = key;
	img->cpu_ptr = ptr;
	img->byte_size = (uint64_t)w * h * bpp;
	img->width = w;
	img->height = h;
	img->format = format;
	img->color_space = bpp == 4 && format != XRT_CONV_FMT_R32_FLOAT ? XRT_CONV_CS_SRGB : XRT_CONV_CS_DATA;
	img->row_stride = w * bpp;
	img->sync.kind = XRT_CONV_SYNC_NONE;
}

/*!
 * The plausibility test (statistics, never bytes: execution providers and
 * CPUs differ in the last bits). For a "larger = nearer" kind the floor makes
 * the per-row mean rise towards the bottom.
 */
static void
check_depth_map(const char *what, const float *d, uint32_t w, uint32_t h, const struct xrt_conv_depth_info *di)
{
	size_t bad = 0;
	float lo = INFINITY, hi = -INFINITY;
	double sum_top = 0, sum_bot = 0;
	size_t n_top = 0, n_bot = 0;
	double sy = 0, sr = 0, syy = 0, srr = 0, syr = 0;
	for (uint32_t y = 0; y < h; y++) {
		double row = 0;
		for (uint32_t x = 0; x < w; x++) {
			const float v = d[(size_t)y * w + x];
			if (!isfinite(v)) {
				bad++;
				continue;
			}
			lo = v < lo ? v : lo;
			hi = v > hi ? v : hi;
			row += v;
			if (y < h / 3) {
				sum_top += v;
				n_top++;
			} else if (y >= 2 * h / 3) {
				sum_bot += v;
				n_bot++;
			}
		}
		row /= (double)w;
		sy += y;
		sr += row;
		syy += (double)y * y;
		srr += row * row;
		syr += (double)y * row;
	}
	const double n = (double)h;
	const double cov = syr / n - (sy / n) * (sr / n);
	const double vy = syy / n - (sy / n) * (sy / n);
	const double vr = srr / n - (sr / n) * (sr / n);
	const double corr = vy > 0 && vr > 0 ? cov / sqrt(vy * vr) : 0.0;
	const double top = n_top > 0 ? sum_top / (double)n_top : 0.0;
	const double bot = n_bot > 0 ? sum_bot / (double)n_bot : 0.0;
	const bool nearer_is_larger = di->kind == XRT_CONV_DEPTH_AFFINE_DISPARITY ||
	                              di->kind == XRT_CONV_DEPTH_AFFINE_INVERSE ||
	                              di->kind == XRT_CONV_DEPTH_METRIC_INVERSE;
	info("%s: %ux%u range [%.4f, %.4f] p05 %.4f p95 %.4f; mean top %.4f, bottom %.4f; corr(row, y) %.3f", what, w,
	     h, (double)lo, (double)hi, (double)di->range_p05, (double)di->range_p95, top, bot, corr);
	check(bad == 0, "%s: every texel written and finite (%zu bad)", what, bad);
	check(hi > lo, "%s: the map is not constant", what);
	check(fabsf(lo - di->range_min) <= 1e-3f * (fabsf(hi) + 1.0f) &&
	          fabsf(hi - di->range_max) <= 1e-3f * (fabsf(hi) + 1.0f),
	      "%s: depth_info range matches the written values", what);
	check(di->range_p05 >= di->range_min && di->range_p95 <= di->range_max && di->range_p05 <= di->range_p95,
	      "%s: percentiles inside the range", what);
	check(nearer_is_larger ? corr > 0.6 : corr < -0.6, "%s: depth follows the floor (corr %.3f, kind %u)", what,
	      corr, di->kind);
	check(nearer_is_larger ? bot > top + 0.1 * (hi - lo) : bot + 0.1 * (hi - lo) < top,
	      "%s: the bottom third reads nearer than the top third", what);
}


/*
 *
 * Lanes.
 *
 */

static void
run_stream(struct xrt_conversion_plugin_iface *iface,
           struct xrt_conversion_plugin_instance *inst,
           const struct xrt_conv_capability *cap,
           uint32_t frames)
{
	const uint32_t W = 320, H = 180;
	struct xrt_conv_stream_info si;
	memset(&si, 0, sizeof(si));
	si.struct_size = (uint32_t)sizeof(si);
	si.input_kind = XRT_CONV_INPUT_MONO;
	si.outputs = XRT_CONV_OUT_DEPTH;
	si.content_hint = cap->task == XRT_CONV_TASK_DEPTH_VIDEO ? XRT_CONV_CONTENT_VIDEO : XRT_CONV_CONTENT_PHOTO;
	si.max_input_width = W;
	si.max_input_height = H;
	uint64_t stream = 0;
	xrt_result_t r = iface->stream_create(inst, cap->cap_id, &si, &stream);
	check(r == XRT_SUCCESS, "stream_create on cap %u (result %d)", cap->cap_id, (int)r);
	if (r != XRT_SUCCESS) {
		return;
	}
	struct xrt_conv_output_desc od;
	memset(&od, 0, sizeof(od));
	od.struct_size = (uint32_t)sizeof(od);
	r = iface->stream_get_output_desc(inst, stream, W, H, &od);
	check(r == XRT_SUCCESS && od.depth_width > 0 && od.depth_height > 0 &&
	          od.depth_format == XRT_CONV_FMT_R32_FLOAT && od.depth_width <= W && od.depth_height <= H,
	      "stream_get_output_desc: depth %ux%u format %u (never above the input)", od.depth_width, od.depth_height,
	      od.depth_format);
	if (r != XRT_SUCCESS || od.depth_width == 0 || od.depth_height == 0) {
		iface->stream_destroy(inst, stream);
		return;
	}
	const size_t n = (size_t)od.depth_width * od.depth_height;
	float *dst = (float *)malloc(n * sizeof(float));
	for (uint32_t f = 0; f < frames && dst != NULL; f++) {
		uint8_t *src = make_scene(W, H, 0.05f * (float)f);
		if (src == NULL) {
			break;
		}
		for (size_t i = 0; i < n; i++) {
			dst[i] = NAN; // detect unwritten texels
		}
		struct xrt_conv_frame_in in;
		memset(&in, 0, sizeof(in));
		in.struct_size = (uint32_t)sizeof(in);
		in.image_count = 1;
		in.image_role[0] = XRT_CONV_IMG_COLOR;
		cpu_image(&in.images[0], src, W, H, XRT_CONV_FMT_RGBA8_SRGB, 4, 1);
		struct xrt_conv_params params;
		memset(&params, 0, sizeof(params));
		params.struct_size = (uint32_t)sizeof(params);
		// A cut on the first frame and once mid-stream: temporal state resets.
		params.frame_flags = (f == 0 || f == frames / 2 + 1) ? XRT_CONV_FRAME_RESET : 0u;
		params.source_time_ns = 1000000ull * (f + 1);
		params.frame_id = f + 1;
		params.strength = 1.0f;
		params.content_hint = si.content_hint;
		params.input_scale = 1.0f;
		struct xrt_conv_frame_out out;
		memset(&out, 0, sizeof(out));
		out.struct_size = (uint32_t)sizeof(out);
		cpu_image(&out.depth, dst, od.depth_width, od.depth_height, XRT_CONV_FMT_R32_FLOAT, 4, 2);
		const uint64_t t0 = oc_now_ms();
		r = iface->stream_convert(inst, stream, &in, &params, &out);
		const uint64_t t1 = oc_now_ms();
		check(r == XRT_SUCCESS, "stream_convert frame %u%s (result %d, %llu ms)", f,
		      params.frame_flags != 0 ? " (reset)" : "", (int)r, (unsigned long long)(t1 - t0));
		if (r == XRT_SUCCESS) {
			const struct xrt_conv_depth_info *di = &out.depth_info;
			check((out.written & XRT_CONV_OUT_DEPTH) != 0, "written has DEPTH (0x%x)", out.written);
			check(di->kind == cap->depth_kind, "depth_info.kind %u == capability depth_kind", di->kind);
			check(di->source_time_ns == params.source_time_ns, "depth_info echoes source_time_ns");
			check(di->source_width == W && di->source_height == H,
			      "depth_info source extent is the input's");
			check(di->temporal == (cap->task == XRT_CONV_TASK_DEPTH_VIDEO ? 1u : 0u),
			      "depth_info.temporal matches the task");
			check(di->intrinsics_source == XRT_CONV_INTRINSICS_NONE && di->fx <= 0.0f && di->fy <= 0.0f &&
			          di->cx <= 0.0f && di->cy <= 0.0f,
			      "no intrinsics invented");
			check(isnan(di->invalid_sample), "no invalid-sample sentinel claimed");
			char what[64];
			snprintf(what, sizeof(what), "%s frame %u", task_name(cap->task), f);
			check_depth_map(what, dst, od.depth_width, od.depth_height, di);
		}
		free(src);
	}
	free(dst);
	iface->stream_destroy(inst, stream);
}

struct sink_ctx
{
	uint32_t format;
	uint64_t bytes;
	bool begun;
	uint8_t *data;
	size_t cap;
};

static bool
sink_begin(void *ctx, uint32_t blob_format, uint64_t size_hint)
{
	struct sink_ctx *s = (struct sink_ctx *)ctx;
	s->format = blob_format;
	s->begun = true;
	s->data = size_hint > 0 && size_hint < (64u << 20) ? (uint8_t *)malloc((size_t)size_hint) : NULL;
	s->cap = s->data != NULL ? (size_t)size_hint : 0;
	return true;
}

static bool
sink_write(void *ctx, const void *bytes, uint64_t count)
{
	struct sink_ctx *s = (struct sink_ctx *)ctx;
	if (s->data != NULL && s->bytes + count <= s->cap) {
		memcpy(s->data + s->bytes, bytes, (size_t)count);
	}
	s->bytes += count;
	return true;
}

static void
job_progress(void *ctx, float fraction, const char *stage)
{
	(void)ctx;
	info("job progress %.2f (%s)", fraction, stage != NULL ? stage : "");
}

static void
run_job(struct xrt_conversion_plugin_iface *iface,
        struct xrt_conversion_plugin_instance *inst,
        const struct xrt_conv_capability *cap)
{
	const uint32_t W = 400, H = 300; // 4:3, another aspect than the stream
	uint8_t *src = make_scene(W, H, 0.0f);
	if (src == NULL) {
		check(false, "allocate job image");
		return;
	}
	struct xrt_conv_job_in in;
	memset(&in, 0, sizeof(in));
	in.struct_size = (uint32_t)sizeof(in);
	in.input_kind = XRT_CONV_INPUT_MONO;
	in.image_count = 1;
	in.image_role[0] = XRT_CONV_IMG_COLOR;
	cpu_image(&in.images[0], src, W, H, XRT_CONV_FMT_RGBA8_UNORM, 4, 3);
	// BGRA as the runtime's D3D11 path delivers it: swap in place.
	for (size_t i = 0; i < (size_t)W * H; i++) {
		const uint8_t t = src[i * 4];
		src[i * 4] = src[i * 4 + 2];
		src[i * 4 + 2] = t;
	}
	in.images[0].format = XRT_CONV_FMT_BGRA8_UNORM;
	struct xrt_conv_params params;
	memset(&params, 0, sizeof(params));
	params.struct_size = (uint32_t)sizeof(params);
	params.content_hint = XRT_CONV_CONTENT_PHOTO;
	params.strength = 1.0f;
	params.source_time_ns = 42;
	struct sink_ctx sc;
	memset(&sc, 0, sizeof(sc));
	struct xrt_conv_blob_sink sink;
	memset(&sink, 0, sizeof(sink));
	sink.struct_size = (uint32_t)sizeof(sink);
	sink.role = XRT_CONV_OUT_DEPTH;
	sink.ctx = &sc;
	sink.begin = sink_begin;
	sink.write = sink_write;
	struct xrt_conv_job_control ctl;
	memset(&ctl, 0, sizeof(ctl));
	ctl.struct_size = (uint32_t)sizeof(ctl);
	ctl.sink_count = 1;
	ctl.sinks = &sink;
	ctl.progress = job_progress;
	ctl.is_cancelled = never_cancelled;
	struct xrt_conv_job_out out;
	memset(&out, 0, sizeof(out));
	out.struct_size = (uint32_t)sizeof(out);
	xrt_result_t r = iface->job_execute(inst, cap->cap_id, &in, &params, &ctl, &out);
	check(r == XRT_SUCCESS, "job_execute on cap %u (result %d)", cap->cap_id, (int)r);
	if (r == XRT_SUCCESS) {
		check(sc.begun && sc.format == XRT_CONV_BLOB_RAW_IMAGE, "job wrote a RAW_IMAGE blob (format %u)",
		      sc.format);
		check((out.written & XRT_CONV_OUT_DEPTH) != 0, "job written has DEPTH");
		check(out.depth_info.source_time_ns == 42, "job depth_info echoes source_time_ns");
		// The blob keeps the input aspect at the model's long edge.
		const uint32_t bw = cap->preferred_input_edge < W ? cap->preferred_input_edge : W;
		const uint32_t bh = (uint32_t)floor((double)H * bw / W + 0.5);
		check(sc.bytes == (uint64_t)bw * bh * 4u, "job blob is %ux%u R32F (%llu bytes)", bw, bh,
		      (unsigned long long)sc.bytes);
		if (sc.data != NULL && sc.bytes == (uint64_t)bw * bh * 4u) {
			check_depth_map("depth-mono job", (const float *)sc.data, bw, bh, &out.depth_info);
		}
	}
	free(sc.data);
	free(src);
}


/*
 *
 * main.
 *
 */

static void
usage(void)
{
	fprintf(
	    stderr,
	    "usage: open-conversion-fakehost <provider> [--set K=V]... [--pack FILE] [--blob-base URL]\n"
	    "         [--store DIR] [--clean-store] [--no-download] [--chunk-ms N] [--caps video,mono] [--frames N]\n"
	    "         [--no-warm] [--expect ready|absent|version|policy]\n"
	    "         [--expect-licence permissive|commercial|research] [--expect-download]\n"
	    "         [--warm-timeout-ms N]\n");
}

static bool
add_setting(const char *key, const char *value)
{
	if (g_setting_count >= MAX_SETTINGS) {
		return false;
	}
	snprintf(g_settings[g_setting_count].key, sizeof(g_settings[0].key), "%s", key);
	snprintf(g_settings[g_setting_count].value, sizeof(g_settings[0].value), "%s", value);
	g_setting_count++;
	return true;
}

static bool
cap_selected(const char *caps, uint32_t task)
{
	if (caps == NULL) {
		return true;
	}
	return (task == XRT_CONV_TASK_DEPTH_VIDEO && strstr(caps, "video") != NULL) ||
	       (task == XRT_CONV_TASK_DEPTH_MONO && strstr(caps, "mono") != NULL);
}

int
main(int argc, char **argv)
{
	const char *path = NULL, *expect = "ready", *expect_licence = NULL, *pack = NULL, *blob_base = NULL;
	const char *store = NULL, *caps_sel = NULL;
	bool no_warm = false, expect_download = false;
	uint32_t warm_timeout_ms = 120000, frames = 3;
	oc_mutex_init(&g_store.lock);
	for (int i = 1; i < argc; i++) {
		const char *a = argv[i];
		const bool more = i + 1 < argc;
		if (strcmp(a, "--set") == 0 && more) {
			const char *kv = argv[++i];
			const char *eq = strchr(kv, '=');
			char key[64];
			if (eq == NULL || (size_t)(eq - kv) >= sizeof(key)) {
				usage();
				return 2;
			}
			snprintf(key, sizeof(key), "%.*s", (int)(eq - kv), kv);
			if (!add_setting(key, eq + 1)) {
				usage();
				return 2;
			}
		} else if (strcmp(a, "--pack") == 0 && more) {
			pack = argv[++i];
		} else if (strcmp(a, "--blob-base") == 0 && more) {
			blob_base = argv[++i];
		} else if (strcmp(a, "--store") == 0 && more) {
			store = argv[++i];
		} else if (strcmp(a, "--clean-store") == 0) {
			g_store.clean = true;
		} else if (strcmp(a, "--no-download") == 0) {
			g_store.no_download = true;
		} else if (strcmp(a, "--chunk-ms") == 0 && more) {
			g_store.chunk_ms = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(a, "--caps") == 0 && more) {
			caps_sel = argv[++i];
		} else if (strcmp(a, "--frames") == 0 && more) {
			frames = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(a, "--no-warm") == 0) {
			no_warm = true;
		} else if (strcmp(a, "--expect") == 0 && more) {
			expect = argv[++i];
		} else if (strcmp(a, "--expect-licence") == 0 && more) {
			expect_licence = argv[++i];
		} else if (strcmp(a, "--expect-download") == 0) {
			expect_download = true;
		} else if (strcmp(a, "--warm-timeout-ms") == 0 && more) {
			warm_timeout_ms = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (path == NULL && a[0] != '-') {
			path = a;
		} else {
			usage();
			return 2;
		}
	}
	if (path == NULL) {
		usage();
		return 2;
	}
	if (pack != NULL) {
		if (store == NULL) {
			fprintf(stderr, "--pack needs --store\n");
			return 2;
		}
		if (!store_setup(pack, blob_base, store)) {
			return 2;
		}
		add_setting("PACK", pack);
		info("store: %s, %u blobs, base %s%s", g_store.root, g_store.blob_count, g_store.blob_base,
		     g_store.no_download ? ", downloads forbidden" : "");
	}

	char full[1100], err[256] = "";
#ifdef _WIN32
	{
		wchar_t w[1024], wf[1024];
		if (MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 1024) == 0 ||
		    GetFullPathNameW(w, 1024, wf, NULL) == 0 ||
		    WideCharToMultiByte(CP_UTF8, 0, wf, -1, full, (int)sizeof(full), NULL, NULL) == 0) {
			fprintf(stderr, "bad path %s\n", path);
			return 2;
		}
	}
#else
	if (realpath(path, full) == NULL) {
		fprintf(stderr, "bad path %s\n", path);
		return 2;
	}
#endif
	void *module = oc_lib_open(full, err, sizeof(err));
	if (module == NULL) {
		fprintf(stderr, "cannot load %s: %s\n", full, err);
		return 2;
	}
	xrt_conversion_plugin_negotiate_fn_t negotiate =
	    (xrt_conversion_plugin_negotiate_fn_t)oc_lib_sym(module, XRT_CONVERSION_PLUGIN_ENTRYPOINT_NAME);
	check(negotiate != NULL, "exports %s", XRT_CONVERSION_PLUGIN_ENTRYPOINT_NAME);
	if (negotiate == NULL) {
		return 1;
	}
	struct xrt_conversion_plugin_iface *iface = NULL;
	uint32_t plugin_api = 0;
	{
		// A different ABI major must be refused before anything else happens.
		struct xrt_conversion_plugin_iface *bogus = NULL;
		uint32_t api = 0;
		check(negotiate(XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT + 1, &g_host, &bogus, &api) != XRT_SUCCESS &&
		          bogus == NULL,
		      "negotiate refuses ABI major %u", XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT + 1);
	}
	xrt_result_t r = negotiate(XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT, &g_host, &iface, &plugin_api);
	check(r == XRT_SUCCESS && iface != NULL, "negotiate (result %d)", (int)r);
	check(plugin_api == XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT, "ABI major %u == %u", plugin_api,
	      XRT_CONVERSION_PLUGIN_API_VERSION_CURRENT);
	if (r != XRT_SUCCESS || iface == NULL) {
		return 1;
	}
	check(iface->struct_size >= sizeof(struct xrt_conversion_plugin_iface), "iface struct_size %u covers v1 (%u)",
	      iface->struct_size, (unsigned)sizeof(struct xrt_conversion_plugin_iface));
	check(iface->id != NULL && strcmp(iface->id, "open-conversion") == 0, "iface id is 'open-conversion'");
	check(iface->get_provider_status != NULL && iface->probe != NULL && iface->describe_cap != NULL &&
	          iface->get_generation != NULL && iface->get_state != NULL && iface->warm != NULL &&
	          iface->evict != NULL && iface->stream_create != NULL && iface->stream_get_output_desc != NULL &&
	          iface->stream_convert != NULL && iface->stream_destroy != NULL && iface->job_execute != NULL &&
	          iface->destroy != NULL,
	      "every v1 slot is set");
	if (iface->id == NULL || iface->probe == NULL) {
		return 1;
	}
	info("provider '%s' (%s), vendor %s, version %s", iface->id, iface->display_name,
	     iface->vendor != NULL ? iface->vendor : "-", iface->version != NULL ? iface->version : "-");
	size_t n = (size_t)snprintf(g_env_prefix, sizeof(g_env_prefix), "DXR_CONV_");
	for (const char *c = iface->id; *c != '\0' && n + 2 < sizeof(g_env_prefix); c++) {
		g_env_prefix[n++] = *c == '-' ? '_' : (char)toupper((unsigned char)*c);
	}
	g_env_prefix[n++] = '_';
	g_env_prefix[n] = '\0';

	struct xrt_conv_provider_status ps;
	memset(&ps, 0, sizeof(ps));
	ps.struct_size = (uint32_t)sizeof(ps);
	check(iface->get_provider_status(&ps), "get_provider_status before probe");
	info("provider status %u: %s", ps.state, ps.hint);

	struct xrt_conversion_plugin_instance *inst = NULL;
	r = iface->probe(&inst);
	check(r == XRT_SUCCESS, "probe (result %d)", (int)r);
	if (r != XRT_SUCCESS) {
		return 1;
	}
	memset(&ps, 0, sizeof(ps));
	ps.struct_size = (uint32_t)sizeof(ps);
	check(iface->get_provider_status(&ps), "get_provider_status after probe");
	info("provider status %u: %s", ps.state, ps.hint);

	uint32_t count = 0;
	r = iface->describe_cap(inst, 0, NULL, &count);
	check(r == XRT_SUCCESS && count == 2, "describe_cap count = %u (DEPTH_VIDEO + DEPTH_MONO)", count);
	if (r != XRT_SUCCESS || count == 0 || count > XRT_CONV_MAX_CAPABILITIES) {
		return 1;
	}
	const uint64_t gen0 = iface->get_generation(inst);
	struct xrt_conv_capability caps[XRT_CONV_MAX_CAPABILITIES];
	for (uint32_t i = 0; i < count; i++) {
		memset(&caps[i], 0, sizeof(caps[i]));
		caps[i].struct_size = (uint32_t)sizeof(caps[i]);
		r = iface->describe_cap(inst, i, &caps[i], NULL);
		check(r == XRT_SUCCESS, "describe_cap(%u)", i);
		validate_cap(&caps[i], i);
		for (uint32_t j = 0; j < i; j++) {
			check(caps[j].cap_id != caps[i].cap_id, "cap_id %u is unique", caps[i].cap_id);
		}
	}
	{
		const size_t trunc = offsetof(struct xrt_conv_capability, licence_class);
		unsigned char buf[sizeof(struct xrt_conv_capability) + 64];
		memset(buf, 0xCD, sizeof(buf));
		struct xrt_conv_capability *c = (struct xrt_conv_capability *)(void *)buf;
		memset(c, 0, trunc);
		c->struct_size = (uint32_t)trunc;
		(void)iface->describe_cap(inst, 0, c, NULL);
		bool clean = true;
		for (size_t k = trunc; k < sizeof(buf); k++) {
			clean = clean && buf[k] == 0xCD;
		}
		check(clean, "describe_cap writes nothing past a truncated struct_size (%zu bytes)", trunc);
		struct xrt_conv_capability past;
		memset(&past, 0, sizeof(past));
		past.struct_size = (uint32_t)sizeof(past);
		check(iface->describe_cap(inst, count, &past, NULL) != XRT_SUCCESS && past.cap_id == 0,
		      "describe_cap(index == count) is refused and writes nothing");
	}
	check(iface->get_generation(inst) == gen0, "generation stable across the walk");

	if (expect_licence != NULL) {
		const uint32_t want = strcmp(expect_licence, "research") == 0     ? XRT_CONV_LICENCE_RESEARCH_ONLY
		                      : strcmp(expect_licence, "commercial") == 0 ? XRT_CONV_LICENCE_COMMERCIAL
		                                                                  : XRT_CONV_LICENCE_PERMISSIVE;
		for (uint32_t i = 0; i < count; i++) {
			if (!cap_selected(caps_sel, caps[i].task)) {
				continue;
			}
			// DirectML-backed caps are COMMERCIAL (proprietary runtime, commercial use
			// allowed); CPU ones PERMISSIVE. "permissive" accepts either.
			const bool ok = want == XRT_CONV_LICENCE_RESEARCH_ONLY
			                    ? caps[i].licence_class == want && caps[i].commercial_ok == 0
			                    : (caps[i].licence_class == XRT_CONV_LICENCE_PERMISSIVE ||
			                       caps[i].licence_class == XRT_CONV_LICENCE_COMMERCIAL) &&
			                          caps[i].commercial_ok == 1;
			check(ok, "cap %u licence: class %u, commercial_ok %u (expected %s)", caps[i].cap_id,
			      caps[i].licence_class, caps[i].commercial_ok, expect_licence);
		}
	}

	struct xrt_conv_state st;
	for (uint32_t i = 0; i < count; i++) {
		memset(&st, 0, sizeof(st));
		st.struct_size = (uint32_t)sizeof(st);
		r = iface->get_state(inst, caps[i].cap_id, &st);
		check(r == XRT_SUCCESS && st.state <= XRT_CONV_STATE_EVICTED,
		      "cap %u cold state: %s (reason %u, retryable %u) '%s'", caps[i].cap_id, state_name(st.state),
		      st.failure_reason, st.retryable, st.detail);
		if (strcmp(expect, "absent") == 0) {
			check(st.state == XRT_CONV_STATE_ABSENT, "cap %u: expected ABSENT", caps[i].cap_id);
		} else if (strcmp(expect, "version") == 0) {
			check(st.state == XRT_CONV_STATE_FAILED && st.failure_reason == XRT_CONV_FAIL_VERSION,
			      "cap %u: expected FAILED(VERSION), never FAILED(MODEL)", caps[i].cap_id);
		} else {
			check(st.state == XRT_CONV_STATE_INSTALLED || st.state == XRT_CONV_STATE_DOWNLOADING,
			      "cap %u: cold state is INSTALLED (or DOWNLOADING), never READY before warm",
			      caps[i].cap_id);
		}
	}
	if (strcmp(expect, "absent") == 0 || strcmp(expect, "version") == 0 || no_warm) {
		iface->destroy(inst);
		printf("%s: %d check(s) failed\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
		return g_failures == 0 ? 0 : 1;
	}

	// Tests that need a real blob are skipped (77) when the source is not reachable.
	if (g_store.enabled && !g_store.no_download) {
		for (uint32_t i = 0; i < count; i++) {
			if (!cap_selected(caps_sel, caps[i].task)) {
				continue;
			}
			char ref[192];
			snprintf(ref, sizeof(ref), "dxr-open-default/%s", caps[i].model_id);
			oc_lock(&g_store.lock);
			const struct blob *b = find_blob(ref);
			const bool reachable = b != NULL && (blob_present(b) || source_available(b));
			oc_unlock(&g_store.lock);
			if (!reachable) {
				printf("SKIPPED: model %s is not available from %s\n", caps[i].model_id,
				       g_store.blob_base);
				iface->destroy(inst);
				return SKIP_CODE;
			}
		}
	}

	for (uint32_t i = 0; i < count; i++) {
		if (!cap_selected(caps_sel, caps[i].task)) {
			continue;
		}
		g_watch.iface = iface;
		g_watch.inst = inst;
		g_watch.cap_id = caps[i].cap_id;
		g_watch.stop = false;
		g_watch.seen = 0;
		g_watch.max_download_progress = -1.0f;
		thread_t watcher;
		const bool watching = thread_start(&watcher, watch_main, NULL);
		const uint64_t t0 = oc_now_ms();
		r = iface->warm(inst, caps[i].cap_id, never_cancelled, NULL);
		// warm may return before READY (the contract lets a provider publish
		// progress and finish later): wait up to --warm-timeout-ms for a final state.
		for (;;) {
			memset(&st, 0, sizeof(st));
			st.struct_size = (uint32_t)sizeof(st);
			(void)iface->get_state(inst, caps[i].cap_id, &st);
			const bool waiting =
			    st.state == XRT_CONV_STATE_WARMING || st.state == XRT_CONV_STATE_DOWNLOADING;
			if (!waiting || oc_now_ms() - t0 > warm_timeout_ms) {
				break;
			}
			oc_sleep_ms(20);
		}
		const uint64_t t1 = oc_now_ms();
		g_watch.stop = true;
		if (watching) {
			thread_join(watcher);
		}
		memset(&st, 0, sizeof(st));
		st.struct_size = (uint32_t)sizeof(st);
		(void)iface->get_state(inst, caps[i].cap_id, &st);
		g_watch.seen |= 1u << st.state;
		info("cap %u warm: result %d in %llu ms, states seen 0x%x, final %s '%s' (version '%s')",
		     caps[i].cap_id, (int)r, (unsigned long long)(t1 - t0), g_watch.seen, state_name(st.state),
		     st.detail, st.version);
		if (strcmp(expect, "policy") == 0) {
			check(r != XRT_SUCCESS && st.state == XRT_CONV_STATE_ABSENT,
			      "cap %u: download forbidden by policy -> ABSENT (%s)", caps[i].cap_id,
			      state_name(st.state));
			continue;
		}
		check(r == XRT_SUCCESS && st.state == XRT_CONV_STATE_READY, "cap %u READY after warm (%s)",
		      caps[i].cap_id, state_name(st.state));
		if (expect_download) {
			check((g_watch.seen & (1u << XRT_CONV_STATE_DOWNLOADING)) != 0,
			      "cap %u reported DOWNLOADING while the host fetched (max progress %.2f)", caps[i].cap_id,
			      (double)g_watch.max_download_progress);
			check(g_watch.max_download_progress > 0.0f && g_watch.max_download_progress <= 1.0f,
			      "cap %u DOWNLOADING progress comes from the host", caps[i].cap_id);
			check((g_watch.seen & (1u << XRT_CONV_STATE_WARMING)) != 0, "cap %u went through WARMING",
			      caps[i].cap_id);
		}
		if (st.state != XRT_CONV_STATE_READY) {
			continue;
		}
		// Describe may have changed (DirectML -> CPU fallback): re-read.
		memset(&caps[i], 0, sizeof(caps[i]));
		caps[i].struct_size = (uint32_t)sizeof(caps[i]);
		(void)iface->describe_cap(inst, i, &caps[i], NULL);
		info("cap %u runs on '%s'", caps[i].cap_id, caps[i].backend);
		if ((caps[i].lanes & XRT_CONV_LANE_STREAM) != 0) {
			run_stream(iface, inst, &caps[i], caps[i].task == XRT_CONV_TASK_DEPTH_VIDEO ? frames : 1u);
		}
		if ((caps[i].lanes & XRT_CONV_LANE_JOB) != 0) {
			run_job(iface, inst, &caps[i]);
		}
		check(iface->evict(inst, caps[i].cap_id) == XRT_SUCCESS, "evict cap %u", caps[i].cap_id);
		memset(&st, 0, sizeof(st));
		st.struct_size = (uint32_t)sizeof(st);
		(void)iface->get_state(inst, caps[i].cap_id, &st);
		check(st.state == XRT_CONV_STATE_EVICTED, "cap %u EVICTED after evict (%s)", caps[i].cap_id,
		      state_name(st.state));
	}
	info("notify_changed calls: %d; model requests: %u; downloads completed: %u", g_notifications, g_store.requests,
	     g_store.downloads_done);
	check(g_notifications > 0, "the provider pushes state changes (notify_changed)");

	iface->destroy(inst);
	for (uint32_t i = 0; i < g_store.blob_count; i++) {
		if (g_store.blobs[i].thread_started) {
			thread_join(g_store.blobs[i].thread);
		}
	}
	printf("%s: %d check(s) failed\n", g_failures == 0 ? "OK" : "FAILED", g_failures);
	return g_failures == 0 ? 0 : 1;
}
