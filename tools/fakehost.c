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
 *       --verify-ms N          a stored blob reads INSTALLED until requested, then WARMING (verifying)
 *                              for N ms: the runtime's store re-hashes a stored blob once per process
 *       --expect-verify        the provider must report WARMING while the store verifies, then READY
 *       --caps video,mono      the capabilities to warm and convert (default: both)
 *       --frames N             frames per stream (default 3; a reset mid-stream)
 *       --no-warm              describe and readiness only
 *       --expect ready|absent|version|policy
 *       --expect-licence permissive|commercial|research
 *       --expect-download      the provider must report DOWNLOADING before READY
 *       --expect-model NAME    every selected capability describes model NAME (cold and after warm)
 *       --warm-timeout-ms N
 *       --scratch DIR          the per-provider scratch dir (get_path SCRATCH; Core ML model cache)
 *       --bench N              after the checked frames, time N more frames per stream (latency stats)
 *       --dump DIR             write every checked depth map as DIR/<task>-f<N>-<w>x<h>.f32 (raw R32F)
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
#include <time.h>
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
static unsigned __stdcall
thread_tramp(void *p)
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

#define FH_MAX_BLOBS (OC_PACK_MAX_MODELS * (1 + OC_PACK_MAX_DATA_FILES))

struct blob
{
	char sha256[65];
	char format[16];
	uint64_t size;
	char refs[4][192]; //!< "<pack>/<model>/<variant>" and "<pack>/<model>"
	/*! A model file's variant siblings (role "data"): a ref is READY only when they
	 *  are too, and a request fetches them all (the runtime store's rule). */
	struct blob *deps[OC_PACK_MAX_DATA_FILES];
	uint32_t dep_count;

	/* Download state, under g_store.lock. */
	uint32_t state;        //!< XRT_CONV_STATE_ABSENT / INSTALLED / WARMING / DOWNLOADING / READY / FAILED
	bool verified;         //!< --verify-ms: hashed by this "process" (or downloaded by it)
	uint64_t verify_until; //!< --verify-ms: a requested verify finishes at this oc_now_ms()
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
	uint32_t verify_ms; //!< --verify-ms
	char root[384]; //!< short enough that root/blobs/<sha256>.ok fits xrt_conv_model_location::path
	char blob_base[1024];
	char pack_id[64]; //!< the --pack's id (refs are "<pack>/<model>")
	struct blob blobs[FH_MAX_BLOBS];
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
	b->verified = ok; // hashed while it was written
	b->failure_reason = ok ? XRT_CONV_FAIL_NONE : reason;
	b->progress = ok ? 1.0f : -1.0f;
	g_store.downloads_done += ok ? 1u : 0u;
	oc_unlock(&g_store.lock);
	info("store: %s %s", b->sha256, ok ? "downloaded, verified, stamped" : "download FAILED");
}

//! The aggregation order of a ref's files (the runtime store's): the worst wins.
static int
store_rank(uint32_t state)
{
	switch (state) {
	case XRT_CONV_STATE_FAILED: return 5;
	case XRT_CONV_STATE_DOWNLOADING: return 4;
	case XRT_CONV_STATE_ABSENT: return 3;
	case XRT_CONV_STATE_WARMING: return 2;
	case XRT_CONV_STATE_INSTALLED: return 1;
	default: return 0; // READY
	}
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
	// The ref's state over every file of its variant:
	// FAILED > DOWNLOADING > ABSENT > WARMING > INSTALLED > READY.
	uint32_t state = XRT_CONV_STATE_READY, reason = 0;
	uint64_t total = 0;
	double done = 0.0;
	for (uint32_t i = 0; i <= b->dep_count; i++) {
		struct blob *f = i == 0 ? b : b->deps[i - 1];
		if (f->state != XRT_CONV_STATE_DOWNLOADING && f->state != XRT_CONV_STATE_FAILED) {
			if (!blob_present(f)) {
				f->state = XRT_CONV_STATE_ABSENT; // a wiped store reads ABSENT
				f->verified = false;
				f->verify_until = 0;
			} else if (g_store.verify_ms == 0 || f->verified) {
				f->state = XRT_CONV_STATE_READY;
			} else if (f->verify_until != 0 && oc_now_ms() >= f->verify_until) {
				f->verified = true; // the requested verify finished
				f->state = XRT_CONV_STATE_READY;
			} else {
				// Stored, not yet hashed by this process: INSTALLED until requested,
				// then WARMING (verifying; no progress, like the runtime's store).
				f->state = f->verify_until != 0 ? XRT_CONV_STATE_WARMING : XRT_CONV_STATE_INSTALLED;
			}
		}
		total += f->size;
		done += f->state == XRT_CONV_STATE_READY ? (double)f->size
		        : f->state == XRT_CONV_STATE_DOWNLOADING
		            ? (double)f->size * (f->progress > 0.0f ? f->progress : 0.0f)
		            : 0.0;
		if (store_rank(f->state) > store_rank(state)) {
			state = f->state;
			reason = f->state == XRT_CONV_STATE_FAILED ? f->failure_reason : 0;
		}
	}
	if (XRT_CONV_COVERS(out, struct xrt_conv_model_location, state)) {
		out->state = state;
	}
	if (XRT_CONV_COVERS(out, struct xrt_conv_model_location, progress)) {
		float pr = total > 0 ? (float)(done / (double)total) : -1.0f;
		out->progress = state == XRT_CONV_STATE_DOWNLOADING ? (pr > 0.99f ? 0.99f : pr) : -1.0f;
	}
	if (XRT_CONV_COVERS(out, struct xrt_conv_model_location, failure_reason)) {
		out->failure_reason = reason;
	}
	if (XRT_CONV_COVERS(out, struct xrt_conv_model_location, path)) {
		out->path[0] = '\0';
		if (state == XRT_CONV_STATE_READY) {
			blob_path(b, out->path, sizeof(out->path)); // the model file; its data files sit beside it
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
	bool missing = false;
	for (uint32_t i = 0; b != NULL && i <= b->dep_count; i++) {
		const struct blob *f = i == 0 ? b : b->deps[i - 1];
		missing |= !blob_present(f) && f->state != XRT_CONV_STATE_DOWNLOADING;
	}
	if (b == NULL || (g_store.no_download && missing)) {
		oc_unlock(&g_store.lock);
		return false; // unknown ref, or a download forbidden by policy
	}
	for (uint32_t i = 0; i <= b->dep_count; i++) {
		struct blob *f = i == 0 ? b : b->deps[i - 1];
		if (blob_present(f)) {
			if (g_store.verify_ms > 0 && !f->verified && f->verify_until == 0) {
				f->verify_until = oc_now_ms() + g_store.verify_ms; // a stored blob: hash it
			}
			continue;
		}
		if (f->state == XRT_CONV_STATE_DOWNLOADING) {
			continue; // one download shared by every request
		}
		if (f->thread_started) {
			thread_join(f->thread); // a finished, failed attempt
			f->thread_started = false;
		}
		f->state = XRT_CONV_STATE_DOWNLOADING;
		f->progress = 0.0f;
		f->failure_reason = 0;
		f->thread_started = thread_start(&f->thread, download_main, f);
		if (!f->thread_started) {
			f->state = XRT_CONV_STATE_FAILED;
			f->failure_reason = XRT_CONV_FAIL_RESOURCES;
		}
	}
	oc_unlock(&g_store.lock);
	return true;
}

//! The blob for @p f, added on first sight; NULL when the table is full.
static struct blob *
add_blob(const struct oc_pack_file *f)
{
	struct blob *b = find_blob(f->sha256);
	if (b == NULL && g_store.blob_count < FH_MAX_BLOBS) {
		b = &g_store.blobs[g_store.blob_count++];
		snprintf(b->sha256, sizeof(b->sha256), "%s", f->sha256);
		snprintf(b->format, sizeof(b->format), "%s", f->format);
		b->size = f->size;
		b->state = XRT_CONV_STATE_ABSENT;
	}
	return b;
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
	oc_copy_str(g_store.pack_id, sizeof(g_store.pack_id), pack->pack);
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
			struct blob *b = add_blob(&var->file);
			if (b == NULL) {
				continue;
			}
			for (uint32_t d = 0; d < var->data_count; d++) {
				struct blob *db = add_blob(&var->data[d]);
				bool linked = false;
				for (uint32_t k = 0; k < b->dep_count; k++) {
					linked = linked || b->deps[k] == db;
				}
				if (db != NULL && !linked && b->dep_count < OC_PACK_MAX_DATA_FILES) {
					b->deps[b->dep_count++] = db;
				}
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

//! Every file of @p b's variant is in the store, or reachable to fetch.
static bool
blob_set_reachable(const struct blob *b)
{
	for (uint32_t i = 0; i <= b->dep_count; i++) {
		const struct blob *f = i == 0 ? b : b->deps[i - 1];
		if (!blob_present(f) && !source_available(f)) {
			return false;
		}
	}
	return true;
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

static const char *g_scratch = NULL; //!< --scratch
static uint32_t g_bench = 0;         //!< --bench
static const char *g_dump = NULL;    //!< --dump

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
	if (kind == XRT_CONV_PATH_SCRATCH && g_scratch != NULL) {
		snprintf(buf, buf_size, "%s", g_scratch);
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
validate_splat_cap(const struct xrt_conv_capability *c, uint32_t index)
{
	check(c->lanes == XRT_CONV_LANE_JOB, "cap %u: SPLATS_MONO is job-only (0x%x)", index, c->lanes);
	check(c->outputs == XRT_CONV_OUT_SPLAT_BLOB, "cap %u: outputs SPLAT_BLOB only (0x%x)", index, c->outputs);
	check(c->input_kinds == XRT_CONV_INPUT_MONO, "cap %u: one photo in", index);
	check(c->splat_formats == (1u << XRT_CONV_BLOB_PLY_3DGS), "cap %u: declares PLY 3DGS (0x%x)", index,
	      c->splat_formats);
	check(c->max_streams == 0 && c->max_views == 0, "cap %u: no streams, no views", index);
	check((c->flags & XRT_CONV_CAP_USES_MODEL_STORE) != 0, "cap %u: USES_MODEL_STORE", index);
	check((c->import_memory_kinds & XRT_CONV_MEM_CPU) != 0, "cap %u: imports XRT_CONV_MEM_CPU", index);
	check(c->intrinsics_source == XRT_CONV_INTRINSICS_ASSUMED, "cap %u: intrinsics are labelled ASSUMED", index);
	check(c->model_id[0] == '\0' || c->licence_class == XRT_CONV_LICENCE_RESEARCH_ONLY ||
	          c->licence_class == XRT_CONV_LICENCE_PERMISSIVE || c->licence_class == XRT_CONV_LICENCE_COMMERCIAL,
	      "cap %u: licence class known (%u)", index, c->licence_class);
	info("cap %u: id %u, %s, model '%s' v%s, backend '%s', licence %s (class %u), commercial_ok %u", index,
	     c->cap_id, task_name(c->task), c->model_id, c->model_version, c->backend, c->licence_spdx,
	     c->licence_class, c->commercial_ok);
}

static void
validate_cap(const struct xrt_conv_capability *c, uint32_t index)
{
	check(c->cap_id != 0, "cap %u: cap_id is non-zero (%u)", index, c->cap_id);
	if (c->task == XRT_CONV_TASK_SPLATS_MONO) {
		validate_splat_cap(c, index);
		return;
	}
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
 * Timing (--bench) and dumps (--dump).
 *
 */

static double
now_us(void)
{
#ifdef _WIN32
	LARGE_INTEGER f, c;
	QueryPerformanceFrequency(&f);
	QueryPerformanceCounter(&c);
	return (double)c.QuadPart * 1e6 / (double)f.QuadPart;
#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
#endif
}

static int
cmp_double(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y ? 1 : 0;
}

static void
report_bench(const char *what, double *ms, uint32_t n)
{
	if (n == 0) {
		return;
	}
	double sum = 0;
	for (uint32_t i = 0; i < n; i++) {
		sum += ms[i];
	}
	qsort(ms, n, sizeof(double), cmp_double);
	info(
	    "BENCH %s: %u frames, min %.1f ms, median %.1f ms, p90 %.1f ms, mean %.1f ms, max %.1f ms "
	    "(%.1f fps at the median)",
	    what, n, ms[0], ms[n / 2], ms[(n * 9) / 10 < n ? (n * 9) / 10 : n - 1], sum / n, ms[n - 1],
	    1000.0 / ms[n / 2]);
}

static void
dump_map(const char *what, uint32_t f, const float *d, uint32_t w, uint32_t h)
{
	if (g_dump == NULL) {
		return;
	}
	char path[1200];
	snprintf(path, sizeof(path), "%s/%s-f%u-%ux%u.f32", g_dump, what, f, w, h);
	FILE *fp = fopen_utf8(path, "wb");
	if (fp == NULL || fwrite(d, sizeof(float), (size_t)w * h, fp) != (size_t)w * h) {
		info("cannot write %s", path);
	}
	if (fp != NULL) {
		fclose(fp);
	}
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
			dump_map(task_name(cap->task), f, dst, od.depth_width, od.depth_height);
		}
		free(src);
	}
	// --bench: N more frames on the same stream (a moving scene, no reset),
	// timed around stream_convert only.
	double *ms = g_bench > 0 ? (double *)calloc(g_bench, sizeof(double)) : NULL;
	uint32_t timed = 0;
	for (uint32_t f = 0; f < g_bench && ms != NULL && dst != NULL; f++) {
		uint8_t *src = make_scene(W, H, 0.05f * (float)(frames + f));
		if (src == NULL) {
			break;
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
		params.source_time_ns = 1000000ull * (frames + f + 1);
		params.frame_id = frames + f + 1;
		params.strength = 1.0f;
		params.content_hint = si.content_hint;
		params.input_scale = 1.0f;
		struct xrt_conv_frame_out out;
		memset(&out, 0, sizeof(out));
		out.struct_size = (uint32_t)sizeof(out);
		cpu_image(&out.depth, dst, od.depth_width, od.depth_height, XRT_CONV_FMT_R32_FLOAT, 4, 2);
		const double t0 = now_us();
		r = iface->stream_convert(inst, stream, &in, &params, &out);
		const double t1 = now_us();
		free(src);
		if (r != XRT_SUCCESS) {
			check(false, "bench frame %u (result %d)", f, (int)r);
			break;
		}
		ms[timed++] = (t1 - t0) / 1000.0;
	}
	if (timed > 0) {
		char what[96];
		snprintf(what, sizeof(what), "%s '%s' on '%s'", task_name(cap->task), cap->model_id, cap->backend);
		report_bench(what, ms, timed);
	}
	free(ms);
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
			dump_map("depth-mono-job", 0, (const float *)sc.data, bw, bh);
		}
	}
	free(sc.data);
	free(src);
}


/*
 *
 * The splat lane: SPLATS_MONO jobs, the PLY they write and the facts.
 *
 */

static const char *g_splat_check = NULL; //!< --splat-check: expected values (tests/make_tiny_splat_model.py)
static const char *g_splat_image = NULL; //!< --splat-image: a binary PPM (P6) photo
static float g_splat_focal = 0.0f;       //!< --splat-focal: a GIVEN focal for the first job
static const char *g_splat_out = NULL;   //!< --splat-out: write the first job's PLY here

struct mem_sink
{
	uint32_t format;
	bool begun;
	uint8_t *data;
	size_t size, cap;
	bool failed;
};

static bool
mem_begin(void *ctx, uint32_t blob_format, uint64_t size_hint)
{
	struct mem_sink *s = (struct mem_sink *)ctx;
	s->format = blob_format;
	s->begun = true;
	s->size = 0;
	if (size_hint > 0 && size_hint > s->cap) {
		uint8_t *n = (uint8_t *)realloc(s->data, (size_t)size_hint);
		if (n != NULL) {
			s->data = n;
			s->cap = (size_t)size_hint;
		}
	}
	return true;
}

static bool
mem_write(void *ctx, const void *bytes, uint64_t count)
{
	struct mem_sink *s = (struct mem_sink *)ctx;
	if (s->size + count > s->cap) {
		size_t want = s->cap * 2 > s->size + count ? s->cap * 2 : (size_t)(s->size + count);
		uint8_t *n = (uint8_t *)realloc(s->data, want);
		if (n == NULL) {
			s->failed = true;
			return false;
		}
		s->data = n;
		s->cap = want;
	}
	memcpy(s->data + s->size, bytes, (size_t)count);
	s->size += (size_t)count;
	return true;
}

//! A binary PPM (P6, maxval 255) as RGBA8.
static uint8_t *
read_ppm(const char *path, uint32_t *w, uint32_t *h)
{
	FILE *f = fopen_utf8(path, "rb");
	if (f == NULL) {
		return NULL;
	}
	unsigned W = 0, H = 0, maxv = 0;
	char magic[3] = {0};
	if (fscanf(f, "%2s %u %u %u", magic, &W, &H, &maxv) != 4 || strcmp(magic, "P6") != 0 || maxv != 255 || W == 0 ||
	    H == 0 || W > 16384 || H > 16384) {
		fclose(f);
		return NULL;
	}
	(void)fgetc(f); // the one whitespace after maxval
	uint8_t *rgb = (uint8_t *)malloc((size_t)W * H * 3u);
	uint8_t *rgba = (uint8_t *)malloc((size_t)W * H * 4u);
	const bool ok = rgb != NULL && rgba != NULL && fread(rgb, 1, (size_t)W * H * 3u, f) == (size_t)W * H * 3u;
	fclose(f);
	if (!ok) {
		free(rgb);
		free(rgba);
		return NULL;
	}
	for (size_t i = 0; i < (size_t)W * H; i++) {
		rgba[i * 4 + 0] = rgb[i * 3 + 0];
		rgba[i * 4 + 1] = rgb[i * 3 + 1];
		rgba[i * 4 + 2] = rgb[i * 3 + 2];
		rgba[i * 4 + 3] = 255;
	}
	free(rgb);
	*w = W;
	*h = H;
	return rgba;
}

//! Parse a 3DGS PLY header: vertex count, property count, body offset.
static bool
ply_header(const uint8_t *d, size_t n, uint64_t *vertices, uint32_t *props, size_t *body)
{
	const char *end = NULL;
	for (size_t i = 0; i + 11 <= n && i < 4096; i++) {
		if (memcmp(d + i, "end_header\n", 11) == 0) {
			end = (const char *)d + i + 11;
			break;
		}
	}
	if (end == NULL || n < 10 || memcmp(d, "ply\nformat binary_little_endian 1.0\n", 36) != 0) {
		return false;
	}
	const char *v = strstr((const char *)d, "element vertex ");
	if (v == NULL || v > end) {
		return false;
	}
	*vertices = strtoull(v + 15, NULL, 10);
	uint32_t pc = 0;
	for (const char *p = (const char *)d; p != NULL && p < end; p = strstr(p + 1, "property float ")) {
		if (strncmp(p, "property float ", 15) == 0) {
			pc++;
		}
	}
	*props = pc;
	*body = (size_t)(end - (const char *)d);
	return true;
}

static void
quat_rot(const float *q, double R[3][3])
{
	double w = q[0], x = q[1], y = q[2], z = q[3];
	const double n = sqrt(w * w + x * x + y * y + z * z);
	w /= n;
	x /= n;
	y /= n;
	z /= n;
	R[0][0] = 1 - 2 * (y * y + z * z);
	R[0][1] = 2 * (x * y - w * z);
	R[0][2] = 2 * (x * z + w * y);
	R[1][0] = 2 * (x * y + w * z);
	R[1][1] = 1 - 2 * (x * x + z * z);
	R[1][2] = 2 * (y * z - w * x);
	R[2][0] = 2 * (x * z - w * y);
	R[2][1] = 2 * (y * z + w * x);
	R[2][2] = 1 - 2 * (x * x + y * y);
}

//! The covariance a PLY row encodes (xx xy xz yy yz zz).
static void
row_cov(const float *row, double cov[6])
{
	double R[3][3];
	quat_rot(row + 10, R);
	const double s2[3] = {exp(2.0 * row[7]), exp(2.0 * row[8]), exp(2.0 * row[9])};
	const int idx[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
	for (int k = 0; k < 6; k++) {
		const int a = idx[k][0], b = idx[k][1];
		cov[k] = R[a][0] * s2[0] * R[b][0] + R[a][1] * s2[1] * R[b][1] + R[a][2] * s2[2] * R[b][2];
	}
}

static double
rel_err(double got, double want)
{
	const double d = fabs(got - want), m = fabs(want);
	return m > 1e-6 ? d / m : d;
}

//! Compare the PLY against expect.json (written by tests/make_tiny_splat_model.py).
static void
check_expectations(const float *rows, uint64_t count, const struct oc_json *ex)
{
	check((double)count == oc_json_num(ex, "count", -1.0), "PLY holds %llu Gaussians (expected %.0f)",
	      (unsigned long long)count, oc_json_num(ex, "count", -1.0));
	const double tol = oc_json_num(ex, "tol", 1e-4);
	const struct oc_json *samples = oc_json_get(ex, "samples");
	uint32_t checked = 0, bad = 0;
	for (const struct oc_json *s = samples != NULL ? samples->child : NULL; s != NULL; s = s->next) {
		const uint64_t i = (uint64_t)oc_json_num(s, "i", -1.0);
		if (i >= count) {
			bad++;
			continue;
		}
		const float *row = rows + i * 14u;
		double worst = 0.0;
		for (size_t k = 0; k < 3; k++) {
			const struct oc_json *xyz = oc_json_at(oc_json_get(s, "xyz"), k);
			const struct oc_json *dc = oc_json_at(oc_json_get(s, "f_dc"), k);
			worst = fmax(worst, xyz != NULL ? rel_err(row[k], xyz->number) : 1.0);
			worst = fmax(worst, dc != NULL ? rel_err(row[3 + k], dc->number) : 1.0);
		}
		worst = fmax(worst, rel_err(row[6], oc_json_num(s, "opacity", 1e9)));
		double cov[6];
		row_cov(row, cov);
		for (size_t k = 0; k < 6; k++) {
			const struct oc_json *c = oc_json_at(oc_json_get(s, "cov"), k);
			// Off-diagonal terms can be ~0: compare against the diagonal's scale.
			const double scale = fmax(fabs(cov[0]), fmax(fabs(cov[3]), fabs(cov[5])));
			worst = fmax(worst, c != NULL ? fabs(cov[k] - c->number) / scale : 1.0);
		}
		if (worst > tol) {
			bad++;
			info("splat %llu: worst relative error %.3g > %.3g", (unsigned long long)i, worst, tol);
		}
		checked++;
	}
	check(checked > 0 && bad == 0, "%u sampled Gaussians match the reference unprojection (%u off)", checked, bad);
}

static xrt_result_t
splat_job_once(struct xrt_conversion_plugin_iface *iface,
               struct xrt_conversion_plugin_instance *inst,
               const struct xrt_conv_capability *cap,
               uint8_t *rgba,
               uint32_t W,
               uint32_t H,
               float focal,
               uint32_t format,
               struct mem_sink *ms,
               struct xrt_conv_job_out *out,
               struct xrt_conv_splat_result *sr,
               double *ms_taken)
{
	struct xrt_conv_job_in in;
	memset(&in, 0, sizeof(in));
	in.struct_size = (uint32_t)sizeof(in);
	in.input_kind = XRT_CONV_INPUT_MONO;
	in.image_count = 1;
	in.image_role[0] = XRT_CONV_IMG_COLOR;
	cpu_image(&in.images[0], rgba, W, H, XRT_CONV_FMT_RGBA8_SRGB, 4, 7);
	in.requested_blob_format = format;
	in.job_flags = focal > 0.0f ? XRT_CONV_JOB_FOCAL_GIVEN : 0u;
	struct xrt_conv_params params;
	memset(&params, 0, sizeof(params));
	params.struct_size = (uint32_t)sizeof(params);
	params.content_hint = XRT_CONV_CONTENT_PHOTO;
	params.focal_px = focal;
	struct xrt_conv_blob_sink sink;
	memset(&sink, 0, sizeof(sink));
	sink.struct_size = (uint32_t)sizeof(sink);
	sink.role = XRT_CONV_OUT_SPLAT_BLOB;
	sink.ctx = ms;
	sink.begin = mem_begin;
	sink.write = mem_write;
	struct xrt_conv_job_control ctl;
	memset(&ctl, 0, sizeof(ctl));
	ctl.struct_size = (uint32_t)sizeof(ctl);
	ctl.sink_count = 1;
	ctl.sinks = &sink;
	ctl.progress = job_progress;
	ctl.is_cancelled = never_cancelled;
	memset(out, 0, sizeof(*out));
	out->struct_size = (uint32_t)sizeof(*out);
	memset(sr, 0, sizeof(*sr));
	sr->struct_size = (uint32_t)sizeof(*sr);
	out->splat_result = sr;
	ms->begun = false;
	ms->size = 0;
	const double t0 = now_us();
	const xrt_result_t r = iface->job_execute(inst, cap->cap_id, &in, &params, &ctl, out);
	*ms_taken = (now_us() - t0) / 1000.0;
	return r;
}

static void
run_splat_jobs(struct xrt_conversion_plugin_iface *iface,
               struct xrt_conversion_plugin_instance *inst,
               const struct xrt_conv_capability *cap)
{
	struct oc_json *ex = NULL;
	if (g_splat_check != NULL) {
		char err[160] = "";
		ex = oc_json_parse_file(g_splat_check, err, sizeof(err));
		check(ex != NULL, "read %s (%s)", g_splat_check, err);
		if (ex == NULL) {
			return;
		}
	}
	uint32_t W = ex != NULL ? (uint32_t)oc_json_num(ex, "width", 400) : 400u;
	uint32_t H = ex != NULL ? (uint32_t)oc_json_num(ex, "height", 300) : 300u;
	float focal = ex != NULL ? (float)oc_json_num(ex, "focal", 0.0) : g_splat_focal;
	uint8_t *rgba = NULL;
	if (g_splat_image != NULL && g_splat_image[0] != '\0') {
		rgba = read_ppm(g_splat_image, &W, &H);
		check(rgba != NULL, "read the photo %s (binary PPM)", g_splat_image);
	} else if (ex != NULL) {
		// The tiny model's reference assumes a uniform grey picture.
		rgba = (uint8_t *)malloc((size_t)W * H * 4u);
		if (rgba != NULL) {
			memset(rgba, 128, (size_t)W * H * 4u);
		}
	} else {
		rgba = make_scene(W, H, 0.0f);
	}
	if (rgba == NULL) {
		oc_json_free(ex);
		return;
	}
	struct mem_sink ms;
	memset(&ms, 0, sizeof(ms));
	struct xrt_conv_job_out out;
	struct xrt_conv_splat_result sr;
	double t = 0.0;

	// 1. Focal given (or not, with neither --splat-focal nor expect.json).
	xrt_result_t r = splat_job_once(iface, inst, cap, rgba, W, H, focal, 0, &ms, &out, &sr, &t);
	check(r == XRT_SUCCESS, "splat job (focal %s, result %d, %.0f ms)", focal > 0.0f ? "given" : "absent", (int)r,
	      t);
	if (r == XRT_SUCCESS) {
		const struct xrt_conv_splat_info *si = &out.splat_info;
		uint64_t verts = 0;
		uint32_t props = 0;
		size_t body = 0;
		check(ms.begun && ms.format == XRT_CONV_BLOB_PLY_3DGS && !ms.failed, "job wrote a PLY blob (format %u)",
		      ms.format);
		check((out.written & XRT_CONV_OUT_SPLAT_BLOB) != 0, "job written has SPLAT_BLOB");
		const bool hdr = ply_header(ms.data, ms.size, &verts, &props, &body);
		check(hdr && props == 14 && verts == si->splat_count && body + verts * 56u == ms.size,
		      "PLY header: %llu vertices x %u floats, %zu bytes (splat_count %llu)", (unsigned long long)verts,
		      props, ms.size, (unsigned long long)si->splat_count);
		if (hdr && body + verts * 56u == ms.size) {
			const float *rows = (const float *)(const void *)(ms.data + body);
			size_t bad = 0;
			double zmin = INFINITY, zmax = -INFINITY;
			for (size_t i = 0; i < verts * 14u; i++) {
				bad += isfinite(rows[i]) ? 0u : 1u;
			}
			for (size_t i = 0; i < verts; i++) {
				zmin = fmin(zmin, rows[i * 14 + 2]);
				zmax = fmax(zmax, rows[i * 14 + 2]);
			}
			check(bad == 0, "every PLY value is finite (%zu bad)", bad);
			info("splat cloud: %llu Gaussians, z %.3f .. %.3f", (unsigned long long)verts, zmin, zmax);
			if (ex != NULL) {
				check_expectations(rows, verts, ex);
			}
		}
		check(si->coords == XRT_CONV_COORDS_OPENCV && si->metric == 0 && si->baseline_m == 0.0f,
		      "facts: OpenCV axes, not metric, no baseline");
		check(si->grid_width == W && si->grid_height == H && si->cx == W * 0.5f && si->cy == H * 0.5f,
		      "facts: intrinsics of the %ux%u input (grid %ux%u, c %.1f,%.1f)", W, H, si->grid_width,
		      si->grid_height, (double)si->cx, (double)si->cy);
		if (focal > 0.0f) {
			check(si->intrinsics_source == XRT_CONV_INTRINSICS_GIVEN && si->fx == focal && si->fy == focal,
			      "facts: the given focal is echoed (fx %.2f, source %u)", (double)si->fx,
			      si->intrinsics_source);
		}
		check((sr.valid & XRT_CONV_SPLAT_VALID_BUILD) != 0 &&
		          strncmp(sr.build_provider, "open-conversion", 15) == 0 &&
		          strlen(sr.build_model_sha256) == 64 && sr.host_kind == XRT_CONV_HOST_KIND_LOCAL &&
		          strcmp(sr.build_runtime, cap->backend) == 0 && sr.build_runtime_version[0] != '\0',
		      "build stamp: '%s' / '%s' / %s / %s %s", sr.build_provider, sr.build_model_id,
		      sr.build_model_sha256, sr.build_runtime, sr.build_runtime_version);
		check(sr.refusal_code == XRT_CONV_REFUSAL_NONE, "no refusal (%u)", sr.refusal_code);
		info("focus suggestion: %s (%.3f, %.3f, %.3f)",
		     (sr.valid & XRT_CONV_SPLAT_VALID_FOCUS) ? "yes" : "none", (double)sr.focus_point[0],
		     (double)sr.focus_point[1], (double)sr.focus_point[2]);
		if (g_splat_out != NULL) {
			FILE *f = fopen_utf8(g_splat_out, "wb");
			check(f != NULL && fwrite(ms.data, 1, ms.size, f) == ms.size && fclose(f) == 0, "wrote %s",
			      g_splat_out);
		}
	}
	// --bench: the same job N more times (warm), timed around job_execute.
	double *times = g_bench > 0 ? (double *)calloc(g_bench, sizeof(double)) : NULL;
	uint32_t timed = 0;
	for (uint32_t b = 0; b < g_bench && times != NULL; b++) {
		if (splat_job_once(iface, inst, cap, rgba, W, H, focal, 0, &ms, &out, &sr, &times[timed]) !=
		    XRT_SUCCESS) {
			check(false, "bench job %u", b);
			break;
		}
		timed++;
	}
	if (timed > 0) {
		char what[200];
		snprintf(what, sizeof(what), "splats-mono '%s' on '%s'", cap->model_id, cap->backend);
		report_bench(what, times, timed);
	}
	free(times);

	// 2. No focal: a labelled assumption (30 mm full-frame equivalent), never EXIF.
	r = splat_job_once(iface, inst, cap, rgba, W, H, 0.0f, 0, &ms, &out, &sr, &t);
	check(r == XRT_SUCCESS, "splat job without a focal (result %d)", (int)r);
	if (r == XRT_SUCCESS) {
		const double want = 30.0 * sqrt((double)W * W + (double)H * H) / sqrt(36.0 * 36.0 + 24.0 * 24.0);
		check(out.splat_info.intrinsics_source == XRT_CONV_INTRINSICS_ASSUMED &&
		          fabs(out.splat_info.fx - want) < 1e-3 * want,
		      "facts: no focal -> ASSUMED 30 mm equivalent (fx %.2f, expected %.2f)", (double)out.splat_info.fx,
		      want);
	}

	// 3. A format it does not write: refused (job fails, no blob), the cap stays READY.
	r = splat_job_once(iface, inst, cap, rgba, W, H, focal, XRT_CONV_BLOB_SOG, &ms, &out, &sr, &t);
	check(r != XRT_SUCCESS && !ms.begun && sr.refusal_code == XRT_CONV_REFUSAL_UNSUPPORTED_INPUT &&
	          strstr(sr.refusal_reason, ".ply") != NULL,
	      "a SOG request is refused with a reason naming .ply (result %d, refusal %u '%s')", (int)r, sr.refusal_code,
	      sr.refusal_reason);
	struct xrt_conv_state st;
	memset(&st, 0, sizeof(st));
	st.struct_size = (uint32_t)sizeof(st);
	(void)iface->get_state(inst, cap->cap_id, &st);
	check(st.state == XRT_CONV_STATE_READY, "the cap stays READY after a refused input (%s)", state_name(st.state));

	free(ms.data);
	free(rgba);
	oc_json_free(ex);
}


/*
 *
 * main.
 *
 */

static void
usage(void)
{
	fprintf(stderr,
	        "usage: open-conversion-fakehost <provider> [--set K=V]... [--pack FILE] [--blob-base URL]\n"
	        "         [--store DIR] [--clean-store] [--no-download] [--chunk-ms N] [--verify-ms N] [--caps "
	        "video,mono] [--frames N]\n"
	        "         [--no-warm] [--expect ready|absent|version|policy]\n"
	        "         [--expect-licence permissive|commercial|research] [--expect-download] [--expect-verify]\n"
	        "         [--warm-timeout-ms N] [--scratch DIR] [--bench N] [--dump DIR]\n"
	        "         [--caps splats] [--splat-check expect.json] [--splat-image photo.ppm] [--splat-focal PX]\n"
	        "         [--splat-out out.ply] [--expect-backend NAME] [--expect-model NAME] [--allow-research]\n");
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
	       (task == XRT_CONV_TASK_DEPTH_MONO && strstr(caps, "mono") != NULL) ||
	       (task == XRT_CONV_TASK_SPLATS_MONO && strstr(caps, "splats") != NULL);
}

int
main(int argc, char **argv)
{
	const char *path = NULL, *expect = "ready", *expect_licence = NULL, *pack = NULL, *blob_base = NULL;
	const char *store = NULL, *caps_sel = NULL, *expect_backend = NULL, *expect_model = NULL;
	bool no_warm = false, expect_download = false, expect_verify = false;
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
		} else if (strcmp(a, "--verify-ms") == 0 && more) {
			g_store.verify_ms = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(a, "--expect-verify") == 0) {
			expect_verify = true;
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
		} else if (strcmp(a, "--scratch") == 0 && more) {
			g_scratch = argv[++i];
			mkdir_utf8(g_scratch);
		} else if (strcmp(a, "--bench") == 0 && more) {
			g_bench = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(a, "--dump") == 0 && more) {
			g_dump = argv[++i];
			mkdir_utf8(g_dump);
		} else if (strcmp(a, "--splat-check") == 0 && more) {
			g_splat_check = argv[++i];
		} else if (strcmp(a, "--splat-image") == 0 && more) {
			g_splat_image = argv[++i];
		} else if (strcmp(a, "--splat-focal") == 0 && more) {
			g_splat_focal = strtof(argv[++i], NULL);
		} else if (strcmp(a, "--splat-out") == 0 && more) {
			g_splat_out = argv[++i];
		} else if (strcmp(a, "--expect-backend") == 0 && more) {
			expect_backend = argv[++i];
		} else if (strcmp(a, "--expect-model") == 0 && more) {
			expect_model = argv[++i];
		} else if (strcmp(a, "--allow-research") == 0) {
			// The provider does not gate on licence (selection does); accepted so a
			// test reads like the runtime's opt-in.
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

	char full[4096], err[256] = ""; // realpath() writes up to PATH_MAX (4096 on Linux)
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
	check(r == XRT_SUCCESS && count == 3, "describe_cap count = %u (DEPTH_VIDEO + DEPTH_MONO + SPLATS_MONO)",
	      count);
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
			if (!cap_selected(caps_sel, caps[i].task) || caps[i].model_id[0] == '\0') {
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

	if (expect_model != NULL) {
		for (uint32_t i = 0; i < count; i++) {
			if (cap_selected(caps_sel, caps[i].task)) {
				check(strcmp(caps[i].model_id, expect_model) == 0,
				      "cap %u cold model '%s' (expected '%s')", caps[i].cap_id, caps[i].model_id,
				      expect_model);
			}
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
		if (caps[i].model_id[0] == '\0') {
			// No pack names a model for the task (the default pack has no splat model): ABSENT,
			// unless the runtime it is bound to already failed (that state comes first).
			const bool runtime_state = strcmp(expect, "version") == 0 &&
			                           st.state == XRT_CONV_STATE_FAILED &&
			                           st.failure_reason == XRT_CONV_FAIL_VERSION;
			check(st.state == XRT_CONV_STATE_ABSENT || runtime_state,
			      "cap %u: no model for %s -> ABSENT (%s)", caps[i].cap_id, task_name(caps[i].task),
			      state_name(st.state));
		} else if (strcmp(expect, "absent") == 0) {
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
			if (!cap_selected(caps_sel, caps[i].task) || caps[i].model_id[0] == '\0') {
				continue;
			}
			char ref[192];
			snprintf(ref, sizeof(ref), "%s/%s", g_store.pack_id, caps[i].model_id);
			oc_lock(&g_store.lock);
			const struct blob *b = find_blob(ref);
			const bool reachable = b != NULL && blob_set_reachable(b);
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
		if (!cap_selected(caps_sel, caps[i].task) || caps[i].model_id[0] == '\0') {
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
		if (expect_verify) {
			check((g_watch.seen & (1u << XRT_CONV_STATE_WARMING)) != 0,
			      "cap %u reported WARMING while the store verified", caps[i].cap_id);
			check(t1 - t0 >= g_store.verify_ms, "cap %u waited out the store's %u ms verify (%llu ms)",
			      caps[i].cap_id, g_store.verify_ms, (unsigned long long)(t1 - t0));
		}
		if (st.state != XRT_CONV_STATE_READY) {
			continue;
		}
		// Describe may have changed (DirectML -> CPU fallback): re-read.
		memset(&caps[i], 0, sizeof(caps[i]));
		caps[i].struct_size = (uint32_t)sizeof(caps[i]);
		(void)iface->describe_cap(inst, i, &caps[i], NULL);
		info("cap %u runs on '%s'", caps[i].cap_id, caps[i].backend);
		if (expect_backend != NULL) {
			check(strcmp(caps[i].backend, expect_backend) == 0, "cap %u backend '%s' (expected '%s')",
			      caps[i].cap_id, caps[i].backend, expect_backend);
		}
		if (expect_model != NULL) {
			check(strcmp(caps[i].model_id, expect_model) == 0,
			      "cap %u model '%s' after warm (expected '%s')", caps[i].cap_id, caps[i].model_id,
			      expect_model);
		}
		if (caps[i].task == XRT_CONV_TASK_SPLATS_MONO) {
			run_splat_jobs(iface, inst, &caps[i]);
		} else if ((caps[i].lanes & XRT_CONV_LANE_STREAM) != 0) {
			run_stream(iface, inst, &caps[i], caps[i].task == XRT_CONV_TASK_DEPTH_VIDEO ? frames : 1u);
		}
		if ((caps[i].lanes & XRT_CONV_LANE_JOB) != 0 && caps[i].task != XRT_CONV_TASK_SPLATS_MONO) {
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
