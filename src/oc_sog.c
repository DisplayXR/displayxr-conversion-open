// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  SOG v2 writer (see oc_sog.h). Format: PlayCanvas splat-transform 1.10's
 *         SOG writer (the last to write version 2), as the native DisplayXR
 *         Gaussian Splat viewer reads it; camera block: docs/sog-camera-block.md.
 */

#include "oc_sog.h"

#include <webp/encode.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OC_SOG_CODEBOOK 256
#define OC_SOG_KMEANS_ITERS 64
#define OC_SOG_META_MAX 65536 //!< what SOG readers accept for meta.json
#define OC_SH_C0 0.28209479177387814

static void
seterr(char *err, size_t n, const char *msg)
{
	if (err != NULL && n > 0) {
		snprintf(err, n, "%s", msg);
	}
}


/*
 *
 * Numbers as JSON.
 *
 */

//! The shortest decimal that reads back as the same float, written the way Python's repr writes
//! a float ("1920.0", "1.847", "1e-05"): positional for exponents -4 .. 15, else scientific.
static void
fmt_float(float v, char *buf, size_t cap)
{
	if (!isfinite(v)) {
		snprintf(buf, cap, "0");
		return;
	}
	int p = 1;
	char e[48];
	for (; p <= 9; p++) {
		snprintf(e, sizeof(e), "%.*e", p - 1, (double)v);
		if ((float)strtod(e, NULL) == v) {
			break;
		}
	}
	const char *ep = strchr(e, 'e');
	const int exp10 = ep != NULL ? atoi(ep + 1) : 0;
	if (exp10 >= -4 && exp10 < 16) {
		const int decimals = p - 1 - exp10 > 0 ? p - 1 - exp10 : 0;
		snprintf(buf, cap, "%.*f", decimals, (double)v);
		if (strchr(buf, '.') == NULL) {
			const size_t l = strlen(buf);
			if (l + 2 < cap) {
				memcpy(buf + l, ".0", 3);
			}
		}
	} else {
		// d[.ddd]e[+-]XX, Python style: no '+' padding beyond two digits.
		snprintf(buf, cap, "%.*ge%s%02d", p, (double)strtod(e, NULL) / pow(10.0, exp10), exp10 < 0 ? "-" : "+",
		         abs(exp10));
	}
}

struct sb
{
	char *p;
	size_t len, cap;
	bool ok;
};

static void
sb_add(struct sb *b, const char *s)
{
	const size_t l = strlen(s);
	if (!b->ok || b->len + l + 1 > b->cap) {
		b->ok = false;
		return;
	}
	memcpy(b->p + b->len, s, l + 1);
	b->len += l;
}

static void
sb_num(struct sb *b, float v)
{
	char t[48];
	fmt_float(v, t, sizeof(t));
	sb_add(b, t);
}

size_t
oc_sog_camera_json(const struct oc_sog_camera *cam, char *buf, size_t cap)
{
	struct sb b = {buf, 0, cap, cap > 0};
	char t[64];
	// Key order and literal forms: the DisplayXR Gallery's online SOG generator (json.dumps,
	// compact separators): ints stay ints, floats print as their shortest round-trip form.
	sb_add(&b,
	       "{\"convention\":\"opencv\",\"rig\":\"camera\",\"rest\":{\"position\":[0,0,0],"
	       "\"rotation\":[0,0,0,1]},\"intrinsics\":{\"fx\":");
	sb_num(&b, cam->fx);
	sb_add(&b, ",\"fy\":");
	sb_num(&b, cam->fy);
	sb_add(&b, ",\"cx\":");
	sb_num(&b, cam->cx);
	sb_add(&b, ",\"cy\":");
	sb_num(&b, cam->cy);
	snprintf(t, sizeof(t), ",\"width\":%u,\"height\":%u}", cam->width, cam->height);
	sb_add(&b, t);
	if (cam->focus.valid && cam->focus.z_m > 0.0f) {
		sb_add(&b, ",\"focus\":{\"point\":[0,0,");
		sb_num(&b, cam->focus.z_m);
		sb_add(&b, "],\"subject_m\":");
		sb_num(&b, cam->focus.z_m);
		if (cam->focus.near_m > 0.0f) {
			sb_add(&b, ",\"near_m\":");
			sb_num(&b, cam->focus.near_m);
		}
		if (cam->focus.far_m > 0.0f) {
			sb_add(&b, ",\"far_m\":");
			sb_num(&b, cam->focus.far_m);
		}
		sb_add(&b, ",\"source\":\"nearest-clump\"}");
	}
	sb_add(&b, ",\"dxr\":{\"ipd_factor\":1.0,\"parallax_factor\":1.0}}");
	return b.ok ? b.len : 0;
}


/*
 *
 * Morton order (splat-transform's: 10 bits per axis over the bounds, stable,
 * runs of more than 256 equal codes re-sorted on their own bounds).
 *
 */

static uint32_t
part1by2(uint32_t x)
{
	x &= 0x000003ffu;
	x = (x ^ (x << 16)) & 0xff0000ffu;
	x = (x ^ (x << 8)) & 0x0300f00fu;
	x = (x ^ (x << 4)) & 0x030c30c3u;
	x = (x ^ (x << 2)) & 0x09249249u;
	return x;
}

struct mk
{
	uint32_t code;
	uint32_t pos; //!< position before the sort: ties keep it (stable)
	uint32_t idx;
};

static int
cmp_mk(const void *pa, const void *pb)
{
	const struct mk *a = (const struct mk *)pa, *b = (const struct mk *)pb;
	if (a->code != b->code) {
		return a->code < b->code ? -1 : 1;
	}
	return a->pos < b->pos ? -1 : (a->pos > b->pos ? 1 : 0);
}

static void
morton_sort(const float *rows, uint32_t *idx, size_t n, struct mk *tmp, int depth)
{
	if (n == 0 || depth > 32) {
		return;
	}
	float mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
	for (size_t i = 0; i < n; i++) {
		const float *r = rows + (size_t)idx[i] * OC_SPLAT_FLOATS;
		for (int a = 0; a < 3; a++) {
			mn[a] = r[a] < mn[a] ? r[a] : mn[a];
			mx[a] = r[a] > mx[a] ? r[a] : mx[a];
		}
	}
	double mul[3];
	bool all_same = true;
	for (int a = 0; a < 3; a++) {
		const double len = (double)mx[a] - (double)mn[a];
		if (!isfinite(len)) {
			return;
		}
		all_same = all_same && len == 0.0;
		mul[a] = len == 0.0 ? 0.0 : 1024.0 / len;
	}
	if (all_same) {
		return;
	}
	for (size_t i = 0; i < n; i++) {
		const float *r = rows + (size_t)idx[i] * OC_SPLAT_FLOATS;
		uint32_t q[3];
		for (int a = 0; a < 3; a++) {
			double v = ((double)r[a] - mn[a]) * mul[a];
			v = v < 1023.0 ? v : 1023.0;
			q[a] = (uint32_t)v;
		}
		tmp[i].code = (part1by2(q[2]) << 2) + (part1by2(q[1]) << 1) + part1by2(q[0]);
		tmp[i].pos = (uint32_t)i;
		tmp[i].idx = idx[i];
	}
	qsort(tmp, n, sizeof(*tmp), cmp_mk);
	for (size_t i = 0; i < n; i++) {
		idx[i] = tmp[i].idx;
	}
	// Codes for the runs, before tmp is reused by the recursion.
	uint32_t *codes = (uint32_t *)malloc(sizeof(uint32_t) * n);
	if (codes == NULL) {
		return;
	}
	for (size_t i = 0; i < n; i++) {
		codes[i] = tmp[i].code;
	}
	size_t start = 0;
	while (start < n) {
		size_t end = start + 1;
		while (end < n && codes[end] == codes[start]) {
			end++;
		}
		if (end - start > 256) {
			morton_sort(rows, idx + start, end - start, tmp, depth + 1);
		}
		start = end;
	}
	free(codes);
}


/*
 *
 * 256-entry 1-D codebooks: k-means (Lloyd) on the sorted values with prefix
 * sums, initialised at the quantiles. Deterministic.
 *
 */

static int
cmp_float(const void *pa, const void *pb)
{
	const float a = *(const float *)pa, b = *(const float *)pb;
	return a < b ? -1 : (a > b ? 1 : 0);
}

//! First index in sorted @p v (size n) with v[i] >= x.
static size_t
lower_bound(const float *v, size_t n, double x)
{
	size_t lo = 0, hi = n;
	while (lo < hi) {
		const size_t mid = lo + (hi - lo) / 2;
		if ((double)v[mid] < x) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	return lo;
}

/*! @p vals (n, sorted in place) → an ascending codebook @p cb of OC_SOG_CODEBOOK entries. */
static bool
codebook_1d(float *vals, size_t n, float *cb)
{
	qsort(vals, n, sizeof(float), cmp_float);
	double *prefix = (double *)malloc(sizeof(double) * (n + 1));
	if (prefix == NULL) {
		return false;
	}
	prefix[0] = 0.0;
	for (size_t i = 0; i < n; i++) {
		prefix[i + 1] = prefix[i] + (double)vals[i];
	}
	double c[OC_SOG_CODEBOOK];
	for (int k = 0; k < OC_SOG_CODEBOOK; k++) {
		size_t i = (size_t)(((double)k + 0.5) / OC_SOG_CODEBOOK * (double)n);
		c[k] = vals[i < n ? i : n - 1];
	}
	for (int it = 0; it < OC_SOG_KMEANS_ITERS; it++) {
		bool moved = false;
		size_t begin = 0;
		double next[OC_SOG_CODEBOOK];
		for (int k = 0; k < OC_SOG_CODEBOOK; k++) {
			const size_t end = k == OC_SOG_CODEBOOK - 1 ? n : lower_bound(vals, n, 0.5 * (c[k] + c[k + 1]));
			const size_t e = end < begin ? begin : end;
			next[k] = e > begin ? (prefix[e] - prefix[begin]) / (double)(e - begin) : c[k];
			begin = e;
		}
		for (int k = 0; k < OC_SOG_CODEBOOK; k++) {
			moved = moved || next[k] != c[k];
			c[k] = next[k];
		}
		// Keep it ascending (an empty cluster keeps its centre, which can tie a neighbour).
		for (int k = 1; k < OC_SOG_CODEBOOK; k++) {
			c[k] = c[k] < c[k - 1] ? c[k - 1] : c[k];
		}
		if (!moved) {
			break;
		}
	}
	for (int k = 0; k < OC_SOG_CODEBOOK; k++) {
		cb[k] = (float)c[k];
	}
	free(prefix);
	return true;
}

//! The nearest codebook entry (ascending @p cb) to @p v.
static uint8_t
label_of(const float *cb, float v)
{
	int lo = 0, hi = OC_SOG_CODEBOOK - 1;
	while (lo < hi) {
		const int mid = (lo + hi) / 2;
		if ((double)v > 0.5 * ((double)cb[mid] + (double)cb[mid + 1])) {
			lo = mid + 1;
		} else {
			hi = mid;
		}
	}
	return (uint8_t)lo;
}


/*
 *
 * WebP and ZIP.
 *
 */

static bool
webp_lossless(const uint8_t *rgba, uint32_t w, uint32_t h, uint8_t **out, size_t *out_size)
{
	WebPConfig config;
	WebPPicture pic;
	WebPMemoryWriter wr;
	if (!WebPConfigInit(&config) || !WebPPictureInit(&pic)) {
		return false;
	}
	config.lossless = 1;
	config.exact = 1; // keep the RGB of transparent texels: they are data, not colour
	config.method = 4;
	config.quality = 50.0f;
	config.thread_level = 0;
	pic.use_argb = 1;
	pic.width = (int)w;
	pic.height = (int)h;
	if (!WebPValidateConfig(&config) || !WebPPictureImportRGBA(&pic, rgba, (int)w * 4)) {
		WebPPictureFree(&pic);
		return false;
	}
	WebPMemoryWriterInit(&wr);
	pic.writer = WebPMemoryWrite;
	pic.custom_ptr = &wr;
	const int ok = WebPEncode(&config, &pic);
	WebPPictureFree(&pic);
	if (!ok) {
		WebPMemoryWriterClear(&wr);
		return false;
	}
	*out = wr.mem;
	*out_size = wr.size;
	return true;
}

static uint32_t g_crc_table[256];

static void
crc_init(void)
{
	for (uint32_t n = 0; n < 256; n++) {
		uint32_t c = n;
		for (int k = 0; k < 8; k++) {
			c = (c & 1u) ? 0xedb88320u ^ (c >> 1) : c >> 1;
		}
		g_crc_table[n] = c;
	}
}

static uint32_t
crc32_of(const uint8_t *d, size_t n)
{
	if (g_crc_table[1] == 0) {
		crc_init();
	}
	uint32_t c = 0xffffffffu;
	for (size_t i = 0; i < n; i++) {
		c = g_crc_table[(c ^ d[i]) & 0xffu] ^ (c >> 8);
	}
	return c ^ 0xffffffffu;
}

struct zip_entry
{
	const char *name;
	uint8_t *data;
	size_t size;
};

static void
put16(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)v;
	p[1] = (uint8_t)(v >> 8);
}

static void
put32(uint8_t *p, uint32_t v)
{
	put16(p, v & 0xffffu);
	put16(p + 2, v >> 16);
}

//! A stored ZIP (no compression, no data descriptors, 1980-01-01 00:00 timestamps).
static bool
zip_store(const struct zip_entry *e, int count, struct oc_sog_file *out)
{
	size_t total = 22;
	for (int i = 0; i < count; i++) {
		if (e[i].size > 0xfffffff0u) {
			return false; // no ZIP64: SOG readers do not take it
		}
		total += 30 + 46 + 2 * strlen(e[i].name) + e[i].size;
	}
	uint8_t *z = (uint8_t *)malloc(total);
	if (z == NULL) {
		return false;
	}
	uint32_t offsets[16], crcs[16];
	size_t at = 0;
	const uint32_t dos_time = 0, dos_date = (0u << 9) | (1u << 5) | 1u;
	for (int i = 0; i < count; i++) {
		const size_t nl = strlen(e[i].name);
		offsets[i] = (uint32_t)at;
		crcs[i] = crc32_of(e[i].data, e[i].size);
		uint8_t *h = z + at;
		put32(h, 0x04034b50u);
		put16(h + 4, 20);
		put16(h + 6, 0);
		put16(h + 8, 0);
		put16(h + 10, dos_time);
		put16(h + 12, dos_date);
		put32(h + 14, crcs[i]);
		put32(h + 18, (uint32_t)e[i].size);
		put32(h + 22, (uint32_t)e[i].size);
		put16(h + 26, (uint32_t)nl);
		put16(h + 28, 0);
		memcpy(h + 30, e[i].name, nl);
		memcpy(h + 30 + nl, e[i].data, e[i].size);
		at += 30 + nl + e[i].size;
	}
	const size_t cd = at;
	for (int i = 0; i < count; i++) {
		const size_t nl = strlen(e[i].name);
		uint8_t *h = z + at;
		put32(h, 0x02014b50u);
		put16(h + 4, 20);
		put16(h + 6, 20);
		put16(h + 8, 0);
		put16(h + 10, 0);
		put16(h + 12, dos_time);
		put16(h + 14, dos_date);
		put32(h + 16, crcs[i]);
		put32(h + 20, (uint32_t)e[i].size);
		put32(h + 24, (uint32_t)e[i].size);
		put16(h + 28, (uint32_t)nl);
		put16(h + 30, 0);
		put16(h + 32, 0);
		put16(h + 34, 0);
		put16(h + 36, 0);
		put32(h + 38, 0);
		put32(h + 42, offsets[i]);
		memcpy(h + 46, e[i].name, nl);
		at += 46 + nl;
	}
	uint8_t *h = z + at;
	put32(h, 0x06054b50u);
	put16(h + 4, 0);
	put16(h + 6, 0);
	put16(h + 8, (uint32_t)count);
	put16(h + 10, (uint32_t)count);
	put32(h + 12, (uint32_t)(at - cd));
	put32(h + 16, (uint32_t)cd);
	put16(h + 20, 0);
	at += 22;
	out->data = z;
	out->size = at;
	return true;
}


/*
 *
 * The writer.
 *
 */

static void
sb_codebook(struct sb *b, const float *cb)
{
	sb_add(b, "[");
	for (int k = 0; k < OC_SOG_CODEBOOK; k++) {
		char t[48];
		snprintf(t, sizeof(t), "%s%.9g", k == 0 ? "" : ", ", (double)cb[k]);
		sb_add(b, t);
	}
	sb_add(b, "]");
}

void
oc_sog_free(struct oc_sog_file *f)
{
	if (f != NULL) {
		free(f->data);
		f->data = NULL;
		f->size = 0;
	}
}

static double
log_t(double v)
{
	return (v < 0.0 ? -1.0 : 1.0) * log(fabs(v) + 1.0);
}

bool
oc_sog_write(const float *rows,
             size_t count,
             const struct oc_sog_camera *cam,
             const char *generator,
             bool (*is_cancelled)(void *ctx),
             void *cancel_ctx,
             struct oc_sog_file *out,
             char *err,
             size_t err_size)
{
	memset(out, 0, sizeof(*out));
	if (rows == NULL || count == 0 || count > 100000000u) {
		seterr(err, err_size, "SOG: no Gaussians");
		return false;
	}
	const uint32_t w = (uint32_t)(ceil(sqrt((double)count) / 4.0) * 4.0);
	const uint32_t h = (uint32_t)(ceil((double)count / w / 4.0) * 4.0);
	const size_t texels = (size_t)w * h;
	enum
	{
		MEANS_L,
		MEANS_U,
		QUATS,
		SCALES,
		SH0,
		PLANES
	};
	static const char *const names[PLANES] = {"means_l.webp", "means_u.webp", "quats.webp", "scales.webp",
	                                          "sh0.webp"};
	struct zip_entry ent[PLANES + 1];
	memset(ent, 0, sizeof(ent));
	uint8_t *plane = (uint8_t *)calloc(texels, 4);
	uint32_t *idx = (uint32_t *)malloc(sizeof(uint32_t) * count);
	struct mk *tmp = (struct mk *)malloc(sizeof(struct mk) * count);
	float *pool = (float *)malloc(sizeof(float) * count * 3u);
	char *meta = (char *)malloc(OC_SOG_META_MAX);
	float scb[OC_SOG_CODEBOOK], ccb[OC_SOG_CODEBOOK];
	double mins[3], maxs[3];
	bool ok = false;
	if (plane == NULL || idx == NULL || tmp == NULL || pool == NULL || meta == NULL) {
		seterr(err, err_size, "SOG: out of memory");
		goto done;
	}
	for (size_t i = 0; i < count; i++) {
		idx[i] = (uint32_t)i;
	}
	morton_sort(rows, idx, count, tmp, 0);
	free(tmp);
	tmp = NULL;

	// Means: the log transform's range per axis.
	for (int a = 0; a < 3; a++) {
		mins[a] = INFINITY;
		maxs[a] = -INFINITY;
	}
	for (size_t i = 0; i < count; i++) {
		const float *r = rows + i * OC_SPLAT_FLOATS;
		for (int a = 0; a < 3; a++) {
			const double v = log_t(r[a]);
			mins[a] = v < mins[a] ? v : mins[a];
			maxs[a] = v > maxs[a] ? v : maxs[a];
		}
	}
	for (int a = 0; a < 3; a++) {
		// The values meta.json will hold, so the quantisation uses exactly what a reader decodes with.
		mins[a] = (double)(float)mins[a];
		maxs[a] = (double)(float)maxs[a];
	}
	for (int half = 0; half < 2; half++) {
		memset(plane, 0, texels * 4u);
		for (size_t t = 0; t < count; t++) {
			const float *r = rows + (size_t)idx[t] * OC_SPLAT_FLOATS;
			uint8_t *px = plane + t * 4u;
			for (int a = 0; a < 3; a++) {
				const double span = maxs[a] - mins[a];
				double q = span > 0.0 ? 65535.0 * (log_t(r[a]) - mins[a]) / span : 0.0;
				q = q < 0.0 ? 0.0 : (q > 65535.0 ? 65535.0 : q);
				const uint32_t qi = (uint32_t)lrint(q);
				px[a] = (uint8_t)(half == 0 ? qi & 0xffu : qi >> 8);
			}
			px[3] = 255;
		}
		if (!webp_lossless(plane, w, h, &ent[half].data, &ent[half].size)) {
			seterr(err, err_size, "SOG: WebP encoding failed (means)");
			goto done;
		}
	}
	if (is_cancelled != NULL && is_cancelled(cancel_ctx)) {
		goto done;
	}

	// Quaternions: smallest three.
	memset(plane, 0, texels * 4u);
	for (size_t t = 0; t < count; t++) {
		const float *r = rows + (size_t)idx[t] * OC_SPLAT_FLOATS;
		double q[4] = {r[10], r[11], r[12], r[13]};
		double l = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
		if (!(l > 0.0)) {
			q[0] = 1.0, q[1] = q[2] = q[3] = 0.0, l = 1.0;
		}
		int m = 0;
		for (int k = 0; k < 4; k++) {
			q[k] /= l;
			m = fabs(q[k]) > fabs(q[m]) ? k : m;
		}
		const double sgn = q[m] < 0.0 ? -1.0 : 1.0;
		uint8_t *px = plane + t * 4u;
		int c = 0;
		for (int k = 0; k < 4; k++) {
			if (k == m) {
				continue;
			}
			double v = 255.0 * (sgn * q[k] * sqrt(2.0) * 0.5 + 0.5);
			v = v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v);
			px[c++] = (uint8_t)lrint(v);
		}
		px[3] = (uint8_t)(252 + m);
	}
	if (!webp_lossless(plane, w, h, &ent[QUATS].data, &ent[QUATS].size)) {
		seterr(err, err_size, "SOG: WebP encoding failed (quats)");
		goto done;
	}
	if (is_cancelled != NULL && is_cancelled(cancel_ctx)) {
		goto done;
	}

	// Scales (log) and sh0 (f_dc): pooled 256-entry codebooks.
	for (int which = 0; which < 2; which++) {
		const int col = which == 0 ? 7 : 3;
		float *cb = which == 0 ? scb : ccb;
		for (size_t i = 0; i < count; i++) {
			memcpy(pool + i * 3u, rows + i * OC_SPLAT_FLOATS + col, sizeof(float) * 3u);
		}
		if (!codebook_1d(pool, count * 3u, cb)) {
			seterr(err, err_size, "SOG: out of memory");
			goto done;
		}
		memset(plane, 0, texels * 4u);
		for (size_t t = 0; t < count; t++) {
			const float *r = rows + (size_t)idx[t] * OC_SPLAT_FLOATS;
			uint8_t *px = plane + t * 4u;
			for (int k = 0; k < 3; k++) {
				px[k] = label_of(cb, r[col + k]);
			}
			if (which == 0) {
				px[3] = 255;
			} else {
				const double a = 255.0 / (1.0 + exp(-(double)r[6]));
				px[3] = (uint8_t)lrint(a < 0.0 ? 0.0 : (a > 255.0 ? 255.0 : a));
			}
		}
		const int e = which == 0 ? SCALES : SH0;
		if (!webp_lossless(plane, w, h, &ent[e].data, &ent[e].size)) {
			seterr(err, err_size, "SOG: WebP encoding failed (scales / sh0)");
			goto done;
		}
		if (is_cancelled != NULL && is_cancelled(cancel_ctx)) {
			goto done;
		}
	}

	// meta.json: splat-transform's layout (two-space indent), the camera block on one line after count.
	{
		struct sb b = {meta, 0, OC_SOG_META_MAX, true};
		char t[512];
		snprintf(t, sizeof(t),
		         "{\n  \"version\": 2,\n  \"asset\": {\n    \"generator\": \"%s\"\n  },\n  \"count\": %zu,\n",
		         generator != NULL ? generator : "displayxr-conversion-open", count);
		sb_add(&b, t);
		if (cam != NULL) {
			char block[2048];
			if (oc_sog_camera_json(cam, block, sizeof(block)) == 0) {
				seterr(err, err_size, "SOG: camera block too long");
				goto done;
			}
			sb_add(&b, "  \"camera\": ");
			sb_add(&b, block);
			sb_add(&b, ",\n");
		}
		sb_add(&b, "  \"means\": {\n    \"mins\": [");
		for (int a = 0; a < 3; a++) {
			snprintf(t, sizeof(t), "%s%.9g", a == 0 ? "" : ", ", mins[a]);
			sb_add(&b, t);
		}
		sb_add(&b, "],\n    \"maxs\": [");
		for (int a = 0; a < 3; a++) {
			snprintf(t, sizeof(t), "%s%.9g", a == 0 ? "" : ", ", maxs[a]);
			sb_add(&b, t);
		}
		sb_add(&b,
		       "],\n    \"files\": [\"means_l.webp\", \"means_u.webp\"]\n  },\n  \"scales\": {\n    "
		       "\"codebook\": ");
		sb_codebook(&b, scb);
		sb_add(
		    &b,
		    ",\n    \"files\": [\"scales.webp\"]\n  },\n  \"quats\": {\n    \"files\": [\"quats.webp\"]\n  },\n"
		    "  \"sh0\": {\n    \"codebook\": ");
		sb_codebook(&b, ccb);
		sb_add(&b, ",\n    \"files\": [\"sh0.webp\"]\n  }\n}\n");
		if (!b.ok) {
			seterr(err, err_size, "SOG: meta.json over 64 KiB");
			goto done;
		}
		ent[PLANES].data = (uint8_t *)meta;
		ent[PLANES].size = b.len;
	}
	for (int i = 0; i < PLANES; i++) {
		ent[i].name = names[i];
	}
	ent[PLANES].name = "meta.json";
	if (!zip_store(ent, PLANES + 1, out)) {
		seterr(err, err_size, "SOG: out of memory (zip)");
		goto done;
	}
	ok = true;
done:
	for (int i = 0; i < PLANES; i++) {
		WebPFree(ent[i].data);
	}
	free(meta);
	free(plane);
	free(idx);
	free(tmp);
	free(pool);
	return ok;
}
