// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Unit test of the SOG v2 writer (src/oc_sog.c) and the nearest-clump
 *         focus (src/oc_splat.c): synthetic clouds, no model. The SOG is read
 *         back the way a SOG v2 reader decodes it (stored ZIP, lossless WebP,
 *         log-transformed 16-bit means, smallest-three quaternions, 256-entry
 *         codebooks, opacity in sh0's alpha) and compared with the rows.
 */

#include "oc_json.h"
#include "oc_sog.h"
#include "oc_splat.h"

#include <webp/decode.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed = 0;

#define CHECK(cond, ...)                                                                                               \
	do {                                                                                                           \
		if (!(cond)) {                                                                                         \
			g_failed++;                                                                                    \
			printf("FAIL: ");                                                                              \
		} else {                                                                                               \
			printf("ok:   ");                                                                              \
		}                                                                                                      \
		printf(__VA_ARGS__);                                                                                   \
		printf("\n");                                                                                          \
	} while (0)

//! A deterministic uniform in [0, 1).
static double
rnd(uint32_t *s)
{
	*s = *s * 1664525u + 1013904223u;
	return (double)(*s >> 8) / 16777216.0;
}

static double
logit(double a)
{
	return log(a / (1.0 - a));
}

//! A row at (x, y, z) with opacity @p a; the rest neutral.
static void
row(float *r, double x, double y, double z, double a)
{
	memset(r, 0, sizeof(float) * OC_SPLAT_FLOATS);
	r[0] = (float)x;
	r[1] = (float)y;
	r[2] = (float)z;
	r[6] = (float)logit(a);
	r[7] = r[8] = r[9] = -4.0f;
	r[10] = 1.0f;
}

static void
test_clump(void)
{
	const double f = 1000.0, W = 1600.0, H = 1200.0;
	const size_t n = 100000;
	float *rows = (float *)malloc(sizeof(float) * OC_SPLAT_FLOATS * n);
	uint32_t s = 1;
	// 80 %: a far wall at 8-12 m filling the frame; 20 %: a subject at 1.9-2.1 m in the centre.
	for (size_t i = 0; i < n; i++) {
		const bool subject = i % 5 == 0;
		const double z = subject ? 1.9 + 0.2 * rnd(&s) : 8.0 + 4.0 * rnd(&s);
		const double span = subject ? 0.2 : 0.9; // fraction of the half-frame
		const double u = (rnd(&s) * 2.0 - 1.0) * span * 0.5, v = (rnd(&s) * 2.0 - 1.0) * span * 0.5;
		row(rows + i * OC_SPLAT_FLOATS, u * W / f * z, v * H / f * z, z, 0.9);
	}
	struct oc_splat_focus fo;
	bool ok = oc_splat_nearest_clump(rows, n, f, W, H, &fo);
	CHECK(ok && fo.valid && fabs(fo.z_m - 2.0) < 0.05, "near clump found at %.3f m (expected ~2.0)",
	      (double)fo.z_m);
	CHECK(fo.near_m > 1.85f && fo.near_m < 2.2f && fo.far_m > 9.0f && fo.far_m < 12.1f,
	      "near / far = weighted p05 / p95 of z over the crop: %.3f / %.3f", (double)fo.near_m, (double)fo.far_m);
	CHECK(fo.n > 0 && fo.n <= 40000u, "stride-sampled to <= 40 000 (%u in the crop)", fo.n);

	// A sliver nearer than the subject holding < 3 % of the mass is skipped (noise, not a subject).
	for (size_t i = 1; i < n; i += 100) {
		row(rows + i * OC_SPLAT_FLOATS, 0.0, 0.0, 0.5, 0.9);
	}
	ok = oc_splat_nearest_clump(rows, n, f, W, H, &fo);
	CHECK(ok && fabs(fo.z_m - 2.0) < 0.05, "a 1 %% sliver at 0.5 m is skipped: %.3f m", (double)fo.z_m);

	// Transparent (opacity < 0.05) Gaussians do not count (the subject and the sliver).
	for (size_t i = 1; i < n; i += 100) {
		rows[i * OC_SPLAT_FLOATS + 6] = (float)logit(0.04);
	}
	for (size_t i = 0; i < n; i += 5) {
		rows[i * OC_SPLAT_FLOATS + 6] = (float)logit(0.04);
	}
	ok = oc_splat_nearest_clump(rows, n, f, W, H, &fo);
	CHECK(ok && fo.z_m > 7.5f, "a transparent subject is ignored: the wall at %.3f m", (double)fo.z_m);

	// Nothing in the central half of the frame: no focus.
	for (size_t i = 0; i < n; i++) {
		row(rows + i * OC_SPLAT_FLOATS, 0.4 * W / f * 3.0, 0.0, 3.0, 0.9);
	}
	ok = oc_splat_nearest_clump(rows, n, f, W, H, &fo);
	CHECK(!ok && !fo.valid, "no Gaussian in the central half: no focus");

	// The clamp: a clump at 50 m reads 20.
	for (size_t i = 0; i < n; i++) {
		row(rows + i * OC_SPLAT_FLOATS, 0.0, 0.0, 50.0 + (double)(i % 7), 0.9);
	}
	ok = oc_splat_nearest_clump(rows, n, f, W, H, &fo);
	CHECK(ok && fo.z_m == 20.0f, "clamped to 20 m (%.3f)", (double)fo.z_m);
	free(rows);
}

static void
test_camera_json(void)
{
	struct oc_sog_camera cam = {1527.441f, 1527.441f, 1920.0f, 1080.0f, 3840, 2160, {0}};
	char buf[1024];
	size_t n = oc_sog_camera_json(&cam, buf, sizeof(buf));
	const char *want0 =
	    "{\"convention\":\"opencv\",\"rig\":\"camera\",\"rest\":{\"position\":[0,0,0],\"rotation\":[0,0,0,1]},"
	    "\"intrinsics\":{\"fx\":1527.441,\"fy\":1527.441,\"cx\":1920.0,\"cy\":1080.0,\"width\":3840,"
	    "\"height\":2160},\"dxr\":{\"ipd_factor\":1.0,\"parallax_factor\":1.0}}";
	CHECK(n > 0 && strcmp(buf, want0) == 0, "camera block without a clump: no focus key\n      %s", buf);
	cam.focus.valid = true;
	cam.focus.z_m = 1.847f;
	cam.focus.near_m = 1.2f;
	cam.focus.far_m = 14.5f;
	n = oc_sog_camera_json(&cam, buf, sizeof(buf));
	CHECK(n > 0 &&
	          strstr(buf,
	                 ",\"focus\":{\"point\":[0,0,1.847],\"subject_m\":1.847,\"near_m\":1.2,\"far_m\":14.5,"
	                 "\"source\":\"nearest-clump\"},\"dxr\":") != NULL &&
	          strstr(buf, "stereo") == NULL,
	      "camera block with the clump\n      %s", buf);
	CHECK(oc_sog_camera_json(&cam, buf, 64) == 0, "a too-small buffer is refused");
}

//! A stored ZIP member (the writer stores every member).
static const uint8_t *
member(const struct oc_sog_file *f, const char *name, size_t *size)
{
	const size_t nl = strlen(name);
	for (size_t at = 0; at + 30 <= f->size;) {
		const uint8_t *h = f->data + at;
		if (!(h[0] == 'P' && h[1] == 'K' && h[2] == 3 && h[3] == 4)) {
			return NULL;
		}
		const uint32_t csize = h[18] | (uint32_t)h[19] << 8 | (uint32_t)h[20] << 16 | (uint32_t)h[21] << 24;
		const uint32_t fl = h[26] | (uint32_t)h[27] << 8, xl = h[28] | (uint32_t)h[29] << 8;
		const uint32_t method = h[8] | (uint32_t)h[9] << 8;
		if (fl == nl && memcmp(h + 30, name, nl) == 0) {
			*size = csize;
			return method == 0 ? h + 30 + fl + xl : NULL;
		}
		at += 30 + fl + xl + csize;
	}
	return NULL;
}

static uint8_t *
plane(const struct oc_sog_file *f, const char *name, int *w, int *h)
{
	size_t sz = 0;
	const uint8_t *d = member(f, name, &sz);
	return d != NULL ? WebPDecodeRGBA(d, sz, w, h) : NULL;
}

static double
inv_log(double v)
{
	return (v < 0.0 ? -1.0 : 1.0) * (exp(fabs(v)) - 1.0);
}

static void
test_round_trip(void)
{
	const size_t n = 5000;
	float *rows = (float *)malloc(sizeof(float) * OC_SPLAT_FLOATS * n);
	uint32_t s = 7;
	for (size_t i = 0; i < n; i++) {
		float *r = rows + i * OC_SPLAT_FLOATS;
		const double z = 0.5 + 30.0 * rnd(&s);
		r[0] = (float)((rnd(&s) - 0.5) * z);
		r[1] = (float)((rnd(&s) - 0.5) * z * 0.6);
		r[2] = (float)z;
		for (int k = 0; k < 3; k++) {
			r[3 + k] = (float)((rnd(&s) - 0.5) * 3.4);
			r[7 + k] = (float)(-7.0 + 5.0 * rnd(&s));
		}
		r[6] = (float)logit(0.001 + 0.998 * rnd(&s));
		double q[4], l = 0.0;
		for (int k = 0; k < 4; k++) {
			q[k] = rnd(&s) - 0.5;
			l += q[k] * q[k];
		}
		for (int k = 0; k < 4; k++) {
			r[10 + k] = (float)(q[k] / sqrt(l));
		}
	}
	struct oc_sog_camera cam = {800.0f, 800.0f, 640.0f, 360.0f, 1280, 720, {0}};
	struct oc_sog_file a, b;
	char err[128] = "";
	const bool ok = oc_sog_write(rows, n, &cam, "sog-test", NULL, NULL, &a, err, sizeof(err));
	CHECK(ok, "oc_sog_write (%zu bytes) %s", a.size, err);
	if (!ok) {
		free(rows);
		return;
	}
	CHECK(oc_sog_write(rows, n, &cam, "sog-test", NULL, NULL, &b, err, sizeof(err)) && b.size == a.size &&
	          memcmp(a.data, b.data, a.size) == 0,
	      "deterministic: two writes, identical bytes");
	oc_sog_free(&b);

	size_t msz = 0;
	const uint8_t *md = member(&a, "meta.json", &msz);
	struct oc_json *meta = md != NULL ? oc_json_parse((const char *)md, msz, err, sizeof(err)) : NULL;
	CHECK(meta != NULL, "meta.json is stored and parses (%zu bytes) %s", msz, err);
	if (meta == NULL) {
		oc_sog_free(&a);
		free(rows);
		return;
	}
	const struct oc_json *means = oc_json_get(meta, "means");
	const struct oc_json *scb = oc_json_get(oc_json_get(meta, "scales"), "codebook");
	const struct oc_json *ccb = oc_json_get(oc_json_get(meta, "sh0"), "codebook");
	CHECK(oc_json_num(meta, "version", 0) == 2 && (size_t)oc_json_num(meta, "count", 0) == n && means != NULL &&
	          scb != NULL && scb->count == 256 && ccb != NULL && ccb->count == 256 &&
	          oc_json_get(meta, "camera") != NULL && oc_json_get(meta, "shN") == NULL,
	      "meta.json: version 2, count, means, 256-entry codebooks, camera, no shN");
	// The camera key follows count (the order the DisplayXR Gallery writes).
	const char *pc = strstr((const char *)md, "\"count\""), *pk = strstr((const char *)md, "\"camera\"");
	const char *pm = strstr((const char *)md, "\"means\"");
	CHECK(pc != NULL && pk != NULL && pm != NULL && pc < pk && pk < pm, "the camera block sits after count");

	int w = 0, h = 0, w2 = 0, h2 = 0;
	uint8_t *ml = plane(&a, "means_l.webp", &w, &h), *mu = plane(&a, "means_u.webp", &w2, &h2);
	uint8_t *qs = plane(&a, "quats.webp", &w2, &h2), *sc = plane(&a, "scales.webp", &w2, &h2);
	uint8_t *sh = plane(&a, "sh0.webp", &w2, &h2);
	const int want_w = (int)(ceil(sqrt((double)n) / 4.0) * 4.0);
	CHECK(ml && mu && qs && sc && sh && w == want_w && h == (int)(ceil((double)n / want_w / 4.0) * 4.0) &&
	          w % 4 == 0 && h % 4 == 0 && (size_t)w * h >= n,
	      "five lossless WebP planes %dx%d (multiples of 4, >= count)", w, h);
	if (ml && mu && qs && sc && sh) {
		double mn[3], mx[3];
		for (int k = 0; k < 3; k++) {
			mn[k] = oc_json_at(oc_json_get(means, "mins"), (size_t)k)->number;
			mx[k] = oc_json_at(oc_json_get(means, "maxs"), (size_t)k)->number;
		}
		// Texel order is Morton order: match each decoded texel to its row by position (unique here).
		double worst_pos = 0.0, worst_q = 0.0, worst_s = 0.0, worst_c = 0.0, worst_a = 0.0;
		size_t matched = 0;
		bool *used = (bool *)calloc(n, sizeof(bool));
		for (size_t t = 0; t < n; t++) {
			const uint8_t *L = ml + t * 4, *U = mu + t * 4, *Q = qs + t * 4, *S = sc + t * 4,
			              *C = sh + t * 4;
			double p[3];
			for (int k = 0; k < 3; k++) {
				const double q = (double)(L[k] | U[k] << 8) / 65535.0;
				p[k] = inv_log(mn[k] + q * (mx[k] - mn[k]));
			}
			// Nearest row (brute force on 5000).
			size_t best = n;
			double bd = INFINITY;
			for (size_t i = 0; i < n; i++) {
				const float *r = rows + i * OC_SPLAT_FLOATS;
				const double d = fabs(r[0] - p[0]) + fabs(r[1] - p[1]) + fabs(r[2] - p[2]);
				if (d < bd) {
					bd = d;
					best = i;
				}
			}
			if (best == n || used[best]) {
				continue;
			}
			used[best] = true;
			matched++;
			const float *r = rows + best * OC_SPLAT_FLOATS;
			for (int k = 0; k < 3; k++) {
				const double rel = fabs(p[k] - r[k]) / (fabs(r[k]) + 1.0);
				worst_pos = rel > worst_pos ? rel : worst_pos;
			}
			// Quaternion: smallest three.
			const int m = Q[3] - 252;
			double q[4];
			int c = 0;
			double ss = 0.0;
			for (int k = 0; k < 4; k++) {
				if (k == m) {
					continue;
				}
				q[k] = ((double)Q[c++] / 255.0 - 0.5) * 2.0 / sqrt(2.0);
				ss += q[k] * q[k];
			}
			q[m] = sqrt(fmax(0.0, 1.0 - ss));
			const double dot = fabs(q[0] * r[10] + q[1] * r[11] + q[2] * r[12] + q[3] * r[13]);
			const double ang = 2.0 * acos(fmin(1.0, dot)) * 180.0 / 3.14159265358979323846;
			worst_q = ang > worst_q ? ang : worst_q;
			for (int k = 0; k < 3; k++) {
				const struct oc_json *se = oc_json_at(scb, S[k]), *ce = oc_json_at(ccb, C[k]);
				const double ds = fabs(se->number - r[7 + k]), dc = fabs(ce->number - r[3 + k]);
				worst_s = ds > worst_s ? ds : worst_s;
				worst_c = dc > worst_c ? dc : worst_c;
			}
			const double da = fabs((double)C[3] / 255.0 - 1.0 / (1.0 + exp(-(double)r[6])));
			worst_a = da > worst_a ? da : worst_a;
		}
		free(used);
		CHECK(matched == n, "every texel decodes to its own row (%zu of %zu)", matched, n);
		CHECK(worst_pos < 1e-3, "positions: worst |error| / (|v| + 1) %.2e (16-bit log quantisation)",
		      worst_pos);
		CHECK(worst_q < 1.5, "rotations: worst angle %.3f deg (8-bit smallest three)", worst_q);
		CHECK(worst_s < 0.03, "log scales: worst codebook error %.4f (range 5)", worst_s);
		CHECK(worst_c < 0.03, "f_dc: worst codebook error %.4f (range 3.4)", worst_c);
		CHECK(worst_a <= 0.5 / 255.0 + 1e-6, "opacity: worst error %.5f (<= 1/510)", worst_a);
	}
	WebPFree(ml);
	WebPFree(mu);
	WebPFree(qs);
	WebPFree(sc);
	WebPFree(sh);
	oc_json_free(meta);
	oc_sog_free(&a);
	free(rows);
}

int
main(void)
{
	test_clump();
	test_camera_json();
	test_round_trip();
	printf("%s (%d failed)\n", g_failed == 0 ? "PASS" : "FAIL", g_failed);
	return g_failed == 0 ? 0 : 1;
}
