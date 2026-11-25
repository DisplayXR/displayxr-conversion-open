// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Splat pre- and post-processing, see oc_splat.h.
 */

#include "oc_splat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OC_SPLAT_CHUNK 8192u         //!< Gaussians per sink write (and per cancel poll)
#define OC_SH_C0 0.28209479177387814 //!< degree-0 spherical-harmonic constant

// Nearest-clump focus (docs/sog-camera-block.md): the DisplayXR Gallery's constants, verbatim.
#define OC_CLUMP_SAMPLE_CAP 40000u
#define OC_CLUMP_MIN_OPACITY 0.05
#define OC_CLUMP_MIN_Z 0.05
#define OC_CLUMP_CENTRAL_FRAC 0.5
#define OC_CLUMP_N_BINS 120
#define OC_CLUMP_MIN_MASS_FRAC 0.03
#define OC_CLUMP_NOISE_FLOOR_FRAC 0.001
#define OC_CLUMP_PIVOT_MIN_M 0.1
#define OC_CLUMP_PIVOT_MAX_M 20.0

static const char *const k_props[OC_SPLAT_FLOATS] = {"x",      "y",       "z",       "f_dc_0",  "f_dc_1",
                                                     "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                                                     "rot_0",  "rot_1",   "rot_2",   "rot_3"};

/*
 *
 * Resampling.
 *
 */

//! One output sample's source span and normalised triangle weights.
struct span
{
	uint32_t first, count;
	size_t weights; //!< offset into the weight array
};

//! Triangle-filter weights for resampling @p in samples into @p out (support widens with downscale).
static bool
make_spans(uint32_t in, uint32_t out, struct span **spans_out, float **weights_out)
{
	const double scale = (double)in / (double)out;
	const double fscale = scale > 1.0 ? scale : 1.0;
	const double support = fscale; // triangle: radius 1 in output units
	const uint32_t max_taps = (uint32_t)ceil(support) * 2u + 2u;
	struct span *spans = (struct span *)malloc(sizeof(*spans) * out);
	float *w = (float *)malloc(sizeof(float) * (size_t)out * max_taps);
	if (spans == NULL || w == NULL) {
		free(spans);
		free(w);
		return false;
	}
	for (uint32_t o = 0; o < out; o++) {
		const double center = ((double)o + 0.5) * scale;
		long lo = (long)floor(center - support + 0.5);
		long hi = (long)floor(center + support + 0.5);
		if (lo < 0) {
			lo = 0;
		}
		if (hi > (long)in) {
			hi = (long)in;
		}
		if (hi - lo > (long)max_taps) {
			hi = lo + (long)max_taps;
		}
		double sum = 0.0;
		float *ow = w + (size_t)o * max_taps;
		for (long i = lo; i < hi; i++) {
			const double x = fabs(((double)i - center + 0.5) / fscale);
			const double v = x < 1.0 ? 1.0 - x : 0.0;
			ow[i - lo] = (float)v;
			sum += v;
		}
		if (sum <= 0.0) {
			// Degenerate (cannot happen for in, out > 0): nearest sample.
			long n = (long)floor(center);
			lo = n < 0 ? 0 : (n >= (long)in ? (long)in - 1 : n);
			hi = lo + 1;
			ow[0] = 1.0f;
			sum = 1.0;
		}
		for (long i = 0; i < hi - lo; i++) {
			ow[i] = (float)(ow[i] / sum);
		}
		spans[o].first = (uint32_t)lo;
		spans[o].count = (uint32_t)(hi - lo);
		spans[o].weights = (size_t)o * max_taps;
	}
	*spans_out = spans;
	*weights_out = w;
	return true;
}

bool
oc_resample_rgb_aa(
    const uint8_t *src, uint32_t stride, uint32_t sw, uint32_t sh, bool bgr, uint32_t dw, uint32_t dh, float *chw)
{
	struct span *hs = NULL, *vs = NULL;
	float *hw = NULL, *vw = NULL;
	float *tmp = (float *)malloc(sizeof(float) * 3u * (size_t)dw * sh); // sh rows of dw RGB
	bool ok = tmp != NULL && make_spans(sw, dw, &hs, &hw) && make_spans(sh, dh, &vs, &vw);
	if (!ok) {
		free(tmp);
		free(hs);
		free(hw);
		free(vs);
		free(vw);
		return false;
	}
	const int ri = bgr ? 2 : 0, bi = bgr ? 0 : 2;
	const float k = 1.0f / 255.0f;
	// Horizontal pass: every source row → dw samples.
	for (uint32_t y = 0; y < sh; y++) {
		const uint8_t *row = src + (size_t)y * stride;
		float *t = tmp + (size_t)y * dw * 3u;
		for (uint32_t x = 0; x < dw; x++) {
			const struct span *s = &hs[x];
			const float *w = hw + s->weights;
			float r = 0.0f, g = 0.0f, b = 0.0f;
			for (uint32_t i = 0; i < s->count; i++) {
				const uint8_t *p = row + (size_t)(s->first + i) * 4u;
				r += w[i] * (float)p[ri];
				g += w[i] * (float)p[1];
				b += w[i] * (float)p[bi];
			}
			t[x * 3u + 0] = r * k;
			t[x * 3u + 1] = g * k;
			t[x * 3u + 2] = b * k;
		}
	}
	// Vertical pass → planar CHW.
	const size_t plane = (size_t)dw * dh;
	for (uint32_t y = 0; y < dh; y++) {
		const struct span *s = &vs[y];
		const float *w = vw + s->weights;
		for (uint32_t x = 0; x < dw; x++) {
			float r = 0.0f, g = 0.0f, b = 0.0f;
			for (uint32_t i = 0; i < s->count; i++) {
				const float *p = tmp + ((size_t)(s->first + i) * dw + x) * 3u;
				r += w[i] * p[0];
				g += w[i] * p[1];
				b += w[i] * p[2];
			}
			const size_t o = (size_t)y * dw + x;
			chw[o] = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
			chw[plane + o] = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
			chw[2u * plane + o] = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
		}
	}
	free(tmp);
	free(hs);
	free(hw);
	free(vs);
	free(vw);
	return true;
}


/*
 *
 * Gaussians.
 *
 */

//! Rotation matrix (row-major) from a quaternion w x y z (normalised here).
static void
quat_to_rot(const float q[4], double R[3][3])
{
	double w = q[0], x = q[1], y = q[2], z = q[3];
	const double n = sqrt(w * w + x * x + y * y + z * z);
	if (n > 0.0) {
		w /= n;
		x /= n;
		y /= n;
		z /= n;
	} else {
		w = 1.0;
	}
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

//! Quaternion w x y z of a proper rotation matrix (Shepperd's method), normalised.
static void
rot_to_quat(const double R[3][3], float q[4])
{
	double w, x, y, z;
	const double tr = R[0][0] + R[1][1] + R[2][2];
	if (tr > 0.0) {
		const double s = sqrt(tr + 1.0) * 2.0;
		w = 0.25 * s;
		x = (R[2][1] - R[1][2]) / s;
		y = (R[0][2] - R[2][0]) / s;
		z = (R[1][0] - R[0][1]) / s;
	} else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
		const double s = sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0;
		w = (R[2][1] - R[1][2]) / s;
		x = 0.25 * s;
		y = (R[0][1] + R[1][0]) / s;
		z = (R[0][2] + R[2][0]) / s;
	} else if (R[1][1] > R[2][2]) {
		const double s = sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0;
		w = (R[0][2] - R[2][0]) / s;
		x = (R[0][1] + R[1][0]) / s;
		y = 0.25 * s;
		z = (R[1][2] + R[2][1]) / s;
	} else {
		const double s = sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0;
		w = (R[1][0] - R[0][1]) / s;
		x = (R[0][2] + R[2][0]) / s;
		y = (R[1][2] + R[2][1]) / s;
		z = 0.25 * s;
	}
	const double n = sqrt(w * w + x * x + y * y + z * z);
	q[0] = (float)(w / n);
	q[1] = (float)(x / n);
	q[2] = (float)(y / n);
	q[3] = (float)(z / n);
}

/*! Eigen-decomposition of a symmetric 3×3 (cyclic Jacobi, double): @p ev
 *  descending, @p V's columns the matching unit eigenvectors, det(V) = +1. */
static void
sym_eig3(double A[3][3], double ev[3], double V[3][3])
{
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			V[i][j] = i == j ? 1.0 : 0.0;
		}
	}
	for (int sweep = 0; sweep < 16; sweep++) {
		const double off = fabs(A[0][1]) + fabs(A[0][2]) + fabs(A[1][2]);
		const double diag = fabs(A[0][0]) + fabs(A[1][1]) + fabs(A[2][2]);
		if (off <= 1e-15 * diag || off == 0.0) {
			break;
		}
		for (int p = 0; p < 2; p++) {
			for (int q = p + 1; q < 3; q++) {
				if (A[p][q] == 0.0) {
					continue;
				}
				const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
				const double t =
				    (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
				const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
				for (int k = 0; k < 3; k++) { // A = J^T A J
					const double akp = A[k][p], akq = A[k][q];
					A[k][p] = c * akp - s * akq;
					A[k][q] = s * akp + c * akq;
				}
				for (int k = 0; k < 3; k++) {
					const double apk = A[p][k], aqk = A[q][k];
					A[p][k] = c * apk - s * aqk;
					A[q][k] = s * apk + c * aqk;
				}
				for (int k = 0; k < 3; k++) {
					const double vkp = V[k][p], vkq = V[k][q];
					V[k][p] = c * vkp - s * vkq;
					V[k][q] = s * vkp + c * vkq;
				}
			}
		}
	}
	int order[3] = {0, 1, 2};
	for (int i = 0; i < 2; i++) {
		for (int j = i + 1; j < 3; j++) {
			if (A[order[j]][order[j]] > A[order[i]][order[i]]) {
				const int t = order[i];
				order[i] = order[j];
				order[j] = t;
			}
		}
	}
	double S[3][3];
	for (int c = 0; c < 3; c++) {
		ev[c] = A[order[c]][order[c]];
		for (int r = 0; r < 3; r++) {
			S[r][c] = V[r][order[c]];
		}
	}
	const double det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) -
	                   S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0]) +
	                   S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 3; c++) {
			V[r][c] = (c == 2 && det < 0.0) ? -S[r][c] : S[r][c];
		}
	}
}

static float
srgb_encode(float v)
{
	v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
	return v <= 0.0031308f ? v * 12.92f : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
}

static bool
all_finite(const float *v, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		if (!isfinite(v[i])) {
			return false;
		}
	}
	return true;
}

static bool
gaussian_finite(const struct oc_splat_raw *raw, size_t i)
{
	return all_finite(raw->means + i * 3u, 3) && all_finite(raw->scales + i * 3u, 3) &&
	       all_finite(raw->quats + i * 4u, 4) && all_finite(raw->colors + i * 3u, 3) && isfinite(raw->opacities[i]);
}

//! One Gaussian → 14 PLY floats.
static void
encode(const struct oc_splat_raw *raw, size_t i, const double S[3], float *o)
{
	const float *m = raw->means + i * 3u, *s = raw->scales + i * 3u, *c = raw->colors + i * 3u;
	o[0] = (float)(m[0] * S[0]);
	o[1] = (float)(m[1] * S[1]);
	o[2] = (float)(m[2] * S[2]);
	for (int k = 0; k < 3; k++) {
		const float v =
		    raw->colors_linear ? srgb_encode(c[k]) : (c[k] < 0.0f ? 0.0f : (c[k] > 1.0f ? 1.0f : c[k]));
		o[3 + k] = (float)((v - 0.5) / OC_SH_C0);
	}
	double a = raw->opacities[i];
	a = a < 1e-6 ? 1e-6 : (a > 1.0 - 1e-6 ? 1.0 - 1e-6 : a);
	o[6] = (float)log(a / (1.0 - a));
	// cov' = D R diag(s^2) R^T D, D = diag(S): re-diagonalise.
	double R[3][3], C[3][3], ev[3], V[3][3];
	quat_to_rot(raw->quats + i * 4u, R);
	const double s2[3] = {(double)s[0] * s[0], (double)s[1] * s[1], (double)s[2] * s[2]};
	for (int r = 0; r < 3; r++) {
		for (int q = r; q < 3; q++) {
			const double v =
			    R[r][0] * s2[0] * R[q][0] + R[r][1] * s2[1] * R[q][1] + R[r][2] * s2[2] * R[q][2];
			C[r][q] = C[q][r] = v * S[r] * S[q];
		}
	}
	sym_eig3(C, ev, V);
	for (int k = 0; k < 3; k++) {
		o[7 + k] = (float)(0.5 * log(ev[k] > 1e-14 ? ev[k] : 1e-14));
	}
	rot_to_quat(V, o + 10);
}

uint64_t
oc_splat_ply_size(size_t count)
{
	char header[1024];
	int n = snprintf(header, sizeof(header), "ply\nformat binary_little_endian 1.0\nelement vertex %zu\n", count);
	for (uint32_t k = 0; k < OC_SPLAT_FLOATS; k++) {
		n += (int)strlen("property float \n") + (int)strlen(k_props[k]);
	}
	n += (int)strlen("end_header\n");
	return (uint64_t)n + (uint64_t)count * OC_SPLAT_FLOATS * sizeof(float);
}

//! One index into the clump crop, ordered by z (ties by position: a stable sort).
struct zi
{
	double z;
	size_t i;
};

static int
cmp_zi(const void *pa, const void *pb)
{
	const struct zi *a = (const struct zi *)pa, *b = (const struct zi *)pb;
	if (a->z != b->z) {
		return a->z < b->z ? -1 : 1;
	}
	return a->i < b->i ? -1 : (a->i > b->i ? 1 : 0);
}

//! numpy-style weighted percentile: the first z (ascending) whose cumulative weight reaches p·total.
static double
weighted_pct(const struct zi *sorted, const double *weight, size_t n, double p)
{
	double total = 0.0;
	for (size_t k = 0; k < n; k++) {
		total += weight[sorted[k].i];
	}
	const double target = p * total;
	double cw = 0.0;
	for (size_t k = 0; k < n; k++) {
		cw += weight[sorted[k].i];
		if (cw >= target) {
			return sorted[k].z;
		}
	}
	return sorted[n - 1].z;
}

bool
oc_splat_nearest_clump(
    const float *rows, size_t count, double f_px, double width, double height, struct oc_splat_focus *out)
{
	memset(out, 0, sizeof(*out));
	if (rows == NULL || count == 0 || !(f_px > 0.0) || !(width > 0.0) || !(height > 0.0)) {
		return false;
	}
	const double cx = width / 2.0, cy = height / 2.0, half = OC_CLUMP_CENTRAL_FRAC / 2.0;
	const size_t stride = count > OC_CLUMP_SAMPLE_CAP ? (count + OC_CLUMP_SAMPLE_CAP - 1) / OC_CLUMP_SAMPLE_CAP : 1;
	const size_t cap = (count + stride - 1) / stride;
	double *invz = (double *)malloc(sizeof(double) * cap);
	double *weight = (double *)malloc(sizeof(double) * cap);
	double *zc = (double *)malloc(sizeof(double) * cap);
	int *bin_of = (int *)malloc(sizeof(int) * cap);
	struct zi *sorted = (struct zi *)malloc(sizeof(struct zi) * cap);
	bool ok = false;
	if (invz == NULL || weight == NULL || zc == NULL || bin_of == NULL || sorted == NULL) {
		goto done;
	}
	size_t n = 0;
	for (size_t i = 0; i < count; i += stride) {
		const float *r = rows + i * OC_SPLAT_FLOATS;
		const double x = r[0], y = r[1], z = r[2];
		const double op = 1.0 / (1.0 + exp(-(double)r[6])); // the activated opacity
		if (op < OC_CLUMP_MIN_OPACITY || !(z > OC_CLUMP_MIN_Z)) {
			continue;
		}
		const double u = (f_px * (x / z) + cx) / width - 0.5;
		const double v = (f_px * (y / z) + cy) / height - 0.5;
		if (fabs(u) > half || fabs(v) > half) {
			continue;
		}
		zc[n] = z;
		invz[n] = 1.0 / z;
		weight[n] = op;
		n++;
	}
	if (n == 0) {
		goto done;
	}
	double lo = invz[0], hi = invz[0];
	for (size_t k = 1; k < n; k++) {
		lo = invz[k] < lo ? invz[k] : lo;
		hi = invz[k] > hi ? invz[k] : hi;
	}
	if (!(hi > lo)) {
		goto done;
	}
	const int nb = OC_CLUMP_N_BINS;
	double bins[OC_CLUMP_N_BINS] = {0}, smooth[OC_CLUMP_N_BINS];
	for (size_t k = 0; k < n; k++) {
		int b = (int)floor((invz[k] - lo) / (hi - lo) * nb);
		b = b > nb - 1 ? nb - 1 : b;
		bin_of[k] = b;
		bins[b] += weight[k];
	}
	double mass_total = 0.0;
	for (int b = 0; b < nb; b++) {
		mass_total += bins[b];
	}
	if (!(mass_total > 0.0)) {
		goto done;
	}
	for (int b = 0; b < nb; b++) {
		const double left = bins[b > 0 ? b - 1 : 0], right = bins[b < nb - 1 ? b + 1 : nb - 1];
		smooth[b] = (left + bins[b] + right) / 3.0;
	}
	const double empty = mass_total * OC_CLUMP_NOISE_FLOOR_FRAC;
	int i = nb - 1;
	while (i >= 0) {
		if (smooth[i] <= empty) {
			i--;
			continue;
		}
		int j = i;
		double run = 0.0;
		while (j >= 0 && smooth[j] > empty) {
			run += bins[j];
			j--;
		}
		if (run >= mass_total * OC_CLUMP_MIN_MASS_FRAC) {
			double w_sum = 0.0, v_sum = 0.0;
			for (size_t k = 0; k < n; k++) {
				if (bin_of[k] <= i && bin_of[k] >= j + 1) {
					w_sum += weight[k];
					v_sum += weight[k] * invz[k];
				}
			}
			const double centroid = w_sum > 0.0 ? v_sum / w_sum : (lo + hi) / 2.0;
			if (!(centroid > 0.0)) {
				goto done;
			}
			double z = 1.0 / centroid;
			z = z < OC_CLUMP_PIVOT_MIN_M ? OC_CLUMP_PIVOT_MIN_M
			                             : (z > OC_CLUMP_PIVOT_MAX_M ? OC_CLUMP_PIVOT_MAX_M : z);
			for (size_t k = 0; k < n; k++) {
				sorted[k].z = zc[k];
				sorted[k].i = k;
			}
			qsort(sorted, n, sizeof(*sorted), cmp_zi);
			out->valid = true;
			out->z_m = (float)z;
			out->mass_frac = (float)(run / mass_total);
			out->near_m = (float)weighted_pct(sorted, weight, n, 0.05);
			out->far_m = (float)weighted_pct(sorted, weight, n, 0.95);
			out->n = (uint32_t)n;
			ok = true;
			goto done;
		}
		i = j;
	}
done:
	free(invz);
	free(weight);
	free(zc);
	free(bin_of);
	free(sorted);
	return ok;
}

bool
oc_splat_encode(const struct oc_splat_raw *raw,
                uint32_t w,
                uint32_t h,
                float focal_px,
                float *rows,
                bool (*is_cancelled)(void *ctx),
                void *cancel_ctx,
                struct oc_splat_stats *stats)
{
	memset(stats, 0, sizeof(*stats));
	if (raw == NULL || rows == NULL || raw->count == 0 || w == 0 || h == 0 || !(focal_px > 0.0f)) {
		return false;
	}
	// inv(NDC · K) for K = [f 0 w/2; 0 f h/2; 0 0 1]: a diagonal scale.
	const double S[3] = {(double)w / (2.0 * focal_px), (double)h / (2.0 * focal_px), 1.0};
	stats->z_min = INFINITY;
	stats->z_max = -INFINITY;
	for (size_t base = 0; base < raw->count; base += OC_SPLAT_CHUNK) {
		if (is_cancelled != NULL && is_cancelled(cancel_ctx)) {
			return false;
		}
		const size_t end = base + OC_SPLAT_CHUNK < raw->count ? base + OC_SPLAT_CHUNK : raw->count;
		for (size_t i = base; i < end; i++) {
			float *o = rows + i * OC_SPLAT_FLOATS;
			if (gaussian_finite(raw, i)) {
				encode(raw, i, S, o);
				if (o[2] > 0.0f) {
					stats->z_min = o[2] < stats->z_min ? o[2] : stats->z_min;
					stats->z_max = o[2] > stats->z_max ? o[2] : stats->z_max;
				}
			} else {
				// Kept (the count is a fact of the model's grid) but invisible.
				memset(o, 0, sizeof(float) * OC_SPLAT_FLOATS);
				o[6] = -13.8f; // logit(1e-6)
				o[7] = o[8] = o[9] = -16.0f;
				o[10] = 1.0f;
				stats->non_finite++;
			}
		}
	}
	if (!(stats->z_min <= stats->z_max)) {
		stats->z_min = stats->z_max = 0.0f;
	}
	stats->written = raw->count;
	oc_splat_nearest_clump(rows, raw->count, focal_px, w, h, &stats->focus);
	return true;
}

bool
oc_splat_write_ply_rows(const float *rows,
                        size_t count,
                        oc_splat_write_fn write,
                        void *write_ctx,
                        bool (*is_cancelled)(void *ctx),
                        void *cancel_ctx)
{
	char header[1024];
	int n = snprintf(header, sizeof(header), "ply\nformat binary_little_endian 1.0\nelement vertex %zu\n", count);
	for (uint32_t k = 0; k < OC_SPLAT_FLOATS; k++) {
		n += snprintf(header + n, sizeof(header) - (size_t)n, "property float %s\n", k_props[k]);
	}
	n += snprintf(header + n, sizeof(header) - (size_t)n, "end_header\n");
	if (!write(write_ctx, header, (uint64_t)n)) {
		return false;
	}
	for (size_t base = 0; base < count; base += OC_SPLAT_CHUNK) {
		if (is_cancelled != NULL && is_cancelled(cancel_ctx)) {
			return false;
		}
		const size_t end = base + OC_SPLAT_CHUNK < count ? base + OC_SPLAT_CHUNK : count;
		if (!write(write_ctx, rows + base * OC_SPLAT_FLOATS,
		           (uint64_t)(end - base) * OC_SPLAT_FLOATS * sizeof(float))) {
			return false;
		}
	}
	return true;
}
