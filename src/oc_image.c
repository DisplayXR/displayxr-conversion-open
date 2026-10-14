// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  CPU pre- and post-processing, see oc_image.h.
 */

#include "oc_image.h"

#include <math.h>
#include <string.h>

void
oc_fit_long_edge(
    uint32_t w, uint32_t h, uint32_t long_edge, uint32_t multiple, bool no_upscale, uint32_t *out_w, uint32_t *out_h)
{
	if (multiple == 0) {
		multiple = 1;
	}
	if (w == 0 || h == 0 || long_edge == 0) {
		*out_w = multiple;
		*out_h = multiple;
		return;
	}
	const uint32_t in_long = w > h ? w : h;
	if (no_upscale && long_edge > in_long) {
		long_edge = in_long;
	}
	const double scale = (double)long_edge / (double)in_long;
	uint32_t ow = (uint32_t)floor((double)w * scale / multiple + 0.5) * multiple;
	uint32_t oh = (uint32_t)floor((double)h * scale / multiple + 0.5) * multiple;
	*out_w = ow < multiple ? multiple : ow;
	*out_h = oh < multiple ? multiple : oh;
}

static inline float
lerp(float a, float b, float t)
{
	return a + (b - a) * t;
}

void
oc_prep_rgb(const uint8_t *src,
            uint32_t stride,
            uint32_t sw,
            uint32_t sh,
            bool bgr,
            uint32_t dw,
            uint32_t dh,
            const float *mean,
            const float *std,
            float *chw)
{
	const size_t plane = (size_t)dw * dh;
	const float sx = (float)sw / (float)dw;
	const float sy = (float)sh / (float)dh;
	const int ri = bgr ? 2 : 0, bi = bgr ? 0 : 2;
	for (uint32_t y = 0; y < dh; y++) {
		float fy = ((float)y + 0.5f) * sy - 0.5f;
		if (fy < 0.0f) {
			fy = 0.0f;
		}
		uint32_t y0 = (uint32_t)fy;
		if (y0 > sh - 1) {
			y0 = sh - 1;
		}
		const uint32_t y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
		const float ty = fy - (float)y0;
		const uint8_t *r0 = src + (size_t)y0 * stride;
		const uint8_t *r1 = src + (size_t)y1 * stride;
		for (uint32_t x = 0; x < dw; x++) {
			float fx = ((float)x + 0.5f) * sx - 0.5f;
			if (fx < 0.0f) {
				fx = 0.0f;
			}
			uint32_t x0 = (uint32_t)fx;
			if (x0 > sw - 1) {
				x0 = sw - 1;
			}
			const uint32_t x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
			const float tx = fx - (float)x0;
			const int ch_index[3] = {ri, 1, bi};
			for (int c = 0; c < 3; c++) {
				const int k = ch_index[c];
				const float a = lerp(r0[x0 * 4 + k], r0[x1 * 4 + k], tx);
				const float b = lerp(r1[x0 * 4 + k], r1[x1 * 4 + k], tx);
				float v = lerp(a, b, ty) * (1.0f / 255.0f);
				if (mean != NULL && std != NULL) {
					v = (v - mean[c]) / std[c];
				}
				chw[(size_t)c * plane + (size_t)y * dw + x] = v;
			}
		}
	}
}

void
oc_resample_map(const float *src, uint32_t sw, uint32_t sh, float *dst, uint32_t dw, uint32_t dh, uint32_t dst_stride)
{
	const float sx = (float)sw / (float)dw;
	const float sy = (float)sh / (float)dh;
	for (uint32_t y = 0; y < dh; y++) {
		float fy = ((float)y + 0.5f) * sy - 0.5f;
		if (fy < 0.0f) {
			fy = 0.0f;
		}
		uint32_t y0 = (uint32_t)fy;
		if (y0 > sh - 1) {
			y0 = sh - 1;
		}
		const uint32_t y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
		const float ty = fy - (float)y0;
		float *row = (float *)((uint8_t *)dst + (size_t)y * dst_stride);
		for (uint32_t x = 0; x < dw; x++) {
			float fx = ((float)x + 0.5f) * sx - 0.5f;
			if (fx < 0.0f) {
				fx = 0.0f;
			}
			uint32_t x0 = (uint32_t)fx;
			if (x0 > sw - 1) {
				x0 = sw - 1;
			}
			const uint32_t x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
			const float tx = fx - (float)x0;
			const float a = lerp(src[(size_t)y0 * sw + x0], src[(size_t)y0 * sw + x1], tx);
			const float b = lerp(src[(size_t)y1 * sw + x0], src[(size_t)y1 * sw + x1], tx);
			row[x] = lerp(a, b, ty);
		}
	}
}

#define OC_HIST_BINS 1024

bool
oc_map_stats(const float *map,
             uint32_t w,
             uint32_t h,
             uint32_t stride,
             float *out_min,
             float *out_max,
             float *out_p05,
             float *out_p95)
{
	float lo = INFINITY, hi = -INFINITY;
	size_t n = 0;
	for (uint32_t y = 0; y < h; y++) {
		const float *row = (const float *)((const uint8_t *)map + (size_t)y * stride);
		for (uint32_t x = 0; x < w; x++) {
			const float v = row[x];
			if (isfinite(v)) {
				lo = v < lo ? v : lo;
				hi = v > hi ? v : hi;
				n++;
			}
		}
	}
	if (n == 0) {
		return false;
	}
	*out_min = lo;
	*out_max = hi;
	if (!(hi > lo)) {
		*out_p05 = lo;
		*out_p95 = hi;
		return true;
	}
	// Percentiles from a fixed histogram: allocation-free, one bin of error.
	uint32_t hist[OC_HIST_BINS];
	memset(hist, 0, sizeof(hist));
	const float scale = (float)OC_HIST_BINS / (hi - lo);
	for (uint32_t y = 0; y < h; y++) {
		const float *row = (const float *)((const uint8_t *)map + (size_t)y * stride);
		for (uint32_t x = 0; x < w; x++) {
			const float v = row[x];
			if (isfinite(v)) {
				int b = (int)((v - lo) * scale);
				b = b < 0 ? 0 : (b >= OC_HIST_BINS ? OC_HIST_BINS - 1 : b);
				hist[b]++;
			}
		}
	}
	const size_t t05 = (size_t)((double)n * 0.05), t95 = (size_t)((double)n * 0.95);
	size_t acc = 0;
	bool got05 = false;
	*out_p95 = hi;
	for (int b = 0; b < OC_HIST_BINS; b++) {
		acc += hist[b];
		if (!got05 && acc > t05) {
			*out_p05 = lo + ((float)b + 0.5f) / scale;
			got05 = true;
		}
		if (acc > t95) {
			*out_p95 = lo + ((float)b + 0.5f) / scale;
			break;
		}
	}
	if (!got05) {
		*out_p05 = lo;
	}
	return true;
}
