// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  CPU pre- and post-processing: colour image → model tensor, model map
 *         → the runtime's depth target, and the per-frame range statistics.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! Fit @p w × @p h into a long edge of @p long_edge, keeping the aspect, each
 *  side rounded to a multiple of @p multiple (≥ 1) and at least one multiple.
 *  Never upscales past the input when @p no_upscale. */
void
oc_fit_long_edge(
    uint32_t w, uint32_t h, uint32_t long_edge, uint32_t multiple, bool no_upscale, uint32_t *out_w, uint32_t *out_h);

/*! RGBA8 / BGRA8 (@p bgr: B first) → planar RGB float [0,1] of
 *  @p dw × @p dh (bilinear, texel centres), optionally normalised per channel
 *  ((v - mean) / std). @p chw holds 3·dw·dh floats. */
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
            float *chw);

/*! Bilinear resample of a float map into a strided target. */
void
oc_resample_map(const float *src, uint32_t sw, uint32_t sh, float *dst, uint32_t dw, uint32_t dh, uint32_t dst_stride);

/*! min / max / 5th / 95th percentile of a strided float map (finite samples
 *  only). false when there is no finite sample. */
bool
oc_map_stats(const float *map,
             uint32_t w,
             uint32_t h,
             uint32_t stride,
             float *out_min,
             float *out_max,
             float *out_p05,
             float *out_p95);

#ifdef __cplusplus
}
#endif
