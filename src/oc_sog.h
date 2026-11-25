// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  The splat capability's SOG writer: 3DGS rows (oc_splat_encode) as a
 *         PlayCanvas SOG **version 2** file (a stored ZIP of `meta.json` and
 *         lossless WebP planes), degree 0, with the mono `camera` block of
 *         docs/sog-camera-block.md.
 *
 * Planes (one texel per Gaussian, Morton order, width = ceil(sqrt(N) / 4) · 4,
 * height = ceil(N / width / 4) · 4, both multiples of 4, padding texels 0):
 * - means_l / means_u: per axis q = round(65535 · (L(v) − min) / (max − min)),
 *   L(v) = sign(v) · ln(|v| + 1), low / high byte in R G B (A = 255); meta
 *   `means.mins` / `means.maxs` = L of each axis's range.
 * - quats: smallest three of the normalised (w, x, y, z), the largest
 *   component made positive, the other three × √2 as round(255 · (c / 2 + 1/2))
 *   in R G B, A = 252 + the largest component's index.
 * - scales: the pooled log scales quantised to a 256-entry ascending codebook
 *   (1-D k-means), labels in R G B; meta `scales.codebook`.
 * - sh0: the pooled f_dc values, same codebook kind, labels in R G B,
 *   A = round(255 · sigmoid(opacity)); meta `sh0.codebook`.
 *
 * Deterministic: same rows, same bytes (fixed ZIP timestamps, no threads).
 */

#pragma once

#include "oc_splat.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! What the camera block says (docs/sog-camera-block.md, mono variant). */
struct oc_sog_camera
{
	float fx, fy, cx, cy;        //!< the intrinsics the Gaussians were unprojected through, input-image px
	uint32_t width, height;      //!< the input image
	struct oc_splat_focus focus; //!< omitted from the block when !valid
};

/*! The camera block as one line of JSON (no newline) into @p buf; the length
 *  written, or 0 when it does not fit. The same text oc_sog_write puts into
 *  meta.json after `count`. */
size_t
oc_sog_camera_json(const struct oc_sog_camera *cam, char *buf, size_t cap);

/*! A whole .sog in memory. */
struct oc_sog_file
{
	uint8_t *data;
	size_t size;
};

/*!
 * Encode @p count rows (OC_SPLAT_FLOATS each, OpenCV camera space) as a SOG v2
 * file with @p cam's camera block (NULL = no block) into @p out (free with
 * oc_sog_free). @p generator names the writer in `asset.generator`. Polls
 * @p is_cancelled between planes. false = cancelled, out of memory or an
 * encoder failure (@p err says which).
 */
bool
oc_sog_write(const float *rows,
             size_t count,
             const struct oc_sog_camera *cam,
             const char *generator,
             bool (*is_cancelled)(void *ctx),
             void *cancel_ctx,
             struct oc_sog_file *out,
             char *err,
             size_t err_size);

void
oc_sog_free(struct oc_sog_file *f);

#ifdef __cplusplus
}
#endif
