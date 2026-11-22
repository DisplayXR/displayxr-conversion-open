// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  The splat capability's CPU work around the model: the colour image
 *         resampled into the model's fixed grid, and the model's Gaussians
 *         (in its projective frame) unprojected through the input's
 *         intrinsics and written as a 3DGS PLY.
 *
 * The model contract (a pack entry with task "splats-mono" and io.output
 * splat_format "sharp-projective"): one [1,3,H,W] float image in [0,1] (sRGB
 * values, the whole picture resampled to the grid, aspect not kept), one [1]
 * float disparity factor = focal_px / input width, and five outputs for N
 * Gaussians: means [1,N,3] in the projective frame (x_ndc·z, y_ndc·z, z;
 * NDC -1..1 left→right / top→bottom, the principal point at the centre),
 * scales [1,N,3] (standard deviations), rotations [1,N,4] (w, x, y, z, not
 * normalised), colors [1,N,3] (linear RGB unless the pack says srgb) and
 * opacities [1,N] (after the sigmoid).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! RGBA8 / BGRA8 (@p bgr: B first) of @p sw × @p sh → planar RGB float in
 *  [0,1] of @p dw × @p dh, separable triangle filter whose support widens with
 *  the downscale factor (an antialiased bilinear resample, so a 6000-px photo
 *  does not alias into the 1536-px grid). @p chw holds 3·dw·dh floats.
 *  false = out of memory. */
bool
oc_resample_rgb_aa(
    const uint8_t *src, uint32_t stride, uint32_t sw, uint32_t sh, bool bgr, uint32_t dw, uint32_t dh, float *chw);

/*! The model's raw outputs for N Gaussians. */
struct oc_splat_raw
{
	const float *means;     //!< N × 3, projective frame (x_ndc·z, y_ndc·z, z)
	const float *scales;    //!< N × 3
	const float *quats;     //!< N × 4, w x y z
	const float *colors;    //!< N × 3, linear RGB
	const float *opacities; //!< N
	size_t count;
	bool colors_linear; //!< true: linear RGB (converted to sRGB for the PLY); false: already sRGB
};

/*! What the writer measured on the unprojected cloud (the splat's own units). */
struct oc_splat_stats
{
	size_t written;
	size_t non_finite; //!< Gaussians with a non-finite value, written as transparent at the origin
	float z_min, z_max;
	//! Focus SUGGESTION: the nearest substantial clump of the frame's central
	//! region (opacity-weighted 10th percentile of depth), and the mean
	//! position of the Gaussians around it. valid = false when there was none.
	bool focus_valid;
	float focus_point[3];
};

/*! Byte sink: false = stop. */
typedef bool (*oc_splat_write_fn)(void *ctx, const void *bytes, uint64_t count);

/*! The PLY's size in bytes for @p count Gaussians (header + body). */
uint64_t
oc_splat_ply_size(size_t count);

/*!
 * Unproject @p raw through the intrinsics of a @p w × @p h input of focal
 * @p focal_px (principal point at the centre: D = diag(w / 2f, h / 2f, 1),
 * mean' = D·mean, so x_cam = x_ndc·z·w / 2f — OpenCV axes, +x right, +y
 * down, +z forward), re-diagonalise each covariance (the unprojection scales x and y
 * differently), and write a binary little-endian 3DGS PLY: x y z, f_dc_0..2
 * (sRGB colour as a degree-0 spherical harmonic), opacity (logit),
 * scale_0..2 (log), rot_0..3 (w x y z). Polls @p is_cancelled between chunks.
 * false = the sink stopped, cancelled, or out of memory.
 */
bool
oc_splat_write_ply(const struct oc_splat_raw *raw,
                   uint32_t w,
                   uint32_t h,
                   float focal_px,
                   oc_splat_write_fn write,
                   void *write_ctx,
                   bool (*is_cancelled)(void *ctx),
                   void *cancel_ctx,
                   struct oc_splat_stats *stats);

#ifdef __cplusplus
}
#endif
