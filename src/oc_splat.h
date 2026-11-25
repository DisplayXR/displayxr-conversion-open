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

#define OC_SPLAT_FLOATS 14u //!< floats per Gaussian row: x y z f_dc_0..2 opacity scale_0..2 rot_0..3

/*! The focus SUGGESTION of a mono splat: the cloud's nearest substantial clump
 *  (oc_splat_nearest_clump; docs/sog-camera-block.md). */
struct oc_splat_focus
{
	bool valid;
	float z_m;       //!< the clump's distance on the optical axis, clamped to [0.1, 20]
	float near_m;    //!< opacity-weighted 5th percentile of z over the central crop
	float far_m;     //!< opacity-weighted 95th percentile
	float mass_frac; //!< the clump's share of the crop's opacity mass
	uint32_t n;      //!< Gaussians in the crop
};

/*! What the encoder measured on the unprojected cloud (the splat's own units). */
struct oc_splat_stats
{
	size_t written;
	size_t non_finite; //!< Gaussians with a non-finite value, written as transparent at the origin
	float z_min, z_max;
	struct oc_splat_focus focus;
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
 * differently), and write one 3DGS row per Gaussian into @p rows
 * (count × OC_SPLAT_FLOATS): x y z, f_dc_0..2 (sRGB colour as a degree-0
 * spherical harmonic), opacity (logit), scale_0..2 (log), rot_0..3 (w x y z).
 * A Gaussian with a non-finite value is kept, transparent at the origin.
 * Measures the depth range and the focus suggestion (oc_splat_nearest_clump,
 * on the rows). Polls @p is_cancelled between chunks. false = cancelled or bad input.
 */
bool
oc_splat_encode(const struct oc_splat_raw *raw,
                uint32_t w,
                uint32_t h,
                float focal_px,
                float *rows,
                bool (*is_cancelled)(void *ctx),
                void *cancel_ctx,
                struct oc_splat_stats *stats);

/*! @p rows (oc_splat_encode) as a binary little-endian 3DGS PLY into @p write.
 *  false = the sink stopped or cancelled. */
bool
oc_splat_write_ply_rows(const float *rows,
                        size_t count,
                        oc_splat_write_fn write,
                        void *write_ctx,
                        bool (*is_cancelled)(void *ctx),
                        void *cancel_ctx);

/*!
 * The nearest substantial clump of a mono splat in the central half of the
 * frame, the DisplayXR Gallery's online SOG generator's rule, ported exactly
 * (docs/sog-camera-block.md): @p rows are 3DGS rows in OpenCV camera space
 * (x y z at 0..2, the opacity LOGIT at 6), @p f_px the focal they were
 * unprojected through and @p width × @p height the input image (principal
 * point at the centre). Stride-sample to ≤ 40 000, drop opacity < 0.05 and
 * z ≤ 0.05, keep |u|, |v| ≤ 0.25, opacity-weighted histogram of 1/z (120 bins),
 * 3-bin smoothing, the first run from the near end holding ≥ 3 % of the mass,
 * its weighted centroid of 1/z inverted and clamped to [0.1, 20]. false (and
 * out->valid = false) when there is no clump.
 */
bool
oc_splat_nearest_clump(
    const float *rows, size_t count, double f_px, double width, double height, struct oc_splat_focus *out);

#ifdef __cplusplus
}
#endif
