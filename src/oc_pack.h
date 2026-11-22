// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Model-pack manifest, schema 2 (discovery spec §5): the parts the
 *         provider and the fake host need.
 *
 * The provider reads its pack for model identity, I/O contract and licence;
 * it never opens a blob path from here. Blob bytes are resolved only through
 * host->model_resolve / host->model_request by the ref
 * "<pack>/<model>/<variant>" (oc_pack_ref).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OC_PACK_MAX_MODELS 16
#define OC_PACK_MAX_VARIANTS 4
#define OC_PACK_MAX_EPS 4
#define OC_PACK_MAX_DEFAULTS 8
#define OC_PACK_MAX_DEPS 4
#define OC_PACK_MAX_DATA_FILES 3 //!< files beside the model file (role "data": ONNX external weights)

struct oc_pack_file
{
	char role[16];
	char format[16];
	char sha256[65]; //!< lower-case hex
	uint64_t size;
};

struct oc_pack_variant
{
	char id[32];
	char backend[32];
	char ep[OC_PACK_MAX_EPS][16]; //!< "dml", "cpu", … in preference order
	uint32_t ep_count;
	char precision[24];
	struct oc_pack_file file; //!< the one "model" file
	/*! The variant's other files (role "data"). The provider never opens them:
	 *  the store hands out a variant only when every file is present and
	 *  verified, and lays them out side by side (blobs/<sha256>), so a model
	 *  that names its external weights by their sha256 finds them. The fake
	 *  host stores them. */
	struct oc_pack_file data[OC_PACK_MAX_DATA_FILES];
	uint32_t data_count;
};

struct oc_pack_model
{
	char name[64];
	char version[32];
	char task[32]; //!< "depth-mono", "depth-video", "aux:inpaint", …

	/* io.input */
	bool fixed;           //!< "fixed-WxH" (else "dynamic-multiple-of-N")
	uint32_t width;       //!< fixed only
	uint32_t height;      //!< fixed only
	uint32_t multiple_of; //!< dynamic only
	uint32_t native_long_edge;
	bool normalize;
	float mean[3], std[3];

	/* io.output */
	char depth_kind[32];   //!< "affine-disparity", … (depth tasks)
	char measure[16];      //!< "planar" / "ray"
	char coords[16];       //!< splat tasks: "opencv" (+x right, +y down, +z forward)
	char splat_format[32]; //!< splat tasks: the raw output's layout, "sharp-projective" (oc_splat.h)
	char color_space[16];  //!< splat tasks: "linear" (default) / "srgb": the colors output's encoding
	uint32_t splat_count;  //!< splat tasks: Gaussians per result (io.output.count); 0 = undeclared

	/* io.tensors */
	char t_image[64], t_depth[64], t_cache_in[64], t_cache_out[64], t_is_first[64];
	/* io.tensors, splat tasks: the focal input and the five Gaussian outputs. */
	char t_disparity_factor[64]; //!< [1] float = focal_px / input width
	char t_means[64], t_scales[64], t_quats[64], t_colors[64], t_opacities[64];

	/* io.state (temporal models) */
	int64_t cache_shape[4];
	uint32_t cache_rank;

	/* license: commercial is the AND over the model and every dependency. */
	char spdx[48];
	bool commercial;

	struct oc_pack_variant variants[OC_PACK_MAX_VARIANTS];
	uint32_t variant_count;
	bool prefetch;
};

struct oc_pack_default
{
	char task[32];
	char low[64], medium[64], high[64];
};

struct oc_pack
{
	char pack[64];
	char blob_base_url[512];
	bool commercial; //!< the pack's own flag (must equal the AND over its models)
	struct oc_pack_model models[OC_PACK_MAX_MODELS];
	uint32_t model_count;
	struct oc_pack_default defaults[OC_PACK_MAX_DEFAULTS];
	uint32_t default_count;
};

/*! Parse a schema-2 manifest. false + @p err on any schema violation. */
bool
oc_pack_load(const char *path, struct oc_pack *out, char *err, size_t err_size);

const struct oc_pack_model *
oc_pack_find(const struct oc_pack *pack, const char *name);

/*! The model `defaults[task][quality]` names ("low" / "medium" / "high"). */
const struct oc_pack_model *
oc_pack_default_model(const struct oc_pack *pack, const char *task, const char *quality);

/*! First variant whose ep list holds @p ep, or NULL. */
const struct oc_pack_variant *
oc_pack_variant_for_ep(const struct oc_pack_model *m, const char *ep);

/*! "<pack>/<model>/<variant>" — the ref the host resolves. */
void
oc_pack_ref(const struct oc_pack *pack,
            const struct oc_pack_model *m,
            const struct oc_pack_variant *v,
            char *buf,
            size_t buf_size);

/*! True when @p sha is 64 lower-case hex characters. */
bool
oc_pack_is_sha256(const char *sha);

#ifdef __cplusplus
}
#endif
