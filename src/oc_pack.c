// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Model-pack manifest reader, see oc_pack.h.
 */

#include "oc_pack.h"
#include "oc_json.h"

#include <stdio.h>
#include <string.h>

static void
copy_str(char *dst, size_t cap, const char *src)
{
	if (cap == 0) {
		return;
	}
	if (src == NULL) {
		dst[0] = '\0';
		return;
	}
	size_t n = strlen(src);
	if (n >= cap) {
		n = cap - 1;
	}
	memcpy(dst, src, n);
	dst[n] = '\0';
}

bool
oc_pack_is_sha256(const char *sha)
{
	if (sha == NULL || strlen(sha) != 64) {
		return false;
	}
	for (int i = 0; i < 64; i++) {
		const char c = sha[i];
		if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
			return false;
		}
	}
	return true;
}

#define BAD(...)                                                                                                       \
	do {                                                                                                           \
		if (err != NULL && err_size > 0) {                                                                     \
			snprintf(err, err_size, __VA_ARGS__);                                                          \
		}                                                                                                      \
		return false;                                                                                          \
	} while (0)

static bool
parse_variant(const struct oc_json *jv, struct oc_pack_variant *v, const char *model, char *err, size_t err_size)
{
	copy_str(v->id, sizeof(v->id), oc_json_str(jv, "id", ""));
	copy_str(v->backend, sizeof(v->backend), oc_json_str(jv, "backend", ""));
	copy_str(v->precision, sizeof(v->precision), oc_json_str(jv, "precision", ""));
	if (v->id[0] == '\0') {
		BAD("model %s: a variant has no id", model);
	}
	const struct oc_json *eps = oc_json_get(jv, "ep");
	for (const struct oc_json *e = eps != NULL ? eps->child : NULL; e != NULL; e = e->next) {
		if (e->type == OC_JSON_STRING && v->ep_count < OC_PACK_MAX_EPS) {
			copy_str(v->ep[v->ep_count++], sizeof(v->ep[0]), e->string);
		}
	}
	const struct oc_json *files = oc_json_get(jv, "files");
	uint32_t models = 0;
	for (const struct oc_json *f = files != NULL ? files->child : NULL; f != NULL; f = f->next) {
		const char *sha = oc_json_str(f, "sha256", NULL);
		const double size = oc_json_num(f, "size", -1.0);
		if (!oc_pack_is_sha256(sha) || size <= 0.0) {
			BAD("model %s/%s: a file has no valid sha256 + size", model, v->id);
		}
		if (strcmp(oc_json_str(f, "role", ""), "model") == 0) {
			copy_str(v->file.role, sizeof(v->file.role), "model");
			copy_str(v->file.format, sizeof(v->file.format), oc_json_str(f, "format", "onnx"));
			copy_str(v->file.sha256, sizeof(v->file.sha256), sha);
			v->file.size = (uint64_t)size;
			models++;
		}
	}
	if (models != 1) {
		BAD("model %s/%s: expected exactly one file with role \"model\" (%u)", model, v->id, models);
	}
	return true;
}

static bool
parse_model(const struct oc_json *jm, struct oc_pack_model *m, char *err, size_t err_size)
{
	copy_str(m->name, sizeof(m->name), oc_json_str(jm, "name", ""));
	copy_str(m->version, sizeof(m->version), oc_json_str(jm, "version", ""));
	copy_str(m->task, sizeof(m->task), oc_json_str(jm, "task", ""));
	if (m->name[0] == '\0' || m->version[0] == '\0' || m->task[0] == '\0') {
		BAD("a model lacks name / version / task");
	}
	m->prefetch = oc_json_bool(jm, "prefetch") == 1;

	const struct oc_json *io = oc_json_get(jm, "io");
	const struct oc_json *in = oc_json_get(io, "input");
	const struct oc_json *out = oc_json_get(io, "output");
	const struct oc_json *t = oc_json_get(io, "tensors");
	const struct oc_json *st = oc_json_get(io, "state");
	const char *res = oc_json_str(in, "resolution", "");
	unsigned a = 0, b = 0;
	if (sscanf(res, "fixed-%ux%u", &a, &b) == 2 && a > 0 && b > 0) {
		m->fixed = true;
		m->width = a;
		m->height = b;
	} else if (sscanf(res, "dynamic-multiple-of-%u", &a) == 1 && a > 0) {
		m->fixed = false;
		m->multiple_of = a;
	} else {
		BAD("model %s: io.input.resolution '%s' is neither fixed-WxH nor dynamic-multiple-of-N", m->name, res);
	}
	m->native_long_edge = (uint32_t)oc_json_num(in, "native_long_edge", 0.0);
	const struct oc_json *norm = oc_json_get(in, "normalize");
	if (norm != NULL && norm->type == OC_JSON_OBJECT) {
		const struct oc_json *mean = oc_json_get(norm, "mean");
		const struct oc_json *std = oc_json_get(norm, "std");
		for (size_t i = 0; i < 3; i++) {
			const struct oc_json *mv = oc_json_at(mean, i);
			const struct oc_json *sv = oc_json_at(std, i);
			if (mv == NULL || sv == NULL || mv->type != OC_JSON_NUMBER || sv->type != OC_JSON_NUMBER ||
			    sv->number <= 0.0) {
				BAD("model %s: io.input.normalize needs 3 means and 3 positive stds", m->name);
			}
			m->mean[i] = (float)mv->number;
			m->std[i] = (float)sv->number;
		}
		m->normalize = true;
	}
	copy_str(m->depth_kind, sizeof(m->depth_kind), oc_json_str(out, "depth_kind", ""));
	copy_str(m->measure, sizeof(m->measure), oc_json_str(out, "measure", "planar"));
	copy_str(m->t_image, sizeof(m->t_image), oc_json_str(t, "image", ""));
	copy_str(m->t_depth, sizeof(m->t_depth), oc_json_str(t, "depth", ""));
	copy_str(m->t_cache_in, sizeof(m->t_cache_in), oc_json_str(t, "cache_in", ""));
	copy_str(m->t_cache_out, sizeof(m->t_cache_out), oc_json_str(t, "cache_out", ""));
	copy_str(m->t_is_first, sizeof(m->t_is_first), oc_json_str(t, "is_first", ""));
	const struct oc_json *cs = oc_json_get(st, "cache_shape");
	for (const struct oc_json *d = cs != NULL ? cs->child : NULL; d != NULL; d = d->next) {
		if (d->type != OC_JSON_NUMBER || d->number < 1.0 || m->cache_rank >= 4) {
			BAD("model %s: io.state.cache_shape is not a list of up to 4 positive dims", m->name);
		}
		m->cache_shape[m->cache_rank++] = (int64_t)d->number;
	}

	// Licence: commercial = AND over the model and every dependency; a missing
	// flag anywhere is false (discovery spec §5.3).
	const struct oc_json *lic = oc_json_get(jm, "license");
	copy_str(m->spdx, sizeof(m->spdx), oc_json_str(lic, "spdx", ""));
	if (m->spdx[0] == '\0') {
		BAD("model %s: license.spdx is missing", m->name);
	}
	bool commercial = oc_json_bool(lic, "commercial") == 1;
	const struct oc_json *deps = oc_json_get(lic, "dependencies");
	for (const struct oc_json *d = deps != NULL ? deps->child : NULL; d != NULL; d = d->next) {
		commercial = commercial && oc_json_bool(d, "commercial") == 1;
	}
	m->commercial = commercial;

	const struct oc_json *vars = oc_json_get(jm, "variants");
	for (const struct oc_json *jv = vars != NULL ? vars->child : NULL; jv != NULL; jv = jv->next) {
		if (m->variant_count >= OC_PACK_MAX_VARIANTS) {
			BAD("model %s: more than %d variants", m->name, OC_PACK_MAX_VARIANTS);
		}
		if (!parse_variant(jv, &m->variants[m->variant_count], m->name, err, err_size)) {
			return false;
		}
		m->variant_count++;
	}
	if (m->variant_count == 0) {
		BAD("model %s: no variants", m->name);
	}
	return true;
}

bool
oc_pack_load(const char *path, struct oc_pack *out, char *err, size_t err_size)
{
	memset(out, 0, sizeof(*out));
	struct oc_json *root = oc_json_parse_file(path, err, err_size);
	if (root == NULL) {
		return false;
	}
	bool ok = false;
	do {
		if (root->type != OC_JSON_OBJECT || oc_json_num(root, "schema", 0.0) != 2.0) {
			snprintf(err, err_size, "pack %s: not a schema-2 manifest", path);
			break;
		}
		copy_str(out->pack, sizeof(out->pack), oc_json_str(root, "pack", ""));
		copy_str(out->blob_base_url, sizeof(out->blob_base_url), oc_json_str(root, "blobBaseUrl", ""));
		if (out->pack[0] == '\0' || strchr(out->pack, '/') != NULL) {
			snprintf(err, err_size, "pack %s: missing or invalid \"pack\" id", path);
			break;
		}
		out->commercial = oc_json_bool(root, "commercial") == 1;
		const struct oc_json *models = oc_json_get(root, "models");
		bool models_ok = models != NULL && models->type == OC_JSON_ARRAY;
		for (const struct oc_json *jm = models_ok ? models->child : NULL; jm != NULL; jm = jm->next) {
			if (out->model_count >= OC_PACK_MAX_MODELS) {
				snprintf(err, err_size, "pack %s: more than %d models", path, OC_PACK_MAX_MODELS);
				models_ok = false;
				break;
			}
			if (!parse_model(jm, &out->models[out->model_count], err, err_size)) {
				models_ok = false;
				break;
			}
			out->model_count++;
		}
		if (!models_ok) {
			if (err != NULL && err[0] == '\0') {
				snprintf(err, err_size, "pack %s: no models array", path);
			}
			break;
		}
		const struct oc_json *defs = oc_json_get(root, "defaults");
		for (const struct oc_json *d = defs != NULL ? defs->child : NULL; d != NULL; d = d->next) {
			if (out->default_count >= OC_PACK_MAX_DEFAULTS || d->key == NULL) {
				continue;
			}
			struct oc_pack_default *pd = &out->defaults[out->default_count++];
			copy_str(pd->task, sizeof(pd->task), d->key);
			copy_str(pd->low, sizeof(pd->low), oc_json_str(d, "low", ""));
			copy_str(pd->medium, sizeof(pd->medium), oc_json_str(d, "medium", ""));
			copy_str(pd->high, sizeof(pd->high), oc_json_str(d, "high", ""));
		}
		ok = true;
	} while (0);
	oc_json_free(root);
	return ok;
}

const struct oc_pack_model *
oc_pack_find(const struct oc_pack *pack, const char *name)
{
	for (uint32_t i = 0; pack != NULL && name != NULL && i < pack->model_count; i++) {
		if (strcmp(pack->models[i].name, name) == 0) {
			return &pack->models[i];
		}
	}
	return NULL;
}

const struct oc_pack_model *
oc_pack_default_model(const struct oc_pack *pack, const char *task, const char *quality)
{
	for (uint32_t i = 0; pack != NULL && i < pack->default_count; i++) {
		const struct oc_pack_default *d = &pack->defaults[i];
		if (strcmp(d->task, task) != 0) {
			continue;
		}
		const char *name = d->medium;
		if (quality != NULL && strcmp(quality, "low") == 0) {
			name = d->low;
		} else if (quality != NULL && strcmp(quality, "high") == 0) {
			name = d->high;
		}
		const struct oc_pack_model *m = oc_pack_find(pack, name);
		return m != NULL && strcmp(m->task, task) == 0 ? m : NULL;
	}
	return NULL;
}

const struct oc_pack_variant *
oc_pack_variant_for_ep(const struct oc_pack_model *m, const char *ep)
{
	for (uint32_t i = 0; m != NULL && i < m->variant_count; i++) {
		for (uint32_t e = 0; e < m->variants[i].ep_count; e++) {
			if (strcmp(m->variants[i].ep[e], ep) == 0) {
				return &m->variants[i];
			}
		}
	}
	return NULL;
}

void
oc_pack_ref(const struct oc_pack *pack,
            const struct oc_pack_model *m,
            const struct oc_pack_variant *v,
            char *buf,
            size_t buf_size)
{
	snprintf(buf, buf_size, "%s/%s/%s", pack->pack, m->name, v->id);
}
