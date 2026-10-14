// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  A small strict JSON reader (RFC 8259): enough for model-pack
 *         manifests, shared by the provider and the fake host.
 *
 * Parses into a malloc'd tree; no comments, no trailing commas. Depth is
 * capped at 64 and input at 4 MiB, so a hostile manifest cannot exhaust the
 * stack or the heap of the hosting process.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OC_JSON_MAX_BYTES (4u * 1024u * 1024u)

enum oc_json_type
{
	OC_JSON_NULL = 0,
	OC_JSON_BOOL,
	OC_JSON_NUMBER,
	OC_JSON_STRING,
	OC_JSON_ARRAY,
	OC_JSON_OBJECT,
};

struct oc_json
{
	enum oc_json_type type;
	bool boolean;
	double number;
	char *string;          //!< OC_JSON_STRING: UTF-8, NUL-terminated
	char *key;             //!< member name when the parent is an object
	struct oc_json *child; //!< first element / member
	struct oc_json *next;  //!< next sibling
	size_t count;          //!< elements / members
};

/*! Parse @p len bytes. Returns NULL and a message in @p err on failure. */
struct oc_json *
oc_json_parse(const char *text, size_t len, char *err, size_t err_size);

/*! Read a whole file (≤ OC_JSON_MAX_BYTES) and parse it. UTF-8 path. */
struct oc_json *
oc_json_parse_file(const char *path, char *err, size_t err_size);

void
oc_json_free(struct oc_json *root);

/*! Member @p key of an object, or NULL. */
const struct oc_json *
oc_json_get(const struct oc_json *obj, const char *key);

/*! Element @p index of an array, or NULL. */
const struct oc_json *
oc_json_at(const struct oc_json *arr, size_t index);

/*! Typed member accessors: the default when absent or of another type. */
const char *
oc_json_str(const struct oc_json *obj, const char *key, const char *def);
double
oc_json_num(const struct oc_json *obj, const char *key, double def);
/*! -1 = absent / not a boolean, 0 = false, 1 = true. */
int
oc_json_bool(const struct oc_json *obj, const char *key);

#ifdef __cplusplus
}
#endif
