// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  A stand-in "ONNX Runtime" that is too old: it exports OrtGetApiBase
 *         but serves no API version the provider was built against. The
 *         provider must report FAILED(VERSION), never FAILED(MODEL) or ABSENT.
 *
 * Self-contained on purpose: onnxruntime_c_api.h declares OrtGetApiBase as an
 * import. The layout is OrtApiBase's (GetApi, GetVersionString), 64-bit only.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define FAKE_EXPORT __declspec(dllexport)
#else
#define FAKE_EXPORT __attribute__((visibility("default")))
#endif

struct fake_api_base
{
	const void *(*get_api)(uint32_t version);
	const char *(*get_version_string)(void);
};

static const void *
fake_get_api(uint32_t version)
{
	(void)version;
	return NULL; // "this runtime predates the requested API"
}

static const char *
fake_version(void)
{
	return "1.0.0";
}

static const struct fake_api_base g_base = {fake_get_api, fake_version};

FAKE_EXPORT const struct fake_api_base *
OrtGetApiBase(void);

FAKE_EXPORT const struct fake_api_base *
OrtGetApiBase(void)
{
	return &g_base;
}
