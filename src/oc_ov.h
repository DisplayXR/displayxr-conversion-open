// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  OpenVINO, loaded at run time by absolute path through its C API
 *         (`openvino/c/openvino.h`), and the one inference shape the splat
 *         capability needs: named float inputs → named float outputs.
 *
 * Like ONNX Runtime, OpenVINO is never statically imported: a missing or
 * older library is a readiness state (the capability moves to the next
 * backend), never a load failure of the provider. The runtime lives in its
 * own folder next to the provider (`openvino/`), because it is many files
 * (the core, the C API, the ONNX front end, one plug-in per device, oneTBB);
 * on Linux its dependencies are mapped one by one by absolute path, in
 * dependency order, so nothing is resolved through the library search path.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oc_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

struct oc_ov;
struct oc_ov_session;

enum oc_ov_status
{
	OC_OV_OK = 0,
	OC_OV_ABSENT,  //!< the runtime folder or its C API library is not there
	OC_OV_TOO_OLD, //!< present, older than OC_OV_MIN_YEAR.OC_OV_MIN_RELEASE
	OC_OV_BROKEN,  //!< present but not usable (a library did not map, a symbol is missing, no core)
};

/*! The oldest OpenVINO release the provider accepts: its ONNX front end must
 *  read ONNX Runtime's `MatMulNBits` (4-bit weight) contrib operator. */
#define OC_OV_MIN_YEAR 2025
#define OC_OV_MIN_RELEASE 0

/*! Linux: the SONAME suffix of the shipped runtime's libraries
 *  (libopenvino.so.<N>, libopenvino_c.so.<N>); cmake/OpenVINO.cmake passes it. */
#ifndef OC_OV_SOVERSION
#define OC_OV_SOVERSION "2641"
#endif

/*! The folder name next to the provider. */
const char *
oc_ov_dir_name(void);

/*! Load OpenVINO from @p dir (absolute; the folder holding the C API
 *  library). @p detail gets a sentence for a human in every case. */
enum oc_ov_status
oc_ov_load(const char *dir, struct oc_ov **out, char *detail, size_t detail_size);

/*! The runtime's build number ("2026.4.1-22982-…"); "" before a load. */
const char *
oc_ov_version(const struct oc_ov *ov);

/*! True when OpenVINO reports a device whose name starts with @p prefix
 *  ("GPU" matches "GPU", "GPU.0", "GPU.1"). @p full gets the first match's
 *  full name (FULL_DEVICE_NAME) when not NULL. */
bool
oc_ov_has_device(struct oc_ov *ov, const char *prefix, char *full, size_t full_size);

/*! Read and compile the model at @p path (UTF-8, absolute) for @p device
 *  ("GPU" / "CPU"). @p cache_dir (NULL = none) is OpenVINO's compiled-model
 *  cache (CACHE_DIR): a later compile of the same model on the same device
 *  imports the cached blob instead of compiling. @p threads > 0 bounds the
 *  CPU device's inference threads. */
bool
oc_ov_session_create(struct oc_ov *ov,
                     const char *path,
                     const char *device,
                     const char *cache_dir,
                     int threads,
                     struct oc_ov_session **out,
                     char *err,
                     size_t err_size);

void
oc_ov_session_destroy(struct oc_ov_session *s);

enum oc_ov_run
{
	OC_OV_RUN_OK = 0,
	OC_OV_RUN_FAILED,
	OC_OV_RUN_CANCELLED,
};

/*! Run once. The inference runs asynchronously and @p is_cancelled is polled
 *  every 100 ms (cancel = the request is cancelled, OC_OV_RUN_CANCELLED). */
enum oc_ov_run
oc_ov_run(struct oc_ov_session *s,
          const struct oc_tensor_in *in,
          uint32_t in_count,
          struct oc_tensor_out *out,
          uint32_t out_count,
          bool (*is_cancelled)(void *ctx),
          void *cancel_ctx,
          char *err,
          size_t err_size);

#ifdef __cplusplus
}
#endif
