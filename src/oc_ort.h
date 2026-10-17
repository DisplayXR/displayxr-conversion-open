// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ONNX Runtime, loaded at run time by absolute path, and the two
 *         inference shapes the provider needs (one image → one map; one frame
 *         + a temporal cache → one map + the next cache).
 *
 * The provider never statically imports ONNX Runtime: a missing or older
 * library is a readiness state (ABSENT / FAILED(VERSION)), never a load
 * failure of the provider itself, and a same-named library elsewhere in the
 * process (System32, another plug-in) can never be picked up by name.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oc_ort;
struct oc_session;
struct oc_temporal;

enum oc_ort_status
{
	OC_ORT_OK = 0,
	OC_ORT_ABSENT,  //!< the library is not there
	OC_ORT_TOO_OLD, //!< present, but it does not serve the API version we were built against
	OC_ORT_BROKEN,  //!< present and loadable, but not usable (missing symbol, env creation failed)
};

enum oc_ep
{
	OC_EP_CPU = 0,
	OC_EP_DML = 1,    //!< Windows
	OC_EP_COREML = 2, //!< macOS (Apple silicon)
};

/*! Core ML compute units (MLComputeUnits): where Core ML may place the graph. */
enum oc_coreml_units
{
	OC_COREML_ALL = 0, //!< CPU + GPU + Neural Engine (Core ML decides per segment)
	OC_COREML_GPU = 1, //!< CPU + GPU
	OC_COREML_ANE = 2, //!< CPU + Neural Engine
	OC_COREML_CPU = 3, //!< CPU only (a reference, never faster than the CPU EP)
};

/*! Execution-provider configuration for oc_session_create. */
struct oc_ep_config
{
	int dml_device;                    //!< DirectML: the DXGI adapter index
	enum oc_coreml_units coreml_units; //!< Core ML: compute units
	const char *coreml_cache_dir;      //!< Core ML: compiled-model cache directory; NULL = none
};

/*! What the provider prints of ONNX Runtime's log (stderr). The environment
 *  itself always logs at least INFO, so the Core ML partitioning summary is
 *  captured whatever is printed. */
enum oc_ort_log
{
	OC_ORT_LOG_WARNING = 0, //!< default
	OC_ORT_LOG_INFO = 1,    //!< + session-level info
	OC_ORT_LOG_VERBOSE = 2, //!< everything ONNX Runtime says; diagnostics only
};

/*! Graph optimisation level (ONNX Runtime's basic / extended / all). */
enum oc_graph_opt
{
	OC_OPT_BASIC = 0,
	OC_OPT_EXTENDED = 1,
	OC_OPT_ALL = 2,
};

enum oc_run_status
{
	OC_RUN_OK = 0,
	OC_RUN_FAILED,
	OC_RUN_DEVICE_LOST,
	OC_RUN_OUT_OF_MEMORY,
};

/*! The library file name next to the provider ("onnxruntime.dll", …). */
const char *
oc_ort_library_name(void);

/*! Load ONNX Runtime from @p lib_path (absolute). @p detail gets a sentence
 *  for a human in every case. @p log is what is printed (fixed for the
 *  process: the environment is never re-created). */
enum oc_ort_status
oc_ort_load(const char *lib_path, enum oc_ort_log log, struct oc_ort **out, char *detail, size_t detail_size);

/*! The runtime's own version string ("1.24.4"). */
const char *
oc_ort_version(const struct oc_ort *ort);

/*! The API version the provider was compiled against (the floor). */
uint32_t
oc_ort_api_floor(void);

/*! True when this ONNX Runtime build carries the DirectML execution provider. */
bool
oc_ort_has_dml(const struct oc_ort *ort);

/*! True when this ONNX Runtime build carries the Core ML execution provider. */
bool
oc_ort_has_coreml(const struct oc_ort *ort);

/*! Windows: the DXGI adapter index (DirectML device id) for @p preferred_luid
 *  (0 = the first hardware adapter), skipping @p avoid_luid and software
 *  adapters. false = no hardware adapter. Other OSes: always false. */
bool
oc_dml_pick_adapter(uint64_t preferred_luid, uint64_t avoid_luid, int *out_index, uint64_t *out_luid);

/*! Create a session for the model file at @p path (UTF-8, absolute; the
 *  verified blob the host resolved). @p threads 0 = ONNX Runtime's default.
 *  Core ML partitions the graph: what it cannot take runs on the CPU EP in
 *  the same session, so a Core ML session always runs. oc_session_ep_nodes
 *  says how much of it Core ML took. */
enum oc_run_status
oc_session_create(struct oc_ort *ort,
                  const char *path,
                  enum oc_ep ep,
                  const struct oc_ep_config *epc,
                  int threads,
                  enum oc_graph_opt opt,
                  struct oc_session **out,
                  char *err,
                  size_t err_size);

void
oc_session_destroy(struct oc_session *s);

/*! Core ML only: the graph nodes Core ML took, the graph's node count and the
 *  number of Core ML partitions, as ONNX Runtime reported them while
 *  partitioning. false when not reported. */
bool
oc_session_ep_nodes(const struct oc_session *s, uint32_t *taken, uint32_t *total, uint32_t *partitions);

/*! One image → one map. @p chw is [1,3,h,w] float; @p out receives the first
 *  output's out_w × out_h floats (the caller checked the model's output
 *  shape rule); @p out_w / @p out_h are written with the actual shape. */
enum oc_run_status
oc_session_run_image(struct oc_session *s,
                     const char *in_name,
                     const float *chw,
                     uint32_t w,
                     uint32_t h,
                     const char *out_name,
                     float *out,
                     size_t out_capacity,
                     uint32_t *out_w,
                     uint32_t *out_h,
                     char *err,
                     size_t err_size);

/*! Names and the cache shape of a streaming model with an explicit cache. */
struct oc_temporal_io
{
	const char *image, *cache_in, *is_first, *depth, *cache_out;
	const int64_t *cache_shape;
	uint32_t cache_rank;
};

/*! One frame of a streaming model. @p state is the per-stream cache (created
 *  on the first call; kept on the session's device between frames). @p first
 *  drops the temporal history BEFORE this frame. */
enum oc_run_status
oc_temporal_run(struct oc_session *s,
                struct oc_temporal **state,
                const struct oc_temporal_io *io,
                const float *chw,
                uint32_t w,
                uint32_t h,
                bool first,
                float *out,
                size_t out_capacity,
                uint32_t *out_w,
                uint32_t *out_h,
                char *err,
                size_t err_size);

void
oc_temporal_destroy(struct oc_temporal *t);

#ifdef __cplusplus
}
#endif
