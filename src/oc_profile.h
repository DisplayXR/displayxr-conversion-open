// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Diagnostics for one inference session (setting DIAG_PROFILE): where
 *         ONNX Runtime placed the graph's nodes (its verbose "Node placements"
 *         report) and which operators took the time (its profiling JSON).
 *         Pure parsing, no ONNX Runtime calls: unit-tested by
 *         tests/profile_test.c.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OC_PLACE_MAX 96u    //!< distinct (execution provider, operator) pairs kept
#define OC_PROFILE_MAX 128u //!< distinct (execution provider, operator) pairs aggregated

/*! The node count of one operator type on one execution provider. */
struct oc_place_entry
{
	char ep[40]; //!< "DmlExecutionProvider", "CPUExecutionProvider", ...
	char op[48]; //!< "Conv", "MemcpyToHost", "DmlFusedNode_0_3" is counted as "DmlFusedNode"
	uint32_t count;
};

/*! ONNX Runtime's node placement report for one session. */
struct oc_place_report
{
	bool seen; //!< a placement message was parsed
	uint32_t entry_count;
	struct oc_place_entry entries[OC_PLACE_MAX];
	uint32_t dropped; //!< nodes whose (ep, op) pair did not fit
};

/*! Feed one ONNX Runtime log message; the placement messages are parsed into
 *  @p r ("All nodes placed on [EP]. Number of nodes: N" and
 *  " Node(s) placed on [EP]. Number of nodes: N" followed by one "  Op (name)"
 *  line per node). Other messages are ignored. Returns true when @p msg was one. */
bool
oc_place_feed(struct oc_place_report *r, const char *msg);

/*! Nodes on @p ep ("" = all). */
uint32_t
oc_place_total(const struct oc_place_report *r, const char *ep);

/*! One line: "EP: n nodes (op k, op k, ...); EP2: ..." with each EP's ops by count, at
 *  most @p max_ops per EP. */
void
oc_place_format(const struct oc_place_report *r, uint32_t max_ops, char *buf, size_t size);

/*! Time spent in one operator type on one execution provider in a profiled run. */
struct oc_profile_entry
{
	char ep[40];
	char op[48];
	uint64_t us; //!< summed kernel time
	uint32_t calls;
};

struct oc_profile_summary
{
	uint32_t runs;         //!< model_run events in the file
	uint64_t last_run_us;  //!< the duration of the last one
	uint64_t first_run_us; //!< ... and of the first one
	uint64_t kernel_us;    //!< the summed kernel time inside the last run
	uint32_t entry_count;
	struct oc_profile_entry entries[OC_PROFILE_MAX]; //!< by us, largest first
};

/*! Summarise ONNX Runtime's profiling JSON (@p len bytes, need not be NUL-terminated):
 *  the node events (`"cat" : "Node"`, names ending "_kernel_time") inside the LAST
 *  `model_run` event, by (provider, op_name). False when the text has no model_run. */
bool
oc_profile_summarize(const char *json, size_t len, struct oc_profile_summary *out);

/*! One line: the top @p n entries as "op@ep ms (calls), ..." plus the run times. */
void
oc_profile_format(const struct oc_profile_summary *s, uint32_t n, char *buf, size_t size);

#ifdef __cplusplus
}
#endif
