// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Unit test of the session diagnostics parsing (src/oc_profile.c) on
 *         text in ONNX Runtime's formats: no inference runtime needed.
 */

#include "oc_profile.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;

#define CHECK(COND)                                                                                                    \
	do {                                                                                                           \
		if (!(COND)) {                                                                                         \
			printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #COND);                                         \
			g_failed++;                                                                                    \
		}                                                                                                      \
	} while (0)

static uint32_t
count_of(const struct oc_place_report *r, const char *ep, const char *op)
{
	for (uint32_t i = 0; i < r->entry_count; i++) {
		if (strcmp(r->entries[i].ep, ep) == 0 && strcmp(r->entries[i].op, op) == 0) {
			return r->entries[i].count;
		}
	}
	return 0;
}

static void
test_placement(void)
{
	struct oc_place_report r;
	memset(&r, 0, sizeof(r));
	CHECK(!oc_place_feed(&r, "Node placements"));
	CHECK(oc_place_feed(&r,
	                    " Node(s) placed on [DmlExecutionProvider]. Number of nodes: 4\n"
	                    "  DmlFusedNode_0_0 (DmlFusedNode_0_0)\n"
	                    "  DmlFusedNode_0_1 (DmlFusedNode_0_1)\n"
	                    "  MemcpyFromHost (Memcpy)\n"
	                    "  MemcpyToHost (Memcpy_token_1)"));
	CHECK(oc_place_feed(&r,
	                    " Node(s) placed on [CPUExecutionProvider]. Number of nodes: 3\n"
	                    "  Shape (/m/Shape)\n"
	                    "  Gather (/m/Gather)\n"
	                    "  Shape (/m/Shape_1)"));
	CHECK(r.seen);
	CHECK(count_of(&r, "DmlExecutionProvider", "DmlFusedNode") == 2);
	CHECK(count_of(&r, "DmlExecutionProvider", "MemcpyToHost") == 1);
	CHECK(count_of(&r, "CPUExecutionProvider", "Shape") == 2);
	CHECK(oc_place_total(&r, "CPUExecutionProvider") == 3);
	CHECK(oc_place_total(&r, "") == 7);
	char line[512];
	oc_place_format(&r, 2, line, sizeof(line));
	printf("placement: %s\n", line);
	CHECK(strstr(line, "DmlExecutionProvider: 4 nodes (DmlFusedNode 2") != NULL);
	CHECK(strstr(line, "CPUExecutionProvider: 3 nodes (Shape 2, Gather 1)") != NULL);

	struct oc_place_report one;
	memset(&one, 0, sizeof(one));
	CHECK(oc_place_feed(&one, " All nodes placed on [CPUExecutionProvider]. Number of nodes: 12"));
	CHECK(count_of(&one, "CPUExecutionProvider", "(all)") == 12);
	oc_place_format(&one, 4, line, sizeof(line));
	printf("placement: %s\n", line);

	struct oc_place_report none;
	memset(&none, 0, sizeof(none));
	oc_place_format(&none, 4, line, sizeof(line));
	CHECK(strstr(line, "no placement report") != NULL);
}

// Two runs (a cold and a warm one), in ONNX Runtime's profiler layout.
static const char k_profile[] =
    "[\n"
    "{\"cat\" : \"Session\",\"pid\" :1,\"tid\" :2,\"dur\" :500,\"ts\" :10,\"ph\" : \"X\",\"name\" "
    ":\"session_initialization\",\"args\" : {}},\n"
    "{\"cat\" : \"Node\",\"pid\" :1,\"tid\" :2,\"dur\" :900,\"ts\" :1000,\"ph\" : \"X\",\"name\" "
    ":\"/a/Conv_kernel_time\",\"args\" : {\"op_name\" : \"Conv\",\"provider\" : \"DmlExecutionProvider\"}},\n"
    "{\"cat\" : \"Session\",\"pid\" :1,\"tid\" :2,\"dur\" :5000,\"ts\" :900,\"ph\" : \"X\",\"name\" "
    ":\"model_run\",\"args\" : {}},\n"
    "{\"cat\" : \"Node\",\"pid\" :1,\"tid\" :2,\"dur\" :3,\"ts\" :10000,\"ph\" : \"X\",\"name\" "
    ":\"/a/Conv_fence_before\",\"args\" : {\"op_name\" : \"Conv\"}},\n"
    "{\"cat\" : \"Node\",\"pid\" :1,\"tid\" :2,\"dur\" :100,\"ts\" :10010,\"ph\" : \"X\",\"name\" "
    ":\"/a/Conv_kernel_time\",\"args\" : {\"thread_scheduling_stats\" : {\"main_thread\" : {\"thread_pool_name\" "
    ": \"session-1\"}},\"op_name\" : \"Conv\",\"provider\" : \"DmlExecutionProvider\"}},\n"
    "{\"cat\" : \"Node\",\"pid\" :1,\"tid\" :2,\"dur\" :250,\"ts\" :10200,\"ph\" : \"X\",\"name\" "
    ":\"/b/Shape_kernel_time\",\"args\" : {\"op_name\" : \"Shape\",\"provider\" : \"CPUExecutionProvider\"}},\n"
    "{\"cat\" : \"Node\",\"pid\" :1,\"tid\" :2,\"dur\" :3000,\"ts\" :10500,\"ph\" : \"X\",\"name\" "
    ":\"Memcpy_token_1_kernel_time\",\"args\" : {\"op_name\" : \"MemcpyToHost\",\"provider\" : "
    "\"DmlExecutionProvider\"}},\n"
    "{\"cat\" : \"Node\",\"pid\" :1,\"tid\" :2,\"dur\" :50,\"ts\" :13600,\"ph\" : \"X\",\"name\" "
    ":\"/c/Conv_kernel_time\",\"args\" : {\"op_name\" : \"Conv\",\"provider\" : \"DmlExecutionProvider\"}},\n"
    "{\"cat\" : \"Session\",\"pid\" :1,\"tid\" :2,\"dur\" :4000,\"ts\" :9990,\"ph\" : \"X\",\"name\" "
    ":\"model_run\",\"args\" : {}}\n"
    "]\n";

static void
test_profile(void)
{
	struct oc_profile_summary s;
	CHECK(oc_profile_summarize(k_profile, strlen(k_profile), &s));
	CHECK(s.runs == 2);
	CHECK(s.first_run_us == 5000);
	CHECK(s.last_run_us == 4000);
	CHECK(s.kernel_us == 3400); // the last run only, kernel_time events only
	CHECK(s.entry_count == 3);
	CHECK(strcmp(s.entries[0].op, "MemcpyToHost") == 0 && s.entries[0].us == 3000);
	CHECK(strcmp(s.entries[1].op, "Shape") == 0 && strcmp(s.entries[1].ep, "CPUExecutionProvider") == 0);
	CHECK(strcmp(s.entries[2].op, "Conv") == 0 && s.entries[2].us == 150 && s.entries[2].calls == 2);
	char line[512];
	oc_profile_format(&s, 10, line, sizeof(line));
	printf("profile: %s\n", line);
	CHECK(strstr(line, "MemcpyToHost@Dml 3.0 ms (1)") != NULL);
	CHECK(strstr(line, "Shape@CPU 0.2 ms (1)") != NULL || strstr(line, "Shape@CPU 0.3 ms (1)") != NULL);

	CHECK(!oc_profile_summarize("[]", 2, &s));
	CHECK(!oc_profile_summarize(NULL, 0, &s));
	// Truncated in the middle of an event: no crash, what is complete counts.
	CHECK(oc_profile_summarize(k_profile, sizeof(k_profile) / 2, &s));
}

int
main(void)
{
	test_placement();
	test_profile();
	printf("%s\n", g_failed == 0 ? "all passed" : "FAILED");
	return g_failed == 0 ? 0 : 1;
}
