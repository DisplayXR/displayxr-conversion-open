// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Unit test of the splat backend ranking (src/oc_rank.c) on described
 *         machines: no GPU, no inference runtime, no model.
 */

#include "oc_rank.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;
static bool g_fp16_on_dml = false; //!< the pack has an fp16 model (DirectML variant)
static bool g_fp16_ready = false;  //!< ... and its blob is in the store

#define GB(x) ((uint64_t)(x) * 1024u)

static const struct oc_dxgi_adapter k_uhd630 = {"Intel(R) UHD Graphics 630", OC_VENDOR_INTEL, 128, false, 0, 0x100};
static const struct oc_dxgi_adapter k_3080_laptop = {
    "NVIDIA GeForce RTX 3080 Laptop GPU", OC_VENDOR_NVIDIA, GB(16), false, 1, 0x200};
static const struct oc_dxgi_adapter k_3080 = {"NVIDIA GeForce RTX 3080", OC_VENDOR_NVIDIA, GB(10), false, 0, 0x300};
static const struct oc_dxgi_adapter k_3050_4g = {"NVIDIA GeForce RTX 3050", OC_VENDOR_NVIDIA, GB(4), false, 0, 0x310};
static const struct oc_dxgi_adapter k_radeon_apu = {"AMD Radeon(TM) Graphics", OC_VENDOR_AMD, 512, false, 0, 0x400};
static const struct oc_dxgi_adapter k_7900xtx = {"AMD Radeon RX 7900 XTX", OC_VENDOR_AMD, GB(24), false, 1, 0x500};
static const struct oc_dxgi_adapter k_warp = {"Microsoft Basic Render Driver", OC_VENDOR_MICROSOFT, 0, true, 1, 0x600};

static const struct oc_ov_gpu k_ov_uhd630 = {.device = "GPU",
                                             .name = "Intel(R) UHD Graphics 630 (iGPU)",
                                             .type = OC_OV_GPU_TYPE_INTEGRATED,
                                             .eu_count = 24,
                                             .vendor_id = OC_VENDOR_INTEL,
                                             .arch_major = 9,
                                             .arch_minor = 5};
static const struct oc_ov_gpu k_ov_nvidia = {.device = "GPU",
                                             .name = "NVIDIA GeForce RTX 3080 (dGPU)",
                                             .type = OC_OV_GPU_TYPE_DISCRETE,
                                             .vendor_id = OC_VENDOR_NVIDIA};
//! What OpenVINO 2026.4 reports for a Core Ultra 7 355's Xe3 iGPU (32 EUs: the EU count
//! of new parts is not comparable to Xe-LP's, so it must not reject this one).
static const struct oc_ov_gpu k_ov_arc = {.device = "GPU",
                                          .name = "Intel(R) Graphics [0xb090] (iGPU)",
                                          .type = OC_OV_GPU_TYPE_INTEGRATED,
                                          .eu_count = 32,
                                          .vendor_id = OC_VENDOR_INTEL,
                                          .arch_major = 30,
                                          .arch_minor = 1,
                                          .hw_matmul = true};

struct machine
{
	const char *name;
	bool windows; //!< ONNX Runtime carries DirectML
	struct oc_dxgi_adapter adapters[4];
	uint32_t adapter_count;
	bool ov_present;
	bool ov_known;
	struct oc_ov_gpu ov_gpus[2];
	uint32_t ov_gpu_count;
	uint64_t ram_mb;
};

static struct machine
mk(const char *name, bool windows, uint64_t ram_mb, bool ov_present, bool ov_known)
{
	struct machine m;
	memset(&m, 0, sizeof(m));
	m.name = name;
	m.windows = windows;
	m.ram_mb = ram_mb;
	m.ov_present = ov_present;
	m.ov_known = ov_known;
	return m;
}

static void
add_adapter(struct machine *m, const struct oc_dxgi_adapter *a)
{
	m->adapters[m->adapter_count++] = *a;
}

static void
add_ov(struct machine *m, const struct oc_ov_gpu *g)
{
	m->ov_gpus[m->ov_gpu_count++] = *g;
}

static void
expect(const struct machine *m,
       const char *ep,
       const char *ov_device,
       const char *model,
       enum oc_splat_backend backend,
       int adapter_index, //!< DXGI index (-1 = n/a)
       enum oc_splat_model model_want,
       bool want_fallback)
{
	const struct oc_splat_env env = {
	    .ep = ep,
	    .ov_device = ov_device,
	    .model = model,
	    .ram_mb = m->ram_mb,
	    .dml_available = m->windows,
	    .coreml_available = false,
	    .adapters = m->adapters,
	    .adapter_count = m->adapter_count,
	    .full_on_dml = true,
	    .fp16_on_dml = g_fp16_on_dml,
	    .fp16_ready = g_fp16_ready,
	    .ov_present = m->ov_present,
	    .ov_known = m->ov_known,
	    .ov_gpus = m->ov_gpus,
	    .ov_gpu_count = m->ov_gpu_count,
	};
	struct oc_splat_choice c;
	oc_rank_splat(&env, &c);
	const int got_index = c.backend == OC_SB_DML ? m->adapters[c.adapter].index : -1;
	const bool ok = c.backend == backend && got_index == adapter_index && c.model == model_want &&
	                (c.fallback[0] != '\0') == want_fallback;
	printf("%s %-34s%s EP=%-8s OV_DEVICE=%-4s MODEL=%-4s -> %s%s%s, model %s | %s%s%s\n", ok ? "ok  " : "FAIL",
	       m->name,
	       g_fp16_ready    ? " [fp16 built]"
	       : g_fp16_on_dml ? " [fp16 in pack]"
	                       : "",
	       ep, ov_device, model, oc_rank_backend_name(c.backend), c.backend == OC_SB_DML ? " #" : "",
	       c.backend == OC_SB_DML ? m->adapters[c.adapter].name : "", oc_rank_model_name(c.model), c.why,
	       c.fallback[0] ? " | fallback: " : "", c.fallback);
	if (!ok) {
		printf("      expected %s (adapter %d), model %s, fallback %d\n", oc_rank_backend_name(backend),
		       adapter_index, oc_rank_model_name(model_want), want_fallback);
		g_failed++;
	}
}

static void
gpu_ok(const char *name, enum oc_ov_gpu_type type, uint32_t eus, const char *arch, bool hw_matmul, bool want)
{
	struct oc_ov_gpu g;
	memset(&g, 0, sizeof(g));
	snprintf(g.device, sizeof(g.device), "GPU");
	snprintf(g.name, sizeof(g.name), "%s", name);
	g.type = type;
	g.eu_count = eus;
	g.hw_matmul = hw_matmul;
	oc_rank_parse_ov_arch(arch, &g);
	char why[160];
	const bool got = oc_rank_ov_gpu_ok(&g, why, sizeof(why));
	printf("%s OpenVINO GPU '%s' (%s, %u EUs%s): %s %s\n", got == want ? "ok  " : "FAIL", name,
	       arch != NULL ? arch : "no arch", eus, hw_matmul ? ", matmul" : "", got ? "used" : "not used:", why);
	if (got != want) {
		g_failed++;
	}
}

int
main(void)
{
	// A laptop: Intel UHD iGPU + RTX 3080 Laptop, 32 GB; OpenVINO not loaded yet, then loaded.
	struct machine conceptd = mk("uhd630+rtx3080-laptop (probe)", true, GB(32), OC_SM_FULL, false);
	add_adapter(&conceptd, &k_uhd630);
	add_adapter(&conceptd, &k_3080_laptop);
	struct machine conceptd_ov = conceptd;
	conceptd_ov.name = "uhd630+rtx3080-laptop (ov listed)";
	conceptd_ov.ov_known = true;
	add_ov(&conceptd_ov, &k_ov_uhd630);
	// Desktop: RTX 3080 only (iGPU disabled), 64 GB; OpenVINO's GPU plug-in sees the NVIDIA card.
	struct machine desk = mk("rtx3080-desktop", true, GB(64), OC_SM_FULL, true);
	add_adapter(&desk, &k_3080);
	add_ov(&desk, &k_ov_nvidia);
	struct machine desk_8g = desk;
	desk_8g.name = "rtx3080-desktop, 8 GB RAM";
	desk_8g.ram_mb = GB(8);
	// Linux laptop with an Intel Arc iGPU only, 32 GB (no DirectML).
	const struct machine arc_probe = mk("linux arc (probe)", false, GB(32), OC_SM_FULL, false);
	struct machine arc = mk("linux arc (ov listed)", false, GB(32), OC_SM_FULL, true);
	add_ov(&arc, &k_ov_arc);
	struct machine arc_8g = arc;
	arc_8g.name = "linux arc, 8 GB RAM";
	arc_8g.ram_mb = GB(8);
	// Linux with an NVIDIA card that OpenVINO reaches through OpenCL.
	struct machine linux_nv = mk("linux nvidia", false, GB(64), OC_SM_FULL, true);
	add_ov(&linux_nv, &k_ov_nvidia);
	// Windows with an AMD APU + a discrete Radeon.
	struct machine amd = mk("radeon apu + rx7900xtx", true, GB(32), OC_SM_FULL, true);
	add_adapter(&amd, &k_radeon_apu);
	add_adapter(&amd, &k_7900xtx);
	// A small discrete card.
	struct machine small = mk("rtx3050 4 GB", true, GB(32), OC_SM_FULL, true);
	add_adapter(&small, &k_3050_4g);
	// No GPU at all: WARP only.
	struct machine nogpu = mk("no gpu (warp)", true, GB(16), OC_SM_FULL, true);
	add_adapter(&nogpu, &k_warp);
	struct machine nogpu_noov = nogpu;
	nogpu_noov.name = "no gpu, no openvino";
	nogpu_noov.ov_present = false;

	// EP=auto.
	expect(&conceptd, "auto", "auto", "auto", OC_SB_DML, 1, OC_SM_FULL, false);
	expect(&conceptd_ov, "auto", "auto", "auto", OC_SB_DML, 1, OC_SM_FULL, false);
	expect(&desk, "auto", "auto", "auto", OC_SB_DML, 0, OC_SM_FULL, false);
	expect(&desk_8g, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&arc_probe, "auto", "auto", "auto", OC_SB_OV_GPU, -1, OC_SM_FULL, false);
	expect(&arc, "auto", "auto", "auto", OC_SB_OV_GPU, -1, OC_SM_FULL, false);
	expect(&arc_8g, "auto", "auto", "auto", OC_SB_OV_GPU, -1, OC_SM_INT4, false);
	expect(&linux_nv, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&amd, "auto", "auto", "auto", OC_SB_DML, 1, OC_SM_FULL, false);
	expect(&small, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&nogpu, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&nogpu_noov, "auto", "auto", "auto", OC_SB_ORT_CPU, -1, OC_SM_INT4, false);
	// DirectML failed in this process (dml_available false): the rest of the ranking.
	struct machine conceptd_nodml = conceptd_ov;
	conceptd_nodml.name = "uhd630+rtx3080 (DirectML failed)";
	conceptd_nodml.windows = false;
	expect(&conceptd_nodml, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);

	// Explicit settings win.
	expect(&conceptd, "dml", "auto", "auto", OC_SB_DML, 1, OC_SM_FULL, false);
	expect(&conceptd, "dml", "auto", "int4", OC_SB_DML, 1, OC_SM_INT4,
	       false); // int4 then runs on the CPU (no variant)
	expect(&small, "dml", "auto", "auto", OC_SB_DML, 0, OC_SM_INT4, false);
	expect(&arc, "dml", "auto", "auto", OC_SB_ORT_CPU, -1, OC_SM_INT4, true);
	expect(&nogpu, "dml", "auto", "auto", OC_SB_ORT_CPU, -1, OC_SM_INT4, true);
	expect(&desk, "cpu", "auto", "auto", OC_SB_ORT_CPU, -1, OC_SM_INT4, false);
	expect(&desk, "cpu", "auto", "full", OC_SB_ORT_CPU, -1, OC_SM_FULL, false);
	expect(&desk, "openvino", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&desk, "openvino", "gpu", "auto", OC_SB_OV_GPU, -1, OC_SM_FULL, false);
	expect(&desk, "openvino", "cpu", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&nogpu_noov, "openvino", "auto", "auto", OC_SB_ORT_CPU, -1, OC_SM_INT4, true);
	expect(&nogpu, "openvino", "gpu", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, true);
	expect(&desk, "auto", "auto", "int4", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&desk_8g, "auto", "auto", "full", OC_SB_DML, 0, OC_SM_FULL, false);
	expect(&arc, "auto", "cpu", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	// OV_DEVICE picks OpenVINO's device; it does not select OpenVINO over DirectML.
	expect(&desk, "auto", "cpu", "auto", OC_SB_DML, 0, OC_SM_FULL, false);
	expect(&linux_nv, "auto", "gpu", "auto", OC_SB_OV_GPU, -1, OC_SM_FULL, false);

	// The DirectML VRAM floors (measured: the fp32 model took 6.1 GB of a 10 GB card).
	struct machine card_8g = mk("rtx 8 GB desktop", true, GB(32), true, true);
	add_adapter(&card_8g,
	            &(struct oc_dxgi_adapter){"NVIDIA GeForce RTX 3070", OC_VENDOR_NVIDIA, GB(8), false, 0, 0x700});
	add_ov(&card_8g, &k_ov_nvidia);
	struct machine card_10g = desk; // RTX 3080, 10 GB
	struct machine card_9g = mk("rtx 10 GB card reporting 9.8 GiB", true, GB(32), true, true);
	add_adapter(&card_9g,
	            &(struct oc_dxgi_adapter){"NVIDIA GeForce RTX 3080", OC_VENDOR_NVIDIA, 10067, false, 0, 0x710});
	add_ov(&card_9g, &k_ov_nvidia);
	// No fp16 model in any pack: 8 GB is under the full model's floor -> the next backend.
	expect(&card_8g, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&card_9g, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&card_10g, "auto", "auto", "auto", OC_SB_DML, 0, OC_SM_FULL, false);
	expect(&small, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	// An fp16 model in the pack but not built: the same.
	g_fp16_on_dml = true;
	expect(&card_8g, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&card_10g, "auto", "auto", "auto", OC_SB_DML, 0, OC_SM_FULL, false);
	// Built: DirectML takes the fp16 model wherever it fits (6 GB floor), first.
	g_fp16_ready = true;
	expect(&card_8g, "auto", "auto", "auto", OC_SB_DML, 0, OC_SM_FP16, false);
	expect(&card_9g, "auto", "auto", "auto", OC_SB_DML, 0, OC_SM_FP16, false);
	expect(&card_10g, "auto", "auto", "auto", OC_SB_DML, 0, OC_SM_FP16, false);
	expect(&conceptd, "auto", "auto", "auto", OC_SB_DML, 1, OC_SM_FP16, false);
	expect(&small, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false);   // 4 GB: under both floors
	expect(&desk_8g, "auto", "auto", "auto", OC_SB_OV_CPU, -1, OC_SM_INT4, false); // 8 GB of RAM
	// Explicit MODEL wins.
	expect(&card_10g, "auto", "auto", "full", OC_SB_DML, 0, OC_SM_FULL, false);
	expect(&card_8g, "auto", "auto", "full", OC_SB_DML, 0, OC_SM_FULL, false);
	expect(&card_8g, "auto", "auto", "int4", OC_SB_OV_CPU, -1, OC_SM_INT4, false);
	expect(&small, "auto", "auto", "fp16", OC_SB_DML, 0, OC_SM_FP16, false);
	expect(&arc, "auto", "auto", "fp16", OC_SB_OV_GPU, -1, OC_SM_FULL, false); // fp16 is DirectML-only
	g_fp16_on_dml = false;
	g_fp16_ready = false;
	expect(&card_8g, "auto", "auto", "fp16", OC_SB_OV_CPU, -1, OC_SM_FULL, false); // no fp16 model anywhere

	// Which OpenVINO GPUs EP=auto uses.
	gpu_ok("Intel(R) Graphics [0xb090] (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 32, "GPU: vendor=0x8086 arch=v30.1.0",
	       true, true);
	gpu_ok("Intel(R) Arc(TM) Graphics (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 128, "GPU: vendor=0x8086 arch=v12.71.4",
	       false, true);
	gpu_ok("Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 64,
	       "GPU: vendor=0x8086 arch=v20.4.4", OC_SM_FULL, true);
	gpu_ok("Intel(R) Arc(TM) A770 Graphics (dGPU)", OC_OV_GPU_TYPE_DISCRETE, 512,
	       "GPU: vendor=0x8086 arch=v12.55.8", OC_SM_FULL, true);
	gpu_ok("Intel(R) Iris(R) Xe Graphics (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 96, "GPU: vendor=0x8086 arch=v12.0.0",
	       false, true);
	gpu_ok("Intel(R) Graphics (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 32, "GPU: vendor=0x8086 arch=v12.0.0", false,
	       false);
	gpu_ok("Intel(R) Graphics (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 32, NULL, OC_SM_INT4, false);
	gpu_ok("Intel(R) UHD Graphics 770 (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 32, "GPU: vendor=0x8086 arch=v12.0.0",
	       false, false);
	gpu_ok("Intel(R) UHD Graphics 630 (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 24, "GPU: vendor=0x8086 arch=v9.5.0",
	       false, false);
	gpu_ok("Intel(R) HD Graphics 620 (iGPU)", OC_OV_GPU_TYPE_INTEGRATED, 24, NULL, OC_SM_INT4, false);
	gpu_ok("NVIDIA GeForce RTX 3080 (dGPU)", OC_OV_GPU_TYPE_DISCRETE, 0, "GPU: vendor=0x10DE arch=v0.0.0", false,
	       false);
	gpu_ok("AMD Radeon RX 7900 XTX (dGPU)", OC_OV_GPU_TYPE_DISCRETE, 0, NULL, OC_SM_INT4, false);

	printf("%s\n", g_failed == 0 ? "all passed" : "FAILED");
	return g_failed == 0 ? 0 : 1;
}
