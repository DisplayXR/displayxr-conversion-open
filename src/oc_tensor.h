// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Named float tensors in and out of one inference, shared by the
 *         ONNX Runtime and OpenVINO back ends (the splat capability's run).
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! One float input, by tensor name (host memory, only read). */
struct oc_tensor_in
{
	const char *name;
	const float *data;
	const int64_t *shape;
	uint32_t rank;
};

/*! One float output, by tensor name: copied into @p data (capacity in
 *  floats); @p count gets the element count. */
struct oc_tensor_out
{
	const char *name;
	float *data;
	size_t capacity;
	size_t count;
};

#ifdef __cplusplus
}
#endif
