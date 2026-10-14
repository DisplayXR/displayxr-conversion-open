// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Compile-only: the vendored headers stay self-contained and agree
 *         with each other (their own static asserts run here).
 */

#include "xrt/xrt_conversion_plugin.h"
#include "xrt/xrt_conv_wire.h"

int
openconv_header_check_anchor(void);

int
openconv_header_check_anchor(void)
{
	return (int)sizeof(struct xrt_conv_capability);
}
