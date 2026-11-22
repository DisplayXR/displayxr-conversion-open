# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# OpenVINO (Apache-2.0) as a PINNED release archive from Intel's public storage, fetched at
# configure time and verified by SHA-256 before anything is extracted. Like ONNX Runtime it is
# never linked: the provider maps it at run time by absolute path from the `openvino/` folder
# next to its binary. Only the C API headers are used at build time.
#
# Linux x86_64 takes the Ubuntu 22.04 build (the oldest release DisplayXR supports, so its
# glibc floor is the lowest); Windows x64 the static-CRT (vc_mt) build, so nothing needs the
# Visual C++ redistributable. macOS is off: there the splat capability runs on ONNX Runtime.
#
# Outputs (when OPENCONV_OPENVINO is ON):
#   OPENCONV_OV_INCLUDE_DIR      openvino/c/openvino.h
#   OPENCONV_OV_RUNTIME_FILES    files for <provider dir>/openvino/
#   OPENCONV_OV_LICENSE_FILES    licence texts shipped with them (+ OPENCONV_OV_LICENSE_NAMES)
#   OPENCONV_OV_SOVERSION        Linux SONAME suffix of libopenvino*.so.<N>

set(OPENCONV_OV_VERSION 2026.4.1)
set(OPENCONV_OV_BUILD 2026.4.1.22982.07f9c262b05)
set(OPENCONV_OV_SOVERSION 2641)
set(_ov_base "https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENCONV_OV_VERSION}")
set(_ov_root "${CMAKE_BINARY_DIR}/_deps/openvino-${OPENCONV_OV_VERSION}")

if(WIN32)
	set(_ov_name "openvino_toolkit_windows_vc_mt_${OPENCONV_OV_BUILD}_x86_64")
	openconv_fetch("${_ov_base}/windows_vc_mt/${_ov_name}.zip" "${_ov_name}.zip"
		5dc0c993c2033cf2937881cc7b1a72bfedb2a296582bf5916ef12d68441fc2ca)
	set(_ov_archive "${OPENCONV_DOWNLOAD_DIR}/${_ov_name}.zip")
	set(_ov_bin "${_ov_name}/runtime/bin/intel64/Release")
	set(_ov_tbb "${_ov_name}/runtime/3rdparty/tbb/bin")
	set(_ov_libs openvino.dll openvino_c.dll openvino_onnx_frontend.dll openvino_ir_frontend.dll
		openvino_intel_cpu_plugin.dll openvino_intel_gpu_plugin.dll cache.json)
	set(_ov_tbb_libs tbb12.dll tbbmalloc.dll tbbbind_2_5.dll)
else()
	set(_ov_name "openvino_toolkit_ubuntu22_${OPENCONV_OV_BUILD}_x86_64")
	openconv_fetch("${_ov_base}/linux/${_ov_name}.tgz" "${_ov_name}.tgz"
		e2541afe65a81d5aabe981f9a86ad2c7efe557922b71e5649b6aafffec7c982b)
	set(_ov_archive "${OPENCONV_DOWNLOAD_DIR}/${_ov_name}.tgz")
	set(_ov_bin "${_ov_name}/runtime/lib/intel64")
	set(_ov_tbb "${_ov_name}/runtime/3rdparty/tbb/lib")
	# The real files under their SONAMEs (the archive's symlink chains are not shipped).
	set(_ov_libs "libopenvino.so.${OPENCONV_OV_VERSION}" "libopenvino_c.so.${OPENCONV_OV_VERSION}"
		"libopenvino_onnx_frontend.so.${OPENCONV_OV_VERSION}" "libopenvino_ir_frontend.so.${OPENCONV_OV_VERSION}"
		libopenvino_intel_cpu_plugin.so libopenvino_intel_gpu_plugin.so cache.json)
	set(_ov_tbb_libs libtbb.so.12.13 libtbbmalloc.so.2.13 libtbbbind_2_5.so.3.13 libhwloc.so.15.6.4)
endif()

if(NOT EXISTS "${_ov_root}/.extracted")
	set(_patterns "${_ov_name}/runtime/include/openvino/c/*" "${_ov_name}/docs/licensing/*"
		"${_ov_name}/runtime/3rdparty/tbb/TBB-LICENSE")
	foreach(_f IN LISTS _ov_libs)
		list(APPEND _patterns "${_ov_bin}/${_f}")
	endforeach()
	foreach(_f IN LISTS _ov_tbb_libs)
		list(APPEND _patterns "${_ov_tbb}/${_f}")
	endforeach()
	file(ARCHIVE_EXTRACT INPUT "${_ov_archive}" DESTINATION "${_ov_root}" PATTERNS ${_patterns})
	file(TOUCH "${_ov_root}/.extracted")
endif()

set(OPENCONV_OV_INCLUDE_DIR "${_ov_root}/${_ov_name}/runtime/include")
set(OPENCONV_OV_RUNTIME_FILES "")
# Staged flat under the names the loader maps.
set(_stage "${_ov_root}/runtime")
foreach(_f IN LISTS _ov_libs _ov_tbb_libs)
	list(FIND _ov_libs "${_f}" _in_bin)
	if(_in_bin GREATER -1)
		set(_src "${_ov_root}/${_ov_bin}/${_f}")
	else()
		set(_src "${_ov_root}/${_ov_tbb}/${_f}")
	endif()
	set(_as "${_f}")
	if(NOT WIN32)
		# libX.so.<version> -> libX.so.<soname suffix>; libtbb.so.12.13 -> libtbb.so.12, …
		string(REGEX REPLACE "\\.so\\.${OPENCONV_OV_VERSION}$" ".so.${OPENCONV_OV_SOVERSION}" _as "${_as}")
		string(REGEX REPLACE "^(lib(tbb|tbbmalloc|tbbbind_2_5|hwloc)\\.so\\.[0-9]+)\\.[0-9.]+$" "\\1" _as "${_as}")
	endif()
	if(NOT EXISTS "${_src}")
		message(FATAL_ERROR "OpenVINO file missing from the archive: ${_src}")
	endif()
	configure_file("${_src}" "${_stage}/${_as}" COPYONLY)
	list(APPEND OPENCONV_OV_RUNTIME_FILES "${_stage}/${_as}")
endforeach()
# Every licence text the archive carries (docs/licensing/*), prefixed, plus oneTBB's.
set(OPENCONV_OV_LICENSE_FILES "")
set(OPENCONV_OV_LICENSE_NAMES "")
file(GLOB _ov_lic "${_ov_root}/${_ov_name}/docs/licensing/*")
foreach(_f IN LISTS _ov_lic)
	if(NOT IS_DIRECTORY "${_f}")
		get_filename_component(_n "${_f}" NAME)
		list(APPEND OPENCONV_OV_LICENSE_FILES "${_f}")
		list(APPEND OPENCONV_OV_LICENSE_NAMES "openvino-${_n}")
	endif()
endforeach()
list(APPEND OPENCONV_OV_LICENSE_FILES "${_ov_root}/${_ov_name}/runtime/3rdparty/tbb/TBB-LICENSE")
list(APPEND OPENCONV_OV_LICENSE_NAMES "oneTBB-LICENSE.txt")
if(NOT EXISTS "${OPENCONV_OV_INCLUDE_DIR}/openvino/c/openvino.h")
	message(FATAL_ERROR "OpenVINO C API headers missing under ${OPENCONV_OV_INCLUDE_DIR}")
endif()
message(STATUS "OpenVINO ${OPENCONV_OV_VERSION}: ${_stage}")
