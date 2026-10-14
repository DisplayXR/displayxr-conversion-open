# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# ONNX Runtime (and, on Windows, DirectML) as PINNED release archives, fetched
# at configure time and verified by SHA-256 before anything is extracted.
# Only headers are used at build time: the provider loads the library at run
# time by absolute path from its own directory, so nothing links against it.
#
# Outputs:
#   OPENCONV_ORT_INCLUDE_DIR   onnxruntime_c_api.h (+ dml_provider_factory.h on Windows)
#   OPENCONV_ORT_RUNTIME_FILES files to place next to the provider binary
#   OPENCONV_ORT_LICENSE_FILES licence texts shipped with them
#
# -DOPENCONV_DOWNLOAD_DIR=<dir> keeps the archives across build trees (CI caches it).

set(OPENCONV_ORT_VERSION 1.24.4)
set(OPENCONV_DIRECTML_VERSION 1.15.4)

set(OPENCONV_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_downloads" CACHE PATH "Where pinned archives are downloaded")
set(_ort_root "${CMAKE_BINARY_DIR}/_deps/onnxruntime-${OPENCONV_ORT_VERSION}")

# file(DOWNLOAD) skips the transfer when the file is already there with the
# expected hash, and fails the configure on a mismatch.
function(openconv_fetch url file sha256)
	set(_dst "${OPENCONV_DOWNLOAD_DIR}/${file}")
	if(EXISTS "${_dst}")
		file(SHA256 "${_dst}" _have)
		if(_have STREQUAL "${sha256}")
			return()
		endif()
		file(REMOVE "${_dst}")
	endif()
	message(STATUS "Downloading ${file}")
	file(DOWNLOAD "${url}" "${_dst}" EXPECTED_HASH SHA256=${sha256} TLS_VERIFY ON STATUS _st)
	list(GET _st 0 _code)
	if(NOT _code EQUAL 0)
		file(REMOVE "${_dst}")
		message(FATAL_ERROR "Download of ${url} failed: ${_st}")
	endif()
endfunction()

if(WIN32)
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
		set(_rid win-arm64)
		set(_dml_arch arm64-win)
	else()
		set(_rid win-x64)
		set(_dml_arch x64-win)
	endif()
	# ONNX Runtime with the DirectML execution provider (NuGet package = zip).
	set(_ort_pkg "microsoft.ml.onnxruntime.directml.${OPENCONV_ORT_VERSION}.nupkg")
	openconv_fetch(
		"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/${OPENCONV_ORT_VERSION}/${_ort_pkg}"
		"${_ort_pkg}" 57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681)
	# The DirectML redistributable ONNX Runtime's DirectML EP requires.
	set(_dml_pkg "microsoft.ai.directml.${OPENCONV_DIRECTML_VERSION}.nupkg")
	openconv_fetch(
		"https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/${OPENCONV_DIRECTML_VERSION}/${_dml_pkg}"
		"${_dml_pkg}" 4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9)
	if(NOT EXISTS "${_ort_root}/.extracted")
		file(ARCHIVE_EXTRACT INPUT "${OPENCONV_DOWNLOAD_DIR}/${_ort_pkg}" DESTINATION "${_ort_root}/ort"
			PATTERNS "build/native/include/*" "runtimes/${_rid}/native/onnxruntime.dll" "LICENSE"
			"ThirdPartyNotices.txt")
		file(ARCHIVE_EXTRACT INPUT "${OPENCONV_DOWNLOAD_DIR}/${_dml_pkg}" DESTINATION "${_ort_root}/dml"
			PATTERNS "bin/${_dml_arch}/DirectML.dll" "LICENSE.txt" "ThirdPartyNotices.txt")
		file(TOUCH "${_ort_root}/.extracted")
	endif()
	set(OPENCONV_ORT_INCLUDE_DIR "${_ort_root}/ort/build/native/include")
	set(OPENCONV_ORT_RUNTIME_FILES
		"${_ort_root}/ort/runtimes/${_rid}/native/onnxruntime.dll"
		"${_ort_root}/dml/bin/${_dml_arch}/DirectML.dll")
	set(OPENCONV_ORT_LICENSE_FILES "${_ort_root}/ort/LICENSE" "${_ort_root}/ort/ThirdPartyNotices.txt"
		"${_ort_root}/dml/LICENSE.txt")
	set(OPENCONV_ORT_LICENSE_NAMES "onnxruntime-LICENSE.txt" "onnxruntime-ThirdPartyNotices.txt"
		"DirectML-LICENSE.txt")
elseif(APPLE)
	if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
		message(FATAL_ERROR "ONNX Runtime ${OPENCONV_ORT_VERSION} ships macOS for arm64 only")
	endif()
	set(_name "onnxruntime-osx-arm64-${OPENCONV_ORT_VERSION}")
	openconv_fetch("https://github.com/microsoft/onnxruntime/releases/download/v${OPENCONV_ORT_VERSION}/${_name}.tgz"
		"${_name}.tgz" 93787795f47e1eee369182e43ed51b9e5da0878ab0346aecf4258979b8bba989)
	set(_lib "libonnxruntime.${OPENCONV_ORT_VERSION}.dylib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
		set(_name "onnxruntime-linux-aarch64-${OPENCONV_ORT_VERSION}")
		set(_sha 866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e)
	else()
		set(_name "onnxruntime-linux-x64-${OPENCONV_ORT_VERSION}")
		set(_sha 3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747)
	endif()
	openconv_fetch("https://github.com/microsoft/onnxruntime/releases/download/v${OPENCONV_ORT_VERSION}/${_name}.tgz"
		"${_name}.tgz" ${_sha})
	set(_lib "libonnxruntime.so.${OPENCONV_ORT_VERSION}")
else()
	message(FATAL_ERROR "Unsupported platform for ONNX Runtime: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT WIN32)
	if(NOT EXISTS "${_ort_root}/.extracted")
		file(ARCHIVE_EXTRACT INPUT "${OPENCONV_DOWNLOAD_DIR}/${_name}.tgz" DESTINATION "${_ort_root}"
			PATTERNS "${_name}/include/*" "${_name}/lib/${_lib}" "${_name}/LICENSE" "${_name}/ThirdPartyNotices.txt")
		file(TOUCH "${_ort_root}/.extracted")
	endif()
	set(OPENCONV_ORT_INCLUDE_DIR "${_ort_root}/${_name}/include")
	# Next to the provider under the name it loads (the SONAME on Linux).
	if(APPLE)
		set(_as "libonnxruntime.dylib")
	else()
		set(_as "libonnxruntime.so.1")
	endif()
	configure_file("${_ort_root}/${_name}/lib/${_lib}" "${_ort_root}/runtime/${_as}" COPYONLY)
	set(OPENCONV_ORT_RUNTIME_FILES "${_ort_root}/runtime/${_as}")
	set(OPENCONV_ORT_LICENSE_FILES "${_ort_root}/${_name}/LICENSE")
	set(OPENCONV_ORT_LICENSE_NAMES "onnxruntime-LICENSE.txt")
	if(EXISTS "${_ort_root}/${_name}/ThirdPartyNotices.txt")
		list(APPEND OPENCONV_ORT_LICENSE_FILES "${_ort_root}/${_name}/ThirdPartyNotices.txt")
		list(APPEND OPENCONV_ORT_LICENSE_NAMES "onnxruntime-ThirdPartyNotices.txt")
	endif()
endif()

if(NOT EXISTS "${OPENCONV_ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
	message(FATAL_ERROR "ONNX Runtime headers missing under ${OPENCONV_ORT_INCLUDE_DIR}")
endif()
foreach(_f IN LISTS OPENCONV_ORT_RUNTIME_FILES)
	if(NOT EXISTS "${_f}")
		message(FATAL_ERROR "ONNX Runtime file missing: ${_f}")
	endif()
endforeach()
message(STATUS "ONNX Runtime ${OPENCONV_ORT_VERSION}: ${OPENCONV_ORT_RUNTIME_FILES}")
