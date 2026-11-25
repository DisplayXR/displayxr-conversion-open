# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# libwebp (BSD-3-Clause) as a PINNED, hash-checked release archive, compiled from
# its sources into a static library linked into the provider (the SOG writer's
# lossless WebP planes). Its own CMake project is not used: it would add install
# rules and tools to this project's install. No threads (WEBP_USE_THREAD unset),
# no config.h: the encoder is single-threaded, and its SIMD paths are picked by
# the compiler's predefined macros and a run-time CPU check, as upstream does.
#
# Outputs:
#   target openconv_webp       static, position-independent, hidden symbols
#   OPENCONV_WEBP_LICENSE_FILE libwebp's COPYING (shipped as libwebp-COPYING.txt)

set(OPENCONV_WEBP_VERSION 1.5.0)
set(_webp_pkg "libwebp-${OPENCONV_WEBP_VERSION}.tar.gz")
openconv_fetch("https://storage.googleapis.com/downloads.webmproject.org/releases/webp/${_webp_pkg}"
	"${_webp_pkg}" 7d6fab70cf844bf6769077bd5d7a74893f8ffd4dfb42861745750c63c2a5c92c)
set(_webp_root "${CMAKE_BINARY_DIR}/_deps/libwebp-${OPENCONV_WEBP_VERSION}")
if(NOT EXISTS "${_webp_root}/.extracted")
	file(ARCHIVE_EXTRACT INPUT "${OPENCONV_DOWNLOAD_DIR}/${_webp_pkg}" DESTINATION "${_webp_root}")
	file(TOUCH "${_webp_root}/.extracted")
endif()
set(_webp_src "${_webp_root}/libwebp-${OPENCONV_WEBP_VERSION}")
file(GLOB _webp_sources
	"${_webp_src}/src/dec/*.c" "${_webp_src}/src/enc/*.c" "${_webp_src}/src/dsp/*.c"
	"${_webp_src}/src/utils/*.c" "${_webp_src}/sharpyuv/*.c")
list(SORT _webp_sources)
add_library(openconv_webp STATIC ${_webp_sources})
target_include_directories(openconv_webp SYSTEM PUBLIC "${_webp_src}/src" "${_webp_src}")
set_target_properties(openconv_webp PROPERTIES POSITION_INDEPENDENT_CODE ON C_VISIBILITY_PRESET hidden)
if(MSVC)
	set_property(TARGET openconv_webp PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	target_compile_options(openconv_webp PRIVATE /W0)
	target_compile_definitions(openconv_webp PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
	target_compile_options(openconv_webp PRIVATE -w)
	if(NOT WIN32)
		target_link_libraries(openconv_webp PRIVATE m)
	endif()
endif()
set(OPENCONV_WEBP_LICENSE_FILE "${_webp_src}/COPYING")
