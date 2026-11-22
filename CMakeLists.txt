# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# DisplayXR open-model conversion provider: DisplayXR-OpenConversion.{dll,so,dylib}
# (one export, xrtConversionPluginNegotiate), ONNX Runtime next to it, OpenVINO in
# openvino/ (Linux x86_64, Windows x64), the default model pack manifest, and a
# fake host that drives it.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# -DOPENCONV_TEST_MODELS_DIR=<dir with <sha256>.onnx blobs> enables the
# inference tests (scripts/fetch_models.py fills such a directory).
# -DOPENCONV_TEST_SPLAT_MODEL=<a splats-mono .onnx> enables the real splat-model test.

cmake_minimum_required(VERSION 3.22)
project(displayxr_conversion_open VERSION 0.2.0 LANGUAGES C)

# The runtime release the vendored include/xrt/ headers match (README "The ABI it builds
# against"). Metadata only: nothing compiles against it. The runtime repository downstream-pins.json
# tracks this literal and opens a repin PR only when the conversion-provider ABI major changes;
# re-vendor the headers in that PR.
set(DXR_RUNTIME_GIT_TAG "v2.69.4")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
# gnu11 on GCC / Clang: POSIX (nanosleep, realpath, dladdr) is visible without feature macros.
set(CMAKE_C_EXTENSIONS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(OPENCONV_WERROR "Treat compiler warnings as errors (CI turns this on)" OFF)
option(OPENCONV_BUILD_TESTS "Build the fake host and register the tests" ON)
set(OPENCONV_TEST_MODELS_DIR "" CACHE PATH "Directory holding <sha256>.onnx blobs for the inference tests")
option(OPENCONV_TEST_GPU "Also run the inference tests on the default execution provider (DirectML on Windows)" OFF)
set(OPENCONV_TEST_SPLAT_MODEL "" CACHE FILEPATH "A splats-mono ONNX model for the real splat test (never committed)")
set(OPENCONV_TEST_SPLAT_IMAGE "" CACHE FILEPATH "A photo (PNG / PPM / raw RGBA) for the real splat test")
if((CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64") OR
   (WIN32 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64"))
	set(_ov_default ON)
else()
	set(_ov_default OFF)
endif()
option(OPENCONV_OPENVINO "Ship OpenVINO (openvino/) for the splat capability" ${_ov_default})

set(OPENCONV_PROVIDER DisplayXR-OpenConversion)
set(OPENCONV_PACK "${CMAKE_CURRENT_SOURCE_DIR}/packs/dxr-open-default.models.json")

include(cmake/OnnxRuntime.cmake)
if(OPENCONV_OPENVINO)
	include(cmake/OpenVINO.cmake)
endif()

# The build stamp's source revision (xrt_conv_splat_result::build_provider).
set(OPENCONV_BUILD_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
	execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty --tags
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE _rev OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_QUIET RESULT_VARIABLE _rev_rc)
	if(_rev_rc EQUAL 0 AND _rev)
		set(OPENCONV_BUILD_COMMIT "${_rev}")
	endif()
endif()

function(openconv_warnings target)
	if(MSVC)
		target_compile_options(${target} PRIVATE /W4 /utf-8 /external:W0)
		target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
		if(OPENCONV_WERROR)
			target_compile_options(${target} PRIVATE /WX)
		endif()
	else()
		target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
		if(OPENCONV_WERROR)
			target_compile_options(${target} PRIVATE -Werror)
		endif()
	endif()
endfunction()

# The provider: a loadable module exporting exactly one symbol. ONNX Runtime is
# NOT linked: it is loaded at run time by absolute path from this directory.
add_library(${OPENCONV_PROVIDER} MODULE
	src/open_conversion.c
	src/oc_image.c
	src/oc_json.c
	src/oc_ort.c
	src/oc_ov.c
	src/oc_pack.c
	src/oc_platform.c
	src/oc_splat.c)
target_include_directories(${OPENCONV_PROVIDER} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${OPENCONV_PROVIDER} SYSTEM PRIVATE ${OPENCONV_ORT_INCLUDE_DIR})
target_compile_definitions(${OPENCONV_PROVIDER} PRIVATE OC_BUILD_COMMIT="${OPENCONV_BUILD_COMMIT}")
if(OPENCONV_OPENVINO)
	target_include_directories(${OPENCONV_PROVIDER} SYSTEM PRIVATE ${OPENCONV_OV_INCLUDE_DIR})
	target_compile_definitions(${OPENCONV_PROVIDER} PRIVATE OC_OV_SOVERSION="${OPENCONV_OV_SOVERSION}")
else()
	target_compile_definitions(${OPENCONV_PROVIDER} PRIVATE OC_OV_DISABLED)
endif()
set_target_properties(${OPENCONV_PROVIDER} PROPERTIES PREFIX "" C_VISIBILITY_PRESET hidden)
if(APPLE)
	set_target_properties(${OPENCONV_PROVIDER} PROPERTIES SUFFIX ".dylib")
endif()
if(WIN32)
	target_link_libraries(${OPENCONV_PROVIDER} PRIVATE dxgi dxguid)
else()
	find_package(Threads REQUIRED)
	target_link_libraries(${OPENCONV_PROVIDER} PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)
endif()
if(MSVC)
	# A provider DLL loads into a process whose CRT it does not control.
	set_property(TARGET ${OPENCONV_PROVIDER} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
openconv_warnings(${OPENCONV_PROVIDER})

# Ship-together files next to the binary: ONNX Runtime (+ DirectML) and the packs. Every
# packs/*.models.json ships (the provider loads every *.models.json next to it): a
# research-only model is its own pack, never the default one, and never pre-fetched.
file(GLOB OPENCONV_PACKS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/packs/*.models.json")
set(_ship ${OPENCONV_ORT_RUNTIME_FILES} ${OPENCONV_PACKS})
add_custom_command(TARGET ${OPENCONV_PROVIDER} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_ship} $<TARGET_FILE_DIR:${OPENCONV_PROVIDER}>
	VERBATIM)
if(OPENCONV_OPENVINO)
	add_custom_command(TARGET ${OPENCONV_PROVIDER} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${OPENCONV_PROVIDER}>/openvino
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPENCONV_OV_RUNTIME_FILES}
			$<TARGET_FILE_DIR:${OPENCONV_PROVIDER}>/openvino
		VERBATIM)
endif()

install(TARGETS ${OPENCONV_PROVIDER} LIBRARY DESTINATION . RUNTIME DESTINATION .)
install(FILES ${_ship} DESTINATION .)
install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION licenses)
if(OPENCONV_OPENVINO)
	install(FILES ${OPENCONV_OV_RUNTIME_FILES} DESTINATION openvino)
	list(APPEND OPENCONV_ORT_LICENSE_FILES ${OPENCONV_OV_LICENSE_FILES})
	list(APPEND OPENCONV_ORT_LICENSE_NAMES ${OPENCONV_OV_LICENSE_NAMES})
endif()
list(LENGTH OPENCONV_ORT_LICENSE_FILES _n)
math(EXPR _last "${_n} - 1")
foreach(_i RANGE ${_last})
	list(GET OPENCONV_ORT_LICENSE_FILES ${_i} _f)
	list(GET OPENCONV_ORT_LICENSE_NAMES ${_i} _as)
	install(FILES ${_f} DESTINATION licenses RENAME ${_as})
endforeach()

if(OPENCONV_BUILD_TESTS)
	enable_testing()

	add_executable(open-conversion-fakehost
		tools/fakehost.c
		tools/sha256.c
		src/oc_json.c
		src/oc_pack.c
		src/oc_platform.c)
	if(OPENCONV_OPENVINO)
		target_compile_definitions(open-conversion-fakehost PRIVATE OPENCONV_HAVE_OPENVINO=1)
	endif()
	target_include_directories(open-conversion-fakehost PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tools)
	if(NOT WIN32)
		target_link_libraries(open-conversion-fakehost PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)
	endif()
	openconv_warnings(open-conversion-fakehost)
	add_dependencies(open-conversion-fakehost ${OPENCONV_PROVIDER})

	# The vendored IPC wire header stays compilable against the vendored ABI header.
	add_library(openconv-header-check OBJECT tests/header_check.c)
	target_include_directories(openconv-header-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

	set(_bin $<TARGET_FILE:${OPENCONV_PROVIDER}>)
	set(_host $<TARGET_FILE:open-conversion-fakehost>)
	set(_stores "${CMAKE_BINARY_DIR}/test-stores")
	file(MAKE_DIRECTORY "${_stores}")
	if(OPENCONV_TEST_MODELS_DIR)
		file(TO_CMAKE_PATH "${OPENCONV_TEST_MODELS_DIR}" _models)
	else()
		set(_models "${CMAKE_BINARY_DIR}/no-test-models")
	endif()
	if(_models MATCHES "^/")
		set(_base "file://${_models}")
	else()
		set(_base "file:///${_models}")
	endif()

	# Negotiate + describe + cold readiness, no model needed.
	add_test(NAME describe COMMAND ${_host} ${_bin} --pack ${OPENCONV_PACK} --store ${_stores}/describe
		--clean-store --no-warm --expect-licence permissive)
	# The inference runtime missing -> ABSENT on every cap (falls through to the next provider).
	add_test(NAME absent-runtime COMMAND ${_host} ${_bin} --set ONNXRUNTIME=${CMAKE_BINARY_DIR}/missing/onnxruntime
		--pack ${OPENCONV_PACK} --store ${_stores}/absent --no-warm --expect absent)
	# ONNX Runtime present but older than the API the provider needs -> FAILED(VERSION), never FAILED(MODEL).
	add_library(openconv-fake-old-ort MODULE tests/fake_old_ort.c)
	set_target_properties(openconv-fake-old-ort PROPERTIES PREFIX "" C_VISIBILITY_PRESET hidden)
	add_dependencies(open-conversion-fakehost openconv-fake-old-ort)
	add_test(NAME runtime-too-old COMMAND ${_host} ${_bin} --set ONNXRUNTIME=$<TARGET_FILE:openconv-fake-old-ort>
		--pack ${OPENCONV_PACK} --store ${_stores}/old --no-warm --expect version)
	# A pack whose model is not commercial-clean -> RESEARCH_ONLY, commercial_ok 0.
	add_test(NAME research-only COMMAND ${_host} ${_bin} --pack ${CMAKE_CURRENT_SOURCE_DIR}/tests/packs/research.models.json
		--store ${_stores}/research --no-warm --expect-licence research)
	# Blob missing and the download forbidden by policy -> ABSENT after warm.
	add_test(NAME download-forbidden COMMAND ${_host} ${_bin} --pack ${OPENCONV_PACK} --store ${_stores}/policy
		--clean-store --no-download --caps mono --expect policy --set EP=cpu)
	# DOWNLOADING -> WARMING -> READY through a file:// pack, then a CPU-EP conversion of a synthetic scene.
	add_test(NAME download-ready-mono COMMAND ${_host} ${_bin} --pack ${OPENCONV_PACK} --blob-base ${_base}
		--store ${_stores}/mono --clean-store --chunk-ms 25 --expect-download --caps mono --set EP=cpu)
	add_test(NAME download-ready-video COMMAND ${_host} ${_bin} --pack ${OPENCONV_PACK} --blob-base ${_base}
		--store ${_stores}/video --clean-store --chunk-ms 25 --expect-download --caps video --frames 4 --set EP=cpu)
	set_tests_properties(download-ready-mono download-ready-video PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
	if(OPENCONV_TEST_GPU)
		add_test(NAME default-ep COMMAND ${_host} ${_bin} --pack ${OPENCONV_PACK} --blob-base ${_base}
			--store ${_stores}/gpu --frames 4)
		set_tests_properties(default-ep PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
	endif()

	# The splat capability without a real model: a tiny synthetic model with the
	# same I/O contract (tests/make_tiny_splat_model.py, needs the `onnx` Python
	# package; skipped without it), run on the CPU — ONNX Runtime, and OpenVINO's
	# CPU device where the build ships OpenVINO. No GPU, no model download.
	find_package(Python3 COMPONENTS Interpreter QUIET)
	set(_tiny "${CMAKE_BINARY_DIR}/tiny-splat")
	if(Python3_FOUND)
		add_test(NAME splats-tiny-model COMMAND Python3::Interpreter -I
			${CMAKE_CURRENT_SOURCE_DIR}/tests/make_tiny_splat_model.py ${_tiny})
		set_tests_properties(splats-tiny-model PROPERTIES FIXTURES_SETUP tiny_splat SKIP_RETURN_CODE 77)
		add_test(NAME splats-cpu COMMAND ${_host} ${_bin} --pack ${_tiny}/tiny.models.json
			--blob-base file://${_tiny} --store ${_stores}/splats-cpu --clean-store --caps splats
			--set EP=cpu --allow-research --splat-check ${_tiny}/expect.json)
		set_tests_properties(splats-cpu PROPERTIES FIXTURES_REQUIRED tiny_splat SKIP_RETURN_CODE 77)
		if(OPENCONV_OPENVINO)
			add_test(NAME splats-openvino-cpu COMMAND ${_host} ${_bin} --pack ${_tiny}/tiny.models.json
				--blob-base file://${_tiny} --store ${_stores}/splats-ov --clean-store --caps splats
				--set EP=openvino --set OV_DEVICE=cpu --expect-backend openvino-cpu --allow-research
				--splat-check ${_tiny}/expect.json --scratch ${_stores}/splats-ov-scratch)
			set_tests_properties(splats-openvino-cpu PROPERTIES FIXTURES_REQUIRED tiny_splat SKIP_RETURN_CODE 77)
		endif()
	endif()
	# The real model, on the default backend (OpenVINO GPU where there is one): only
	# with -DOPENCONV_TEST_SPLAT_MODEL (the user's own export; nothing is fetched).
	if(OPENCONV_TEST_SPLAT_MODEL AND OPENCONV_TEST_GPU)
		# The fake host's store fetches <base>/<sha256>.onnx: link the model under that name.
		file(SHA256 "${OPENCONV_TEST_SPLAT_MODEL}" _splat_sha)
		file(SIZE "${OPENCONV_TEST_SPLAT_MODEL}" _splat_size)
		set(_splat_link_dir "${CMAKE_BINARY_DIR}/splat-model-link")
		file(MAKE_DIRECTORY "${_splat_link_dir}")
		file(CREATE_LINK "${OPENCONV_TEST_SPLAT_MODEL}" "${_splat_link_dir}/${_splat_sha}.onnx" COPY_ON_ERROR SYMBOLIC)
		configure_file(tests/packs/splats-real.models.json.in ${CMAKE_BINARY_DIR}/splats-real.models.json @ONLY)
		add_test(NAME splats-default-ep COMMAND ${_host} ${_bin} --pack ${CMAKE_BINARY_DIR}/splats-real.models.json
			--store ${_stores}/splats-real --caps splats --allow-research
			--splat-image "${OPENCONV_TEST_SPLAT_IMAGE}" --scratch ${_stores}/splats-real-scratch)
		set_tests_properties(splats-default-ep PROPERTIES TIMEOUT 1800)
	endif()

	if(Python3_FOUND)
		add_test(NAME check-model-pack COMMAND Python3::Interpreter -I
			${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_model_pack.py ${OPENCONV_PACK})
		add_test(NAME check-model-pack-research COMMAND Python3::Interpreter -I
			${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_model_pack.py --allow-research
			${CMAKE_CURRENT_SOURCE_DIR}/tests/packs/research.models.json)
		add_test(NAME check-model-pack-sharp COMMAND Python3::Interpreter -I
			${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_model_pack.py --allow-research
			${CMAKE_CURRENT_SOURCE_DIR}/packs/dxr-open-sharp.models.json)
	endif()
endif()
