// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  The provider's whole OS layer: a mutex, a sleep, a clock, the
 *         provider's own directory, and loading a dependency by absolute path.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
typedef SRWLOCK oc_mutex;
#define OC_PATH_SEP '\\'
#else
typedef pthread_mutex_t oc_mutex;
#define OC_PATH_SEP '/'
#endif

void
oc_mutex_init(oc_mutex *m);
void
oc_lock(oc_mutex *m);
void
oc_unlock(oc_mutex *m);
void
oc_mutex_destroy(oc_mutex *m);

void
oc_sleep_ms(uint32_t ms);

uint64_t
oc_now_ms(void);

/*! Directory of the module that contains this code (UTF-8, no trailing
 *  separator). false when it cannot be determined. */
bool
oc_module_dir(char *buf, size_t buf_size);

/*! True when the file exists (UTF-8 path). */
bool
oc_file_exists(const char *path);

/*! Create the directory @p path (its parent must exist; owner-only on POSIX).
 *  True when it exists afterwards. */
bool
oc_make_dir(const char *path);

/*! Load a shared library by ABSOLUTE path; its own dependencies resolve from
 *  its directory first (Windows: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, with
 *  backslashes). Never via PATH. */
void *
oc_lib_open(const char *path, char *err, size_t err_size);

void *
oc_lib_sym(void *lib, const char *name);

/*! Copy at most cap-1 bytes and terminate. */
void
oc_copy_str(char *dst, size_t cap, const char *src);

#ifdef __cplusplus
}
#endif
