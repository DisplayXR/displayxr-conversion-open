// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  OS layer, see oc_platform.h.
 */

#ifndef _WIN32
#define _GNU_SOURCE // dladdr / Dl_info (glibc)
#endif

#include "oc_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <dlfcn.h>
#include <sys/stat.h>
#include <time.h>
#endif

void
oc_copy_str(char *dst, size_t cap, const char *src)
{
	if (dst == NULL || cap == 0) {
		return;
	}
	if (src == NULL) {
		dst[0] = '\0';
		return;
	}
	size_t n = strlen(src);
	if (n >= cap) {
		n = cap - 1;
	}
	memcpy(dst, src, n);
	dst[n] = '\0';
}

#ifdef _WIN32

void
oc_mutex_init(oc_mutex *m)
{
	InitializeSRWLock(m);
}
void
oc_lock(oc_mutex *m)
{
	AcquireSRWLockExclusive(m);
}
void
oc_unlock(oc_mutex *m)
{
	ReleaseSRWLockExclusive(m);
}
void
oc_mutex_destroy(oc_mutex *m)
{
	(void)m;
}
void
oc_sleep_ms(uint32_t ms)
{
	Sleep(ms);
}
uint64_t
oc_now_ms(void)
{
	return (uint64_t)GetTickCount64();
}

static bool
to_wide(const char *s, wchar_t *w, int cap)
{
	return MultiByteToWideChar(CP_UTF8, 0, s, -1, w, cap) != 0;
}

bool
oc_module_dir(char *buf, size_t buf_size)
{
	HMODULE self = NULL;
	if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                        (LPCWSTR)(void *)&oc_module_dir, &self)) {
		return false;
	}
	wchar_t w[MAX_PATH * 2];
	const DWORD n = GetModuleFileNameW(self, w, (DWORD)(sizeof(w) / sizeof(w[0])));
	if (n == 0 || n >= sizeof(w) / sizeof(w[0])) {
		return false;
	}
	wchar_t *slash = wcsrchr(w, L'\\');
	if (slash == NULL) {
		return false;
	}
	*slash = L'\0';
	return WideCharToMultiByte(CP_UTF8, 0, w, -1, buf, (int)buf_size, NULL, NULL) != 0;
}

bool
oc_file_exists(const char *path)
{
	wchar_t w[MAX_PATH * 2];
	if (path == NULL || !to_wide(path, w, (int)(sizeof(w) / sizeof(w[0])))) {
		return false;
	}
	const DWORD a = GetFileAttributesW(w);
	return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool
oc_make_dir(const char *path)
{
	wchar_t w[MAX_PATH * 2];
	if (path == NULL || !to_wide(path, w, (int)(sizeof(w) / sizeof(w[0])))) {
		return false;
	}
	(void)CreateDirectoryW(w, NULL);
	const DWORD a = GetFileAttributesW(w);
	return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

void *
oc_lib_open(const char *path, char *err, size_t err_size)
{
	wchar_t w[MAX_PATH * 2];
	if (path == NULL || !to_wide(path, w, (int)(sizeof(w) / sizeof(w[0])))) {
		snprintf(err, err_size, "bad path");
		return NULL;
	}
	for (wchar_t *c = w; *c != 0; c++) {
		if (*c == L'/') {
			*c = L'\\'; // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs backslashes
		}
	}
	HMODULE m = LoadLibraryExW(w, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
	if (m == NULL) {
		snprintf(err, err_size, "LoadLibraryExW failed (error %lu)", (unsigned long)GetLastError());
	}
	return (void *)m;
}

void *
oc_lib_sym(void *lib, const char *name)
{
	return lib != NULL ? (void *)GetProcAddress((HMODULE)lib, name) : NULL;
}

#else // POSIX

void
oc_mutex_init(oc_mutex *m)
{
	pthread_mutex_init(m, NULL);
}
void
oc_lock(oc_mutex *m)
{
	pthread_mutex_lock(m);
}
void
oc_unlock(oc_mutex *m)
{
	pthread_mutex_unlock(m);
}
void
oc_mutex_destroy(oc_mutex *m)
{
	pthread_mutex_destroy(m);
}
void
oc_sleep_ms(uint32_t ms)
{
	struct timespec ts;
	ts.tv_sec = (time_t)(ms / 1000u);
	ts.tv_nsec = (long)(ms % 1000u) * 1000000L;
	nanosleep(&ts, NULL);
}
uint64_t
oc_now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

bool
oc_module_dir(char *buf, size_t buf_size)
{
	Dl_info info;
	if (dladdr((void *)&oc_module_dir, &info) == 0 || info.dli_fname == NULL) {
		return false;
	}
	char full[4096];
	if (realpath(info.dli_fname, full) == NULL) {
		oc_copy_str(full, sizeof(full), info.dli_fname);
	}
	char *slash = strrchr(full, '/');
	if (slash == NULL) {
		return false;
	}
	*slash = '\0';
	oc_copy_str(buf, buf_size, full);
	return strlen(full) < buf_size;
}

bool
oc_file_exists(const char *path)
{
	struct stat st;
	return path != NULL && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

bool
oc_make_dir(const char *path)
{
	if (path == NULL) {
		return false;
	}
	(void)mkdir(path, 0700);
	struct stat st;
	return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

void *
oc_lib_open(const char *path, char *err, size_t err_size)
{
	void *m = path != NULL ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : NULL;
	if (m == NULL) {
		const char *e = dlerror();
		snprintf(err, err_size, "dlopen failed: %s", e != NULL ? e : "bad path");
	}
	return m;
}

void *
oc_lib_sym(void *lib, const char *name)
{
	return lib != NULL ? dlsym(lib, name) : NULL;
}

#endif
