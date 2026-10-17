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
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <ftw.h>
#include <signal.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
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
	return oc_dir_exists(path);
}

bool
oc_dir_exists(const char *path)
{
	wchar_t w[MAX_PATH * 2];
	if (path == NULL || !to_wide(path, w, (int)(sizeof(w) / sizeof(w[0])))) {
		return false;
	}
	const DWORD a = GetFileAttributesW(w);
	return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

bool
oc_remove_tree(const char *path)
{
	(void)path;
	return false;
}

bool
oc_rename(const char *from, const char *to)
{
	wchar_t wf[MAX_PATH * 2], wt[MAX_PATH * 2];
	return from != NULL && to != NULL && to_wide(from, wf, (int)(sizeof(wf) / sizeof(wf[0]))) &&
	       to_wide(to, wt, (int)(sizeof(wt) / sizeof(wt[0]))) && MoveFileExW(wf, wt, 0) != 0;
}

bool
oc_touch(const char *path)
{
	wchar_t w[MAX_PATH * 2];
	if (path == NULL || !to_wide(path, w, (int)(sizeof(w) / sizeof(w[0])))) {
		return false;
	}
	HANDLE h = CreateFileW(w, GENERIC_WRITE, 0, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (h == INVALID_HANDLE_VALUE) {
		return false;
	}
	CloseHandle(h);
	return true;
}

uint64_t
oc_process_id(void)
{
	return (uint64_t)GetCurrentProcessId();
}

void
oc_sweep_stale(const char *dir, const char *prefix)
{
	(void)dir;
	(void)prefix;
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
	return oc_dir_exists(path);
}

bool
oc_dir_exists(const char *path)
{
	struct stat st;
	return path != NULL && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int
remove_entry(const char *path, const struct stat *st, int flag, struct FTW *ftw)
{
	(void)st;
	(void)flag;
	(void)ftw;
	return remove(path) == 0 ? 0 : -1;
}

bool
oc_remove_tree(const char *path)
{
	struct stat st;
	if (path == NULL || lstat(path, &st) != 0) {
		return path != NULL; // already gone
	}
	if (!S_ISDIR(st.st_mode)) {
		return remove(path) == 0;
	}
	return nftw(path, remove_entry, 16, FTW_DEPTH | FTW_PHYS) == 0;
}

bool
oc_rename(const char *from, const char *to)
{
	return from != NULL && to != NULL && rename(from, to) == 0;
}

bool
oc_touch(const char *path)
{
	FILE *f = path != NULL ? fopen(path, "ab") : NULL;
	if (f == NULL) {
		return false;
	}
	return fclose(f) == 0;
}

uint64_t
oc_process_id(void)
{
	return (uint64_t)getpid();
}

void
oc_sweep_stale(const char *dir, const char *prefix)
{
	DIR *d = dir != NULL ? opendir(dir) : NULL;
	if (d == NULL) {
		return;
	}
	const size_t plen = strlen(prefix);
	struct dirent *e;
	while ((e = readdir(d)) != NULL) {
		if (strncmp(e->d_name, prefix, plen) != 0) {
			continue;
		}
		const char *dash = strrchr(e->d_name, '-');
		char *end = NULL;
		const long pid = dash != NULL ? strtol(dash + 1, &end, 10) : 0;
		if (pid <= 0 || end == NULL || *end != '\0' || (pid_t)pid == getpid()) {
			continue;
		}
		if (kill((pid_t)pid, 0) == 0 || errno != ESRCH) {
			continue; // alive (or not ours to judge)
		}
		char path[2048];
		snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
		(void)oc_remove_tree(path);
	}
	closedir(d);
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
