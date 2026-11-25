// Copyright 2026, The DisplayXR Project
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Session diagnostics parsing, see oc_profile.h.
 */

#include "oc_profile.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
copy_n(char *dst, size_t size, const char *src, size_t n)
{
	if (size == 0) {
		return;
	}
	if (n >= size) {
		n = size - 1;
	}
	memcpy(dst, src, n);
	dst[n] = '\0';
}

/*! A fused partition's op type carries its partition number ("DmlFusedNode_0_3"):
 *  drop trailing "_<digits>" groups so the partitions count as one operator. */
static void
normalize_op(char *op)
{
	size_t n = strlen(op);
	for (;;) {
		size_t i = n;
		while (i > 0 && isdigit((unsigned char)op[i - 1])) {
			i--;
		}
		if (i == n || i < 2 || op[i - 1] != '_') {
			break;
		}
		n = i - 1;
		op[n] = '\0';
	}
}

static void
place_add(struct oc_place_report *r, const char *ep, const char *op, uint32_t count)
{
	for (uint32_t i = 0; i < r->entry_count; i++) {
		if (strcmp(r->entries[i].ep, ep) == 0 && strcmp(r->entries[i].op, op) == 0) {
			r->entries[i].count += count;
			return;
		}
	}
	if (r->entry_count >= OC_PLACE_MAX) {
		r->dropped += count;
		return;
	}
	struct oc_place_entry *e = &r->entries[r->entry_count++];
	copy_n(e->ep, sizeof(e->ep), ep, strlen(ep));
	copy_n(e->op, sizeof(e->op), op, strlen(op));
	e->count = count;
}

bool
oc_place_feed(struct oc_place_report *r, const char *msg)
{
	if (r == NULL || msg == NULL) {
		return false;
	}
	const char *at = strstr(msg, "placed on [");
	if (at == NULL) {
		return false;
	}
	const char *ep0 = at + strlen("placed on [");
	const char *ep1 = strchr(ep0, ']');
	if (ep1 == NULL) {
		return false;
	}
	char ep[40];
	copy_n(ep, sizeof(ep), ep0, (size_t)(ep1 - ep0));
	const char *num = strstr(ep1, "Number of nodes: ");
	const unsigned long total = num != NULL ? strtoul(num + strlen("Number of nodes: "), NULL, 10) : 0ul;
	r->seen = true;
	// One "\n  Op (name)" line per node follows (absent in the one-EP form).
	uint32_t listed = 0;
	for (const char *line = strchr(ep1, '\n'); line != NULL; line = strchr(line + 1, '\n')) {
		const char *s = line + 1;
		while (*s == ' ') {
			s++;
		}
		size_t n = 0;
		while (s[n] != '\0' && s[n] != ' ' && s[n] != '\n' && s[n] != '(') {
			n++;
		}
		if (n == 0) {
			continue;
		}
		char op[48];
		copy_n(op, sizeof(op), s, n);
		normalize_op(op);
		place_add(r, ep, op, 1);
		listed++;
	}
	if (listed == 0 && total > 0) {
		place_add(r, ep, "(all)", (uint32_t)total);
	}
	return true;
}

uint32_t
oc_place_total(const struct oc_place_report *r, const char *ep)
{
	uint32_t n = 0;
	for (uint32_t i = 0; r != NULL && i < r->entry_count; i++) {
		if (ep == NULL || ep[0] == '\0' || strcmp(r->entries[i].ep, ep) == 0) {
			n += r->entries[i].count;
		}
	}
	return n;
}

static void
append(char *buf, size_t size, size_t *len, const char *fmt, ...)
{
	if (*len + 1 >= size) {
		return;
	}
	va_list ap;
	va_start(ap, fmt);
	const int w = vsnprintf(buf + *len, size - *len, fmt, ap);
	va_end(ap);
	if (w > 0) {
		*len += (size_t)w < size - *len ? (size_t)w : size - *len - 1;
	}
}

void
oc_place_format(const struct oc_place_report *r, uint32_t max_ops, char *buf, size_t size)
{
	size_t len = 0;
	buf[0] = '\0';
	if (r == NULL || !r->seen) {
		snprintf(buf, size, "no placement report (ONNX Runtime logged none)");
		return;
	}
	bool done_ep[OC_PLACE_MAX] = {false};
	for (uint32_t i = 0; i < r->entry_count; i++) {
		if (done_ep[i]) {
			continue;
		}
		const char *ep = r->entries[i].ep;
		append(buf, size, &len, "%s%s: %u nodes", len > 0 ? "; " : "", ep, oc_place_total(r, ep));
		// This EP's ops by count, largest first.
		bool used[OC_PLACE_MAX] = {false};
		for (uint32_t k = 0; k < max_ops; k++) {
			int best = -1;
			for (uint32_t j = 0; j < r->entry_count; j++) {
				if (!used[j] && strcmp(r->entries[j].ep, ep) == 0 &&
				    (best < 0 || r->entries[j].count > r->entries[best].count)) {
					best = (int)j;
				}
			}
			if (best < 0) {
				break;
			}
			used[best] = true;
			append(buf, size, &len, k == 0 ? " (%s %u" : ", %s %u", r->entries[best].op,
			       r->entries[best].count);
		}
		uint32_t rest = 0;
		for (uint32_t j = 0; j < r->entry_count; j++) {
			if (strcmp(r->entries[j].ep, ep) == 0) {
				done_ep[j] = true;
				rest += used[j] ? 0u : 1u;
			}
		}
		if (rest > 0) {
			append(buf, size, &len, " +%u op types", rest);
		}
		append(buf, size, &len, ")");
	}
	if (r->dropped > 0) {
		append(buf, size, &len, "; %u nodes not itemised", r->dropped);
	}
}


/*
 *
 * Profiling JSON.
 *
 */

//! The first occurrence of @p needle in [s, end); NULL when absent.
static const char *
find(const char *s, const char *end, const char *needle)
{
	const size_t n = strlen(needle);
	for (; s + n <= end; s++) {
		if (memcmp(s, needle, n) == 0) {
			return s;
		}
	}
	return NULL;
}

//! After `"key"`: skip spaces and the colon; NULL when not found in [s, end).
static const char *
value_of(const char *s, const char *end, const char *key)
{
	const char *at = find(s, end, key);
	if (at == NULL) {
		return NULL;
	}
	at += strlen(key);
	while (at < end && (*at == ' ' || *at == ':')) {
		at++;
	}
	return at < end ? at : NULL;
}

static bool
num_of(const char *s, const char *end, const char *key, uint64_t *out)
{
	const char *v = value_of(s, end, key);
	if (v == NULL || !isdigit((unsigned char)*v)) {
		return false;
	}
	uint64_t x = 0;
	while (v < end && isdigit((unsigned char)*v)) {
		x = x * 10u + (uint64_t)(*v - '0');
		v++;
	}
	*out = x;
	return true;
}

static bool
str_of(const char *s, const char *end, const char *key, char *buf, size_t size)
{
	const char *v = value_of(s, end, key);
	if (v == NULL || *v != '"') {
		return false;
	}
	v++;
	const char *q = v;
	while (q < end && *q != '"') {
		q++;
	}
	if (q >= end) {
		return false;
	}
	copy_n(buf, size, v, (size_t)(q - v));
	return true;
}

struct event
{
	const char *s, *e; //!< [s, e): from this event's "cat" to the next one's
};

//! The next event at or after @p *cursor; false at the end.
static bool
next_event(const char **cursor, const char *end, struct event *ev)
{
	const char *s = find(*cursor, end, "\"cat\"");
	if (s == NULL) {
		return false;
	}
	const char *e = find(s + 5, end, "\"cat\"");
	ev->s = s;
	ev->e = e != NULL ? e : end;
	*cursor = ev->e;
	return true;
}

bool
oc_profile_summarize(const char *json, size_t len, struct oc_profile_summary *out)
{
	memset(out, 0, sizeof(*out));
	if (json == NULL) {
		return false;
	}
	const char *end = json + len;
	struct event ev;
	char cat[16], name[160];
	uint64_t run_ts = 0, run_dur = 0;
	for (const char *c = json; next_event(&c, end, &ev);) {
		uint64_t ts = 0, dur = 0;
		if (str_of(ev.s, ev.e, "\"cat\"", cat, sizeof(cat)) && strcmp(cat, "Session") == 0 &&
		    str_of(ev.s, ev.e, "\"name\"", name, sizeof(name)) && strcmp(name, "model_run") == 0 &&
		    num_of(ev.s, ev.e, "\"ts\"", &ts) && num_of(ev.s, ev.e, "\"dur\"", &dur)) {
			if (out->runs == 0) {
				out->first_run_us = dur;
			}
			out->runs++;
			run_ts = ts;
			run_dur = dur;
		}
	}
	if (out->runs == 0) {
		return false;
	}
	out->last_run_us = run_dur;
	for (const char *c = json; next_event(&c, end, &ev);) {
		uint64_t ts = 0, dur = 0;
		char op[48] = "", ep[40] = "";
		if (!str_of(ev.s, ev.e, "\"cat\"", cat, sizeof(cat)) || strcmp(cat, "Node") != 0 ||
		    !str_of(ev.s, ev.e, "\"name\"", name, sizeof(name)) || !num_of(ev.s, ev.e, "\"ts\"", &ts) ||
		    !num_of(ev.s, ev.e, "\"dur\"", &dur) || ts < run_ts || ts > run_ts + run_dur) {
			continue;
		}
		const size_t nl = strlen(name);
		const char *suffix = "_kernel_time";
		if (nl < strlen(suffix) || strcmp(name + nl - strlen(suffix), suffix) != 0) {
			continue; // fence_before / fence_after
		}
		if (!str_of(ev.s, ev.e, "\"op_name\"", op, sizeof(op))) {
			snprintf(op, sizeof(op), "?");
		}
		normalize_op(op);
		if (!str_of(ev.s, ev.e, "\"provider\"", ep, sizeof(ep))) {
			snprintf(ep, sizeof(ep), "?");
		}
		out->kernel_us += dur;
		uint32_t i = 0;
		while (i < out->entry_count &&
		       (strcmp(out->entries[i].op, op) != 0 || strcmp(out->entries[i].ep, ep) != 0)) {
			i++;
		}
		if (i == out->entry_count) {
			if (out->entry_count >= OC_PROFILE_MAX) {
				continue;
			}
			out->entry_count++;
			memset(&out->entries[i], 0, sizeof(out->entries[i]));
			copy_n(out->entries[i].op, sizeof(out->entries[i].op), op, strlen(op));
			copy_n(out->entries[i].ep, sizeof(out->entries[i].ep), ep, strlen(ep));
		}
		out->entries[i].us += dur;
		out->entries[i].calls++;
	}
	// Largest first (insertion sort: at most OC_PROFILE_MAX entries).
	for (uint32_t i = 1; i < out->entry_count; i++) {
		const struct oc_profile_entry t = out->entries[i];
		uint32_t j = i;
		while (j > 0 && out->entries[j - 1].us < t.us) {
			out->entries[j] = out->entries[j - 1];
			j--;
		}
		out->entries[j] = t;
	}
	return true;
}

//! "DmlExecutionProvider" -> "Dml", "CPUExecutionProvider" -> "CPU".
static void
short_ep(const char *ep, char *buf, size_t size)
{
	const char *x = strstr(ep, "ExecutionProvider");
	copy_n(buf, size, ep, x != NULL ? (size_t)(x - ep) : strlen(ep));
}

void
oc_profile_format(const struct oc_profile_summary *s, uint32_t n, char *buf, size_t size)
{
	int w =
	    snprintf(buf, size, "%u run(s), first %.1f ms, last %.1f ms (kernels %.1f ms); top:", s->runs,
	             (double)s->first_run_us / 1000.0, (double)s->last_run_us / 1000.0, (double)s->kernel_us / 1000.0);
	size_t len = w > 0 && (size_t)w < size ? (size_t)w : 0;
	for (uint32_t i = 0; i < n && i < s->entry_count && len + 1 < size; i++) {
		char ep[24];
		short_ep(s->entries[i].ep, ep, sizeof(ep));
		w = snprintf(buf + len, size - len, "%s %s@%s %.1f ms (%u)", i == 0 ? "" : ",", s->entries[i].op, ep,
		             (double)s->entries[i].us / 1000.0, s->entries[i].calls);
		if (w <= 0) {
			break;
		}
		len += (size_t)w < size - len ? (size_t)w : size - len - 1;
	}
}
