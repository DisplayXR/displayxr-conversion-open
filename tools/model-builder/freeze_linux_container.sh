#!/usr/bin/env bash
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# Dev-box wrapper: run tools/model-builder/freeze_linux.sh inside ubuntu:22.04 (podman, else
# docker), so the frozen builder's glibc floor is 22.04's whatever the host is.
#
#   tools/model-builder/freeze_linux_container.sh [--out <dir>] [--cache <dir>] [--memory 10g]
#
# The repo is mounted read-write at the same path; --out / --cache must be under the repo or
# under $DXR_MB_MOUNT (another directory to mount, e.g. a big work disk).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
mem=10g
args=()
while [ $# -gt 0 ]; do
	case $1 in
	--memory) mem=$2; shift 2 ;;
	*) args+=("$1"); shift ;;
	esac
done
eng=$(command -v podman || command -v docker || true)
[ -n "$eng" ] || { echo "needs podman or docker" >&2; exit 2; }
mounts=(-v "$repo:$repo")
[ -n "${DXR_MB_MOUNT:-}" ] && mounts+=(-v "$DXR_MB_MOUNT:$DXR_MB_MOUNT")
# Root inside the container to apt-get the two tools. Rootless podman maps container root to the
# caller, so the output is the caller's; rootful docker gets a chown at the end.
chown_to=""
case $eng in *docker) chown_to="$(id -u):$(id -g)" ;; esac
exec "$eng" run --rm ${DXR_MB_CTR_NAME:+--name "$DXR_MB_CTR_NAME"} --memory "$mem" --memory-swap "$mem" "${mounts[@]}" -w "$repo" \
	-e HOME=/tmp -e DEBIAN_FRONTEND=noninteractive -e CHOWN_TO="$chown_to" -e DXR_MB_MOUNT="${DXR_MB_MOUNT:-}" docker.io/library/ubuntu:22.04 bash -c '
set -e
apt-get update -qq >/dev/null
apt-get install -y -qq --no-install-recommends curl ca-certificates binutils libquadmath0 >/dev/null
rc=0; "$@" || rc=$?
if [ -n "$CHOWN_TO" ]; then chown -R "$CHOWN_TO" "'"$repo"'/build" ${DXR_MB_MOUNT:+"$DXR_MB_MOUNT"} 2>/dev/null || true; fi
exit $rc' bash "$here/freeze_linux.sh" "${args[@]}"
