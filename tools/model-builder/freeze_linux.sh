#!/usr/bin/env bash
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# Freeze dxr-model-builder for Linux x86_64 (PyInstaller onedir) from the pinned set.
#
#   tools/model-builder/freeze_linux.sh [--out <dir>] [--cache <dir>]
#
# Run it inside ubuntu:22.04 (CI does; tools/model-builder/freeze_linux_container.sh does it on
# a dev box with podman or docker): a frozen binary's glibc floor is its build host's, and the
# runtime's Linux packages support 22.04 / 24.04 / 26.04 (runtime #1656). The script refuses
# a newer host unless DXR_MB_ALLOW_ANY_HOST=1.
#
# Python is CPython 3.12.10 from python-build-standalone (pinned by sha256), the same version the
# Windows builder is frozen with; the packages are tools/model-builder/requirements-freeze.txt.
# Output: <out>/dxr-model-builder/ (the onedir) and <out>/dxr-model-builder-<ver>-linux-x86_64.tar.gz.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=$repo/build/mb
cache=$repo/build/mb-cache
while [ $# -gt 0 ]; do
	case $1 in
	--out) out=$2; shift 2 ;;
	--cache) cache=$2; shift 2 ;;
	*) echo "usage: $0 [--out <dir>] [--cache <dir>]" >&2; exit 2 ;;
	esac
done

PBS_TAG=20250517
PBS_PY=3.12.10
PBS_FILE="cpython-${PBS_PY}+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"
PBS_SHA256=896254d77c86a9d9a2bb17efa760fcdb6250273fb4c91d4be867e347d0a5fb82
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_FILE//+/%2B}"

[ "$(uname -m)" = x86_64 ] || { echo "freeze_linux.sh: x86_64 only" >&2; exit 2; }
glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
if [ "$glibc" != 2.35 ] && [ "${DXR_MB_ALLOW_ANY_HOST:-0}" != 1 ]; then
	echo "freeze_linux.sh: glibc $glibc; freeze on ubuntu:22.04 (glibc 2.35) or set DXR_MB_ALLOW_ANY_HOST=1" >&2
	exit 2
fi
for t in curl objdump tar sha256sum; do
	command -v "$t" >/dev/null || { echo "freeze_linux.sh: needs $t (apt install curl binutils ca-certificates libquadmath0)" >&2; exit 2; }
done

mkdir -p "$cache" "$out"
pbs=$cache/$PBS_FILE
if ! echo "$PBS_SHA256  $pbs" | sha256sum -c --status 2>/dev/null; then
	curl -sSfL --retry 3 -o "$pbs.part" "$PBS_URL"
	echo "$PBS_SHA256  $pbs.part" | sha256sum -c --status || { echo "freeze_linux.sh: $PBS_FILE hash mismatch" >&2; exit 1; }
	mv "$pbs.part" "$pbs"
fi
py=$cache/python-$PBS_PY
if [ ! -x "$py/bin/python3.12" ]; then
	rm -rf "$py" "$cache/python"
	tar -C "$cache" -xzf "$pbs"
	mv "$cache/python" "$py"
fi
venv=$cache/venv-$PBS_PY
if [ ! -x "$venv/bin/python" ] || ! cmp -s "$here/requirements-freeze.txt" "$venv/.requirements"; then
	rm -rf "$venv"
	"$py/bin/python3.12" -m venv "$venv"
	"$venv/bin/python" -m pip install --quiet --disable-pip-version-check --no-cache-dir \
		-r "$here/requirements-freeze.txt"
	cp "$here/requirements-freeze.txt" "$venv/.requirements"
fi

ver=$("$venv/bin/python" -I -c "import ast,sys;t=ast.parse(open(sys.argv[1]).read());print(next(n.value.value for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='TOOL_VERSION'))" "$here/dxr_model_builder.py")

rm -rf "$out/dist" "$out/work"
(cd "$repo" && "$venv/bin/python" -m PyInstaller --noconfirm --clean --log-level WARN \
	--distpath "$out/dist" --workpath "$out/work" "$here/dxr-model-builder.spec")
rm -rf "$out/work"

bin=$out/dist/dxr-model-builder
# Every file group-/world-read only, no write bit beyond the owner (the .deb ships them root-owned).
chmod -R u+rwX,go+rX,go-w "$bin"
floor=$(find "$bin" -type f \( -name '*.so*' -o -name dxr-model-builder \) -exec objdump -T {} + 2>/dev/null |
	grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
echo "$floor" > "$bin/GLIBC_FLOOR"
tarball=$out/dxr-model-builder-$ver-linux-x86_64.tar.gz
tar -C "$out/dist" --owner=0 --group=0 --numeric-owner --sort=name -czf "$tarball" dxr-model-builder
echo "frozen: $bin ($(du -sh "$bin" | cut -f1), glibc floor $floor)"
echo "tarball: $tarball ($(du -sh "$tarball" | cut -f1))"
