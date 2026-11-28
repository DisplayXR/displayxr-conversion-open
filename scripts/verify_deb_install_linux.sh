#!/usr/bin/env bash
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# Install-verify the provider .deb on a CLEAN Ubuntu release. Run it as root inside a
# pristine ubuntu:<release> container (CI's DebInstall matrix does this for 22.04, 24.04 and
# 26.04); locally:
#
#   docker run --rm -v "$PWD:/w" -w /w ubuntu:22.04 \
#     ./scripts/verify_deb_install_linux.sh out/displayxr-open-conversion_*.deb \
#                                           runtime-deb/displayxr-runtime_*.deb
#
# The runtime .deb is passed alongside because the provider Depends on it; apt installs both
# from the same command, as the Linux meta-bundle does, so an unsatisfiable dependency in
# either one fails here.
#
# Fails if:
#   * apt cannot resolve the package's Depends from that release's archive (installed with
#     --no-install-recommends: Depends alone must be enough);
#   * a Depends / Recommends / Suggests name does not exist on that release;
#   * `ldd -r` on the provider, ONNX Runtime or OpenVINO's core libraries reports a missing
#     library, symbol or symbol version (the GPU plugin is checked once its Recommends is in);
#   * a shipped file or directory is not root-owned, or is group/world writable (the
#     runtime's platform-trusted rule);
#   * the discovery manifest is missing, or does not point at the installed .so and packs;
#   * (with the runtime installed) `displayxr-cli conv list` does not list open-conversion.
set -euo pipefail

[ "$#" -ge 1 ] || { echo "usage: $0 <displayxr-open-conversion.deb> [displayxr-runtime.deb]" >&2; exit 2; }
[ "$(id -u)" = 0 ] || { echo "error: run as root in a throwaway container." >&2; exit 2; }

DEBS=()
for d in "$@"; do DEBS+=("$(readlink -f "$d")"); done
# shellcheck disable=SC1091
. /etc/os-release
echo "==> $PRETTY_NAME: installing ${DEBS[*]##*/}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends "${DEBS[@]}"

PKG=displayxr-open-conversion
PLUGINS=/usr/lib/displayxr/plugins
DEST=$PLUGINS/open-conversion
SO=$DEST/DisplayXR-OpenConversion.so
MANIFEST=$PLUGINS/200-open-conversion-conversion-provider.json
dpkg-query -W -f='==> installed ${Package} ${Version}\n    Depends: ${Depends}\n    Recommends: ${Recommends}\n    Suggests: ${Suggests}\n' "$PKG"

fail=0

for field in Depends Recommends Suggests; do
	for p in $(dpkg-query -W -f="\${$field}" "$PKG" | tr ',|' '\n\n' | sed 's/(.*)//; s/:any//; s/ //g' | sed '/^$/d'); do
		if apt-cache show "$p" >/dev/null 2>&1 || dpkg-query -W "$p" >/dev/null 2>&1; then
			echo "    $field $p: available"
		else
			echo "error: $field '$p' does not exist on $PRETTY_NAME." >&2
			fail=1
		fi
	done
done

[ -f "$SO" ] || { echo "error: $SO not installed." >&2; exit 1; }
[ "$(dpkg-query -S "$SO" | cut -d: -f1)" = "$PKG" ] || { echo "error: $SO is not owned by $PKG." >&2; fail=1; }
[ -f "$MANIFEST" ] || { echo "error: $MANIFEST not installed." >&2; exit 1; }
# A clean container has no python3, and the package needs none: plain text checks.
grep -q "\"id\": *\"open-conversion\"" "$MANIFEST" || { echo "error: $MANIFEST has no id open-conversion." >&2; fail=1; }
grep -q "\"binary_path\": *\"$SO\"" "$MANIFEST" || { echo "error: $MANIFEST does not point at $SO." >&2; fail=1; }
packs="$(grep -o "\"$DEST/[^\"]*\.models\.json\"" "$MANIFEST" | tr -d '"')"
grep -qx "$DEST/dxr-open-default.models.json" <<<"$packs" || { echo "error: $MANIFEST lacks the default pack." >&2; fail=1; }
for pk in $packs; do
	[ -f "$pk" ] || { echo "error: $MANIFEST names a pack that is not installed: $pk" >&2; fail=1; }
done
echo "    manifest: $SO, packs: $(echo $packs)"

# Platform-trusted: root-owned, not group/world writable, every file and the chain up to /.
loose="$(find "$DEST" "$MANIFEST" \( ! -user root -o -perm /022 \) -print)"
[ -z "$loose" ] || { echo "error: not root-owned or group/world writable:" >&2; echo "$loose" >&2; fail=1; }
d=$DEST
while [ "$d" != / ]; do
	if [ "$(stat -c %u "$d")" != 0 ] || [ -n "$(find "$d" -maxdepth 0 -perm /022)" ]; then
		echo "error: $d is not root-owned 0755-or-tighter." >&2
		fail=1
	fi
	d="$(dirname "$d")"
done

check_ldd() {
	local out
	# OpenVINO's libraries carry no RUNPATH: the provider maps them by absolute path in
	# dependency order, so their siblings resolve from openvino/ (LD_LIBRARY_PATH says the same).
	out="$(LD_LIBRARY_PATH="$DEST/openvino" ldd -r "$1" 2>&1)" || true
	if grep -E 'not found|undefined symbol|version .* not found' <<<"$out"; then
		echo "error: unresolved dependency in $1 on $PRETTY_NAME." >&2
		fail=1
	else
		echo "    ldd -r ${1#"$DEST"/}: ok"
	fi
}
check_ldd "$SO"
check_ldd "$DEST/libonnxruntime.so.1"
gpu_plugins=()
if [ -d "$DEST/openvino" ]; then
	for f in "$DEST"/openvino/*.so*; do
		case "$f" in *intel_gpu_plugin*) gpu_plugins+=("$f"); continue ;; esac
		check_ldd "$f"
	done
fi

if command -v displayxr-cli >/dev/null 2>&1; then
	unset XR_RUNTIME_JSON XRT_PLUGIN_SEARCH_PATH
	# Captured, not piped: `cmd | grep -q` under pipefail fails on grep's SIGPIPE.
	if conv_out="$(displayxr-cli conv list 2>&1)"; then
		echo "$conv_out" | sed 's/^/    /'
		if grep -q 'open-conversion' <<<"$conv_out"; then
			echo "    displayxr-cli conv list: open-conversion is discovered"
		else
			echo "error: displayxr-cli conv list does not list open-conversion on $PRETTY_NAME." >&2
			fail=1
		fi
	else
		echo "$conv_out" | sed 's/^/    /'
		echo "    note: this runtime's displayxr-cli has no working 'conv list' here; discovery not checked."
	fi
else
	echo "    note: displayxr-cli not installed (no runtime .deb passed); discovery not checked."
fi

# With the Recommends in (the OpenCL loader), OpenVINO's GPU plugin must resolve too.
if [ "${#gpu_plugins[@]}" -gt 0 ]; then
	rec="$(dpkg-query -W -f='${Recommends}' "$PKG" | tr ',' ' ')"
	if [ -n "${rec// /}" ]; then
		# shellcheck disable=SC2086
		apt-get install -y -qq --no-install-recommends $rec
	fi
	for f in "${gpu_plugins[@]}"; do check_ldd "$f"; done
fi

# Removal leaves nothing of ours behind (the model store is never ours to touch).
apt-get remove -y -qq "$PKG"
if [ -e "$MANIFEST" ] || [ -e "$DEST" ]; then
	echo "error: $PKG removal left $MANIFEST or $DEST behind." >&2
	fail=1
else
	echo "    remove: manifest and payload gone"
fi

[ "$fail" = 0 ] || { echo "==> FAIL on $PRETTY_NAME" >&2; exit 1; }
echo "==> PASS on $PRETTY_NAME"
