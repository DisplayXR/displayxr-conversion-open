#!/usr/bin/env bash
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# Package a `cmake --install` tree as a system-wide Debian package:
#   package_deb.sh <dist-dir> <version> <out-dir>
# Output: <out-dir>/displayxr-open-conversion_<version>_<arch>.deb
#
# The per-user tarball (package_posix.sh) is unchanged; this is the system install the
# Linux meta-bundle chains (one apt-get transaction, as root).
#
# ONE package for Ubuntu 22.04, 24.04 and 26.04: build it on the OLDEST of the three (an
# ubuntu:22.04 container; CI does this) and set DXR_DEB_MAX_GLIBC to that release's glibc
# (2.35), so a package built on anything newer fails here instead of shipping.
# scripts/verify_deb_install_linux.sh is the install-side check.
#
# --- Installed layout ------------------------------------------------------------------
#   /usr/lib/displayxr/plugins/200-open-conversion-conversion-provider.json   (discovery)
#   /usr/lib/displayxr/plugins/open-conversion/                               (the dist tree)
#       DisplayXR-OpenConversion.so, libonnxruntime.so.1, *.models.json, openvino/, licenses/
# The manifest sits in the runtime's built-in POSIX plug-in root; the payload goes one level
# down because the provider resolves everything from its own directory: ONNX Runtime next to
# it, OpenVINO in openvino/, and EVERY *.models.json next to it is loaded as one of its packs.
# Flat in the shared root it would also load other providers' packs (the runtime's own
# sim-conversion.models.json lives there). The whole tree is root-owned and not group/world
# writable, so the runtime's provenance check classifies the binary platform-trusted.
#
# --- Model weights: none in the package -------------------------------------------------
# The provider never downloads or writes a blob; the runtime's model store does, in the
# process that hosts the provider (displayxr-service / displayxr-conversion-host, run as the
# logged-in user by the systemd user unit). Its POSIX root is $XDG_DATA_HOME/displayxr/models
# (default ~/.local/share/displayxr/models) while that is writable, which is per-user and
# independent of where the provider is installed (runtime conversion-provider-discovery.md
# §5.4). So a system install changes nothing about where weights go, and this package
# creates no store directory: /var/lib/displayxr/models is only the store's fallback for a
# process without a writable home, and is the runtime's to provision, not a provider's.
#
# --- Maintainer scripts -----------------------------------------------------------------
# One thing only: ask the runtime's helper to try-restart a running displayxr.service. The
# service's manifest watch polls the per-user dir only (§3.8.1), so without a restart a
# manifest written into the system root is not noticed until something else wakes it, and a
# running host keeps the previous provider mapped (runtime #2812). The helper restarts only a
# service that is running, never starts one, always exits 0. A runtime too old to ship it is
# left alone.
set -euo pipefail

[ "$#" -eq 3 ] || { echo "usage: $0 <dist-dir> <version> <out-dir>" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$(cd "$1" && pwd)"; VERSION="$2"; OUT="$3"
PKG=displayxr-open-conversion
PLUGINS=/usr/lib/displayxr/plugins
DEST=$PLUGINS/open-conversion
MANIFEST_NAME=200-open-conversion-conversion-provider.json

# The oldest runtime this package Depends on (CMake OPENCONV_MIN_RUNTIME_VERSION).
MIN_RUNTIME_VERSION="$(sed -nE 's/^set\(OPENCONV_MIN_RUNTIME_VERSION[[:space:]]+"([^"]+)".*/\1/p' "$ROOT/CMakeLists.txt")"
[ -n "$MIN_RUNTIME_VERSION" ] || { echo "error: OPENCONV_MIN_RUNTIME_VERSION not found in CMakeLists.txt" >&2; exit 1; }

for tool in dpkg-deb dpkg-shlibdeps objdump nm python3; do
	command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool not found (apt install dpkg-dev binutils python3)." >&2; exit 1; }
done

# Debian-legal upstream version: v1.2.3-4-gabc -> 1.2.3+4.gabc, -dirty -> +dirty.
VERSION="$(echo "$VERSION" | sed -e 's/^v//' -e 's/-dirty$/+dirty/' -e 's/-\([0-9]\+\)-g/+\1.g/')"
case "$VERSION" in [0-9]*) : ;; *) echo "error: version '$VERSION' does not start with a digit" >&2; exit 1 ;; esac
ARCH="$(dpkg --print-architecture)"

SO_NAME=DisplayXR-OpenConversion.so
[ -f "$DIST/$SO_NAME" ] || { echo "error: $DIST/$SO_NAME not found (run cmake --install first)." >&2; exit 1; }
[ -f "$DIST/dxr-open-default.models.json" ] || { echo "error: the default pack is not in $DIST." >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
STAGE="$work/${PKG}_${VERSION}_${ARCH}"
mkdir -p "$STAGE/DEBIAN" "$STAGE$DEST"
cp -a "$DIST"/. "$STAGE$DEST/"
# Files 0644, directories and nothing else 0755: no group/world-writable bit may ship (the
# provenance check walks the owner chain), and no shared library needs +x.
find "$STAGE/usr" -type d -exec chmod 0755 {} +
find "$STAGE/usr" -type f -exec chmod 0644 {} +

STAGED="$STAGE$DEST/$SO_NAME"
syms="$(nm -D --defined-only "$STAGED" | awk '$2 == "T" && $3 != "_init" && $3 != "_fini" { print $3 }')"
[ "$syms" = "xrtConversionPluginNegotiate" ] ||
	{ echo "error: $SO_NAME must export exactly xrtConversionPluginNegotiate, exports: $syms" >&2; exit 1; }
if objdump -p "$STAGED" | grep -E 'NEEDED' | grep -qiE 'onnxruntime|openvino'; then
	echo "error: $SO_NAME links an inference runtime at load time; it must load it by absolute path." >&2
	exit 1
fi
# No build-machine path may ship in any ELF: the provider loads its neighbours by absolute
# path, and the bundled libraries' own RUNPATH ($ORIGIN) is all they need.
while IFS= read -r -d '' f; do
	rp="$(objdump -p "$f" 2>/dev/null | awk '/RPATH|RUNPATH/ {print $2}')"
	case "$rp" in "" | *'$ORIGIN'*) : ;; *) echo "error: ${f#"$STAGE"} carries RPATH/RUNPATH '$rp'" >&2; exit 1 ;; esac
done < <(find "$STAGE/usr" -type f -name '*.so*' -print0)

# --- Packs and the discovery manifest ---------------------------------------------------
packs=""
for p in "$STAGE$DEST"/*.models.json; do
	[ -e "$p" ] || continue
	packs="${packs:+$packs, }\"$DEST/$(basename "$p")\""
done
MANIFEST="$STAGE$PLUGINS/$MANIFEST_NAME"
cat > "$MANIFEST" <<JSON
{
    "file_format_version": "1.0",
    "plugin": {
        "id":           "open-conversion",
        "display_name": "DisplayXR Open Conversion",
        "vendor":       "DisplayXR",
        "version":      "$VERSION",
        "binary_path":  "$DEST/$SO_NAME",
        "probe_order":  200,
        "packs": [ $packs ]
    }
}
JSON
chmod 0644 "$MANIFEST"
python3 -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$MANIFEST" ||
	{ echo "error: the manifest is not valid JSON" >&2; exit 1; }

# --- Depends: one .deb for Ubuntu 22.04, 24.04 and 26.04 --------------------------------
# The runtime's #1656 rules, as in its own package and the other component .debs:
#  1. The glibc / libstdc++ floor is the build host's, so the package is built on the oldest
#     supported release, dpkg-shlibdeps turns the symbol versions every shipped ELF
#     references into VERSIONED Depends, and DXR_DEB_MAX_GLIBC makes a floor above the
#     oldest release a hard error.
#  2. A library's PACKAGE NAME is not stable across releases, so every DT_NEEDED that this
#     package does not ship itself must be on STABLE_SONAMES ("the name dpkg-shlibdeps
#     derives on the oldest release resolves on all three").
# One exception, outside dpkg-shlibdeps: OpenVINO's GPU plugin (openvino/
# libopenvino_intel_gpu_plugin.so) needs the system's OpenCL loader. It is a Recommends, not
# a Depends: OpenVINO loads that plugin only for its GPU device, and without it the splat
# capability runs on OpenVINO's CPU device (the provider ranks the backend at run time). The
# GPU compute driver itself (intel-opencl-icd) is a Suggests: nothing should bundle or force
# a GPU driver.
STABLE_SONAMES=(
	libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 ld-linux-x86-64.so.2
	libstdc++.so.6 libgcc_s.so.1
)
OPENCL_SONAME=libOpenCL.so.1
mapfile -d '' elfs < <(find "$STAGE/usr" -type f -name '*.so*' -print0 | sort -z)
shipped=""
for f in "${elfs[@]}"; do
	shipped="$shipped $(basename "$f") $(objdump -p "$f" | awk '/SONAME/ {print $2}')"
done
bad=""; opencl_users=""; shlib_elfs=()
for f in "${elfs[@]}"; do
	uses_opencl=0
	for so in $(objdump -p "$f" | awk '/NEEDED/ {print $2}'); do
		case " $shipped " in *" $so "*) continue ;; esac
		[ "$so" = "$OPENCL_SONAME" ] && { uses_opencl=1; continue; }
		ok=0
		for s in "${STABLE_SONAMES[@]}"; do [ "$so" = "$s" ] && ok=1 && break; done
		[ "$ok" = 1 ] || bad="$bad ${f#"$STAGE"}:$so"
	done
	if [ "$uses_opencl" = 1 ]; then
		opencl_users="$opencl_users ${f#"$STAGE"}"
	else
		shlib_elfs+=("$f")
	fi
done
if [ -n "$bad" ]; then
	echo "error: DT_NEEDED on libraries not known to resolve under one package name" >&2
	echo "       across Ubuntu 22.04/24.04/26.04:$bad" >&2
	echo "       Drop the dependency, ship it, or add it to STABLE_SONAMES once the install" >&2
	echo "       check proves the 22.04-derived name resolves on all three." >&2
	exit 1
fi
[ -z "$opencl_users" ] || echo "==> needs $OPENCL_SONAME (Recommends ocl-icd-libopencl1):$opencl_users"

# dpkg-shlibdeps over every shipped ELF but the OpenCL users; -l lets it resolve the
# libraries this package ships itself (they need no Depends), the shlibs file below names
# them as ours.
shl="$work/shlibs"
mkdir -p "$shl/debian"
printf 'Source: %s\n\nPackage: %s\nArchitecture: any\n' "$PKG" "$PKG" > "$shl/debian/control"
libdirs=()
while IFS= read -r d; do libdirs+=("-l$d"); done < <(for f in "${elfs[@]}"; do dirname "$f"; done | sort -u)
: > "$shl/debian/shlibs.local"
for f in "${elfs[@]}"; do
	soname="$(objdump -p "$f" | awk '/SONAME/ {print $2}')"
	[ -n "$soname" ] || continue
	# shlibs format: <library> <soversion> <dependency>; "libfoo.so.1" -> "libfoo 1".
	case "$soname" in
	*.so.*) printf '%s %s %s\n' "${soname%%.so.*}" "${soname#*.so.}" "$PKG" >> "$shl/debian/shlibs.local" ;;
	*.so)   printf '%s %s\n' "${soname%.so}" "$PKG" >> "$shl/debian/shlibs.local" ;;
	esac
done
LIB_DEPENDS="$(cd "$shl" && dpkg-shlibdeps -O "${libdirs[@]}" "${shlib_elfs[@]}" 2> "$work/shlibdeps.err" |
	sed -n 's/^shlibs:Depends=//p')" || { cat "$work/shlibdeps.err" >&2; echo "error: dpkg-shlibdeps failed." >&2; exit 1; }
sed 's/^/    shlibdeps: /' "$work/shlibdeps.err"
# Our own package name never belongs in our Depends.
LIB_DEPENDS="$(echo "$LIB_DEPENDS" | tr ',' '\n' | sed 's/^ *//' | grep -v "^$PKG\( \|$\)" | paste -sd, - | sed 's/,/, /g')"
[ -n "$LIB_DEPENDS" ] || { echo "error: dpkg-shlibdeps produced no Depends." >&2; exit 1; }
DEPENDS="displayxr-runtime (>= $MIN_RUNTIME_VERSION), $LIB_DEPENDS"
RECOMMENDS=""
[ -z "$opencl_users" ] || RECOMMENDS="ocl-icd-libopencl1"
echo "==> Depends: $DEPENDS"
echo "==> Recommends: ${RECOMMENDS:-<none>}"

GLIBC_FLOOR="$(for f in "${elfs[@]}"; do objdump -T "$f"; done | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)"
echo "==> glibc floor: GLIBC_$GLIBC_FLOOR"
if [ -n "${DXR_DEB_MAX_GLIBC:-}" ] &&
	[ "$(printf '%s\n%s\n' "$GLIBC_FLOOR" "$DXR_DEB_MAX_GLIBC" | sort -V | tail -1)" != "$DXR_DEB_MAX_GLIBC" ]; then
	echo "error: the package needs GLIBC_$GLIBC_FLOOR, above DXR_DEB_MAX_GLIBC=$DXR_DEB_MAX_GLIBC" >&2
	echo "       (the oldest supported release). Build the .deb on that release." >&2
	exit 1
fi

INSTALLED_KB="$(du -sk "$STAGE/usr" | cut -f1)"
{
	echo "Package: $PKG"
	echo "Version: $VERSION"
	echo "Section: libs"
	echo "Priority: optional"
	echo "Architecture: $ARCH"
	echo "Depends: $DEPENDS"
	[ -z "$RECOMMENDS" ] || echo "Recommends: $RECOMMENDS"
	[ -z "$opencl_users" ] || echo "Suggests: intel-opencl-icd"
	echo "Installed-Size: $INSTALLED_KB"
	echo "Maintainer: The DisplayXR Project <noreply@displayxr.dev>"
	echo "Homepage: https://github.com/DisplayXR/displayxr-conversion-open"
	cat <<'EOF_DESC'
Description: DisplayXR open-model conversion provider (2D to 3D)
 The conversion provider every DisplayXR install gets without a vendor module:
 depth from open models through the bundled ONNX Runtime (CPU), and photo to
 3D Gaussian splats through the bundled OpenVINO (Intel GPU or CPU) when a pack
 names a splat model. Installs system-wide into the runtime's built-in plug-in
 directory (/usr/lib/displayxr/plugins) with probe order 200.
 .
 No model weights are in this package: the DisplayXR runtime's model store
 downloads them on first use into the user's ~/.local/share/displayxr/models.
EOF_DESC
} > "$STAGE/DEBIAN/control"

write_restart_script() {
	cat > "$1" <<'EOS'
#!/bin/sh
set -e
case "$1" in
	__WHEN__)
		if [ -x /usr/lib/displayxr/bin/displayxr-restart-service ]; then
			/usr/lib/displayxr/bin/displayxr-restart-service || true
		fi
		;;
esac
exit 0
EOS
	sed -i.bak "s/__WHEN__/$2/" "$1" && rm -f "$1.bak"
	chmod 0755 "$1"
}
write_restart_script "$STAGE/DEBIAN/postinst" "configure"
write_restart_script "$STAGE/DEBIAN/postrm" "remove|purge"

mkdir -p "$OUT"
DEB="$(cd "$OUT" && pwd)/${PKG}_${VERSION}_${ARCH}.deb"
if command -v fakeroot >/dev/null 2>&1; then
	fakeroot dpkg-deb --build --root-owner-group "$STAGE" "$DEB" >/dev/null
else
	dpkg-deb --build --root-owner-group "$STAGE" "$DEB" >/dev/null
fi
echo "==> $DEB"
dpkg-deb --info "$DEB" | sed 's/^/    /'
echo "    --- contents ---"
dpkg-deb --contents "$DEB" | sed 's/^/    /'
