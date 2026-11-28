#!/usr/bin/env bash
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# Package the frozen Linux builder (tools/model-builder/freeze_linux.sh) as a .deb:
#
#   tools/model-builder/package_deb_linux.sh <onedir> <version> <out-dir>
#
# Layout (the runtime's conversion-provider-discovery spec, section 5.6.3 / 5.6.6):
#   /usr/lib/displayxr/model-builder/dxr-model-builder/           the PyInstaller onedir
#   /usr/share/displayxr/model-builders/open-model-builder.json  {"id","version","binary"}
#   /usr/share/doc/displayxr-model-builder/copyright
# Every file root-owned and not group- or world-writable, so the store classifies the binary
# platform-trusted and admits the manifest (both the root-owned owner chain of plugin-discovery
# section 4.3). The model store reads the manifest on every fetch: no service restart, so no
# maintainer scripts. It downloads nothing and builds nothing by itself.
set -euo pipefail
onedir=$1; ver=$2; out=$3
[ -x "$onedir/dxr-model-builder" ] || { echo "not a frozen builder: $onedir" >&2; exit 2; }
command -v dpkg-deb >/dev/null || { echo "needs dpkg-deb" >&2; exit 2; }

pkg=displayxr-model-builder
tool_id=open-model-builder
libdir=/usr/lib/displayxr/model-builder
bin=$libdir/dxr-model-builder/dxr-model-builder
debver=${ver#v}

root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
stage=$root/$pkg
mkdir -p "$stage/DEBIAN" "$stage$libdir" "$stage/usr/share/displayxr/model-builders" \
	"$stage/usr/share/doc/$pkg"
cp -a "$onedir" "$stage$libdir/dxr-model-builder"

# The builder's own version is the manifest's: the store compares it with tool.min_version.
cat > "$stage/usr/share/displayxr/model-builders/$tool_id.json" <<JSON
{
    "id": "$tool_id",
    "version": "$debver",
    "binary": "$bin",
    "display_name": "DisplayXR Model Builder (research preview)"
}
JSON

cat > "$stage/usr/share/doc/$pkg/copyright" <<'TXT'
DisplayXR Model Builder (dxr-model-builder), research preview.
Source: https://github.com/DisplayXR/displayxr-conversion-open (tools/model-builder)

The builder's own code: BSL-1.0, Copyright 2026, The DisplayXR Project.

It carries apple/ml-sharp's src/ at a pinned commit under that project's licence
(lib/displayxr/model-builder/dxr-model-builder/_internal/ml-sharp/LICENSE) and the Python
packages of its pinned set under their own licences (their dist-info metadata in _internal/).

It ships no model weights and no file made from them. It builds Apple's SHARP models on this
machine from the official checkpoint, which the DisplayXR model store downloads only after the
user accepts Apple's research-only licence (Apple Machine Learning Research Model licence).
TXT

floor=$(cat "$onedir/GLIBC_FLOOR" 2>/dev/null || true)
floor=${floor#GLIBC_}
depends="libc6 (>= ${floor:-2.35})"
size=$(du -sk "$stage" | cut -f1)
cat > "$stage/DEBIAN/control" <<CTL
Package: $pkg
Version: $debver
Architecture: amd64
Maintainer: The DisplayXR Project <noreply@displayxr.dev>
Section: science
Priority: optional
Installed-Size: $size
Depends: $depends
Enhances: displayxr-runtime
Homepage: https://github.com/DisplayXR/displayxr-conversion-open
Description: DisplayXR Model Builder (research preview, non-commercial models)
 Builds research-only 2D-to-3D models (Apple's SHARP) on this machine from the
 official upstream checkpoint, for the DisplayXR runtime's model store
 ("displayxr-cli conv models fetch <pack> --model <name>", the Dashboard's Build
 button). It downloads nothing and builds nothing by itself, and ships no model
 weights or files made from them.
CTL

find "$stage" -type d -exec chmod 0755 {} +
find "$stage" -type f -exec chmod go-w,a+r {} +
chmod 0755 "$stage$bin"
mkdir -p "$out"
deb=$out/${pkg}_${debver}_amd64.deb
dpkg-deb --root-owner-group -Zxz --build "$stage" "$deb" >/dev/null
echo "$deb"
