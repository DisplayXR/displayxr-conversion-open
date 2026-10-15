#!/usr/bin/env bash
# Copyright 2026, The DisplayXR Project
# SPDX-License-Identifier: BSL-1.0
#
# Package a `cmake --install` tree as a user-level tarball:
#   package_posix.sh <dist-dir> <version> <out-dir>
# The tarball carries install.sh, which copies the payload to
# ${XDG_DATA_HOME:-~/.local/share}/DisplayXR/OpenConversion and writes the discovery manifest
# (NNN-<id>-conversion-provider.json) beside the other plug-in manifests. No model blob is
# shipped or downloaded: the runtime model store fetches them on first use.
set -euo pipefail
dist=$1; ver=$2; out=$3
name=displayxr-open-conversion-$ver-linux-x86_64
root=$(mktemp -d)
stage=$root/$name
mkdir -p "$stage/payload" "$out"
cp -a "$dist"/. "$stage/payload/"
echo "$ver" > "$stage/VERSION"

cat > "$stage/install.sh" <<'INST'
#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
data=${XDG_DATA_HOME:-$HOME/.local/share}/DisplayXR
dest=$data/OpenConversion
mkdir -p "$dest" "$data/DisplayProcessors"
cp -a "$here/payload/." "$dest/"
ver=$(cat "$here/VERSION")
tmp=$(mktemp "$data/DisplayProcessors/.oc.XXXXXX")
cat > "$tmp" <<JSON
{
    "file_format_version": "1.0",
    "plugin": {
        "id":           "open-conversion",
        "display_name": "DisplayXR Open Conversion",
        "vendor":       "DisplayXR",
        "version":      "$ver",
        "binary_path":  "$dest/DisplayXR-OpenConversion.so",
        "probe_order":  200,
        "packs": [ "$dest/dxr-open-default.models.json" ]
    }
}
JSON
mv "$tmp" "$data/DisplayProcessors/200-open-conversion-conversion-provider.json"
echo "installed to $dest"
INST

cat > "$stage/uninstall.sh" <<'UNINST'
#!/usr/bin/env bash
# Removes only this provider manifest and files, never the runtime model store.
set -euo pipefail
data=${XDG_DATA_HOME:-$HOME/.local/share}/DisplayXR
rm -f "$data/DisplayProcessors/200-open-conversion-conversion-provider.json"
rm -rf "$data/OpenConversion"
UNINST

chmod +x "$stage/install.sh" "$stage/uninstall.sh"
tar -C "$root" -czf "$out/$name.tar.gz" "$name"
echo "$out/$name.tar.gz"
