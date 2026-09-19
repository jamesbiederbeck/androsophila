#!/bin/sh
# Build only the optional observer. Do not replace the primary ViZDoom binary.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
build_root="$root/tools/vizdoom-observer"
archive="$root/tools/vizdoom-1.3.0.tar.gz"
mkdir -p "$root/tools"
if [ ! -f "$archive" ]; then
 curl --fail --location https://github.com/Farama-Foundation/ViZDoom/archive/refs/tags/1.3.0.tar.gz --output "$archive"
fi
printf '%s  %s\n' 76ddf186d7f093ef85cbcb0e7e387757d60e45190eb5da6d075aab31ffc316ed "$archive" | shasum -a 256 -c -
if [ ! -d "$build_root" ]; then
 mkdir -p "$build_root"
 tar -xzf "$archive" -C "$build_root" --strip-components=1
 patch -d "$build_root" -p1 < "$root/deploy/doomfly/native-observer/observer.patch"
fi
cmake -S "$build_root" -B "$build_root/build-observer" -DBUILD_PYTHON=OFF -DNO_OPENAL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$build_root/build-observer" --target vizdoom -j 4
binary="$build_root/build-observer/bin/vizdoom"
if [ -f "$build_root/build-observer/bin/vizdoom.app/Contents/MacOS/vizdoom" ]; then
 binary="$build_root/build-observer/bin/vizdoom.app/Contents/MacOS/vizdoom"
fi
destination="$root/outputs/connectome_sim/native-spectator-v1/engine"
mkdir -p "$destination"
cp "$binary" "$destination/vizdoom.next"
mv "$destination/vizdoom.next" "$destination/vizdoom"
"$root/.venv-neural/bin/python" - "$destination" <<'PY'
import shutil,sys
from pathlib import Path
import vizdoom
assert vizdoom.__version__=='1.3.0', 'The observer requires the same ViZDoom 1.3.0 assets as the primary'
shutil.copy2(Path(vizdoom.__file__).parent/'vizdoom.pk3',Path(sys.argv[1])/'vizdoom.pk3')
PY
