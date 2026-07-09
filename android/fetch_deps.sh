#!/bin/sh
# Clones the native dependencies (pinned tags) into app/src/main/cpp/third_party.
# Run once before the first Gradle build.
set -e
dest="$(dirname "$0")/app/src/main/cpp/third_party"
mkdir -p "$dest"

clone() { # name url tag
    if [ -d "$dest/$1/.git" ]; then
        echo "$1 already present, skipping"
    else
        git clone --depth 1 --branch "$3" "$2" "$dest/$1"
    fi
}
clone libusb https://github.com/libusb/libusb.git v1.0.27
clone libuvc https://github.com/libuvc/libuvc.git v0.0.7
echo "Done. Native deps are in $dest"
