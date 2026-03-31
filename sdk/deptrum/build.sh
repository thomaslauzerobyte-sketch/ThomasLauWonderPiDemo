#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Building deptrum_bridge.so …"
g++ -shared -fPIC -O2 -std=c++17 \
    -I include \
    -L lib -Wl,--no-as-needed -ldeptrum_stream_aurora900 \
    -Wl,-rpath,'$ORIGIN' \
    -o lib/libdeptrum_bridge.so \
    deptrum_bridge.cpp

echo "Done → lib/libdeptrum_bridge.so"
ls -lh lib/libdeptrum_bridge.so
