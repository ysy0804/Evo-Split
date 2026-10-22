#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/build-demo"
RESULTS="$ROOT/demo-results"
mkdir -p "$RESULTS"
cmake -S "$ROOT" -B "$BUILD" -DEVOSPLIT_WITH_MNN=OFF -DEVOSPLIT_BUILD_TESTS=ON -DEVOSPLIT_BUILD_DEMO=ON
cmake --build "$BUILD" --config Release
"$BUILD/evosplit_demo" --output "$RESULTS/demo_trace.csv"
python3 "$ROOT/examples/minimal_demo/plot_demo.py" --input "$RESULTS/demo_trace.csv" --output "$RESULTS/demo_summary.png"

