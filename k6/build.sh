#!/bin/bash
set -euo pipefail

# This script builds the k6 test archive inside the k6 directory.
# It should be run from the project root, e.g., ./k6/build.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Creating build directory..."
mkdir -p "$SCRIPT_DIR/build"

echo "Building k6 archive..."
# We mount the k6 directory (where the script lives) to /home/k6 inside the container
docker run --rm -i \
  -u "$(id -u):$(id -g)" \
  -v "$SCRIPT_DIR:/home/k6:ro" \
  -v "$SCRIPT_DIR/build:/home/k6/build:rw" \
  -w /home/k6 \
  grafana/k6:1.1.0 archive -O /home/k6/build/k6_archive.tar script.js

echo "k6 archive created at k6/build/k6_archive.tar" 