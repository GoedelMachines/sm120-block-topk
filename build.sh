#!/usr/bin/env bash
# Compile both extensions into ./build (no GPU attached; ~50 s each when the source changed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=none exec "$HERE/run_in_image.sh" python3 -c "
import sys; sys.path.insert(0, '$HERE')
import sm120_sparse_indexer as sp
for so in sp.build(): print('built', so)
"
