#!/usr/bin/env bash
# NSL-KDD (Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/nsl.html). Mirror:
set -euo pipefail
mkdir -p data/raw && cd data/raw
for f in KDDTrain+.txt KDDTest+.txt; do curl -sSfO "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/$f"; done
wc -l *
