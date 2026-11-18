#!/usr/bin/env bash
# Default fixed-mass, two-category 4+1D Upsilon fit.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 TENSOR.hdf5 OUTPUT_DIRECTORY" >&2
    exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cell_axes=eta1,eta2,pt1,pt2
options=()
for channel in barrel high; do
    options+=(
        --paramModel AxisNormModel "$channel" "Y1S_${channel},Y2S_${channel},Y3S_${channel}" "$cell_axes" pou
        --paramModel AxisExpModel "$channel" "background_${channel}" mass "$cell_axes" pou constraint:10:2
    )
done
for axis in mass eta1 eta2 pt1 pt2; do
    combined=inclusive
    [[ "$axis" == mass ]] || combined="inclusive_${axis}"
    options+=(
        -m Project barrel "$axis" -m Project high "$axis"
        -m scripts.rabbit.upsilon_mappings.SumUpsilonChannels "$combined" "$axis"
    )
done

python3 "$repo/rabbit/bin/rabbit_fit.py" "$1" -o "$2" -t 0 \
    --noChi2 --saveHists --saveHistsPerProcess --computeHistErrors \
    --hessianChunkSize 256 "${options[@]}"
