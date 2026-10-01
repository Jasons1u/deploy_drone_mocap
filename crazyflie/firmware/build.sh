#!/usr/bin/env bash
# Build Crazyflie 2.1 Brushless firmware in Bitcraze's build image (Docker).
#
#     ./build.sh           the mocap-estimator app  -> mocap_estimator/build/cf21bl.bin
#     ./build.sh --stock   stock, same pinned tree  -> crazyflie-firmware/build/cf21bl.bin
#
# Flash either with ./flash.py [--stock].
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -d crazyflie-firmware/.git ]] || ./setup_firmware.sh

HERE=$(pwd)
if [[ ${1:-} == --stock ]]; then
    WORKDIR=$HERE/crazyflie-firmware
    OUT=crazyflie-firmware/build/cf21bl.bin
else
    WORKDIR=$HERE/mocap_estimator
    OUT=mocap_estimator/build/cf21bl.bin
fi
docker run --rm --user "$(id -u):$(id -g)" -v "$HERE":"$HERE" -w "$WORKDIR" \
    bitcraze/builder bash -c "make cf21bl_defconfig && make -j$(nproc)"
ls -l "$OUT"
