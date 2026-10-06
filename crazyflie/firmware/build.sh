#!/usr/bin/env bash
# Build cf21bl firmware: ./build.sh (mocap_velctl app) or ./build.sh --stock; native arm-none-eabi-gcc, else Docker bitcraze/builder.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -d crazyflie-firmware/.git ]] || ./setup_firmware.sh

HERE=$(pwd)
if [[ ${1:-} == --stock ]]; then
    WORKDIR=$HERE/crazyflie-firmware
    OUT=crazyflie-firmware/build/cf21bl.bin
else
    WORKDIR=$HERE/mocap_velctl
    OUT=mocap_velctl/build/cf21bl.bin
fi
CMD="make cf21bl_defconfig && make -j$(nproc)"
if command -v arm-none-eabi-gcc >/dev/null; then
    (cd "$WORKDIR" && bash -c "$CMD")
else
    docker run --rm --user "$(id -u):$(id -g)" -v "$HERE":"$HERE" -w "$WORKDIR" bitcraze/builder bash -c "$CMD"
fi
ls -l "$OUT"
