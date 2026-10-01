#!/usr/bin/env bash
# Fetch the exact Crazyflie firmware this app builds against, into ./crazyflie-firmware (ignored).
#
# Pinned to what the drone ran before this app: "2026.04 +2" (53142faa), a Crazyflie 2.1 Brushless.
# That predates bi-directional DSHOT (#1556) and the RPM arming check (#1641), so flashing this
# build changes nothing on the drone except the added estimator.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO=https://github.com/bitcraze/crazyflie-firmware.git
REF=53142faab672f00124ce326549767a18e0eb9108
DIR=crazyflie-firmware
LOCAL_MIRROR=${LOCAL_MIRROR:-$HOME/drone_ws/crazy_flies/firmware/crazyflie-firmware}  # optional speed-up

if [[ ! -d $DIR/.git ]]; then
    if [[ -d $LOCAL_MIRROR/.git ]]; then
        git clone --quiet --reference-if-able "$LOCAL_MIRROR" --dissociate "$REPO" "$DIR"   # self-contained: the build runs git in Docker
    else
        git clone --quiet "$REPO" "$DIR"
    fi
fi
git -C "$DIR" fetch --quiet origin
git -C "$DIR" -c advice.detachedHead=false checkout --quiet "$REF"
git -C "$DIR" submodule update --init --recursive --quiet
echo "firmware at $(git -C "$DIR" describe --tags) ($REF)"
