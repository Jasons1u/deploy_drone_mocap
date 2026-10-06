#!/usr/bin/env bash
# Fetch the pinned Crazyflie firmware tree (2026.04 +2, 53142faa, what the CF2.1 Brushless shipped with) into ./crazyflie-firmware.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO=https://github.com/bitcraze/crazyflie-firmware.git
REF=53142faab672f00124ce326549767a18e0eb9108
DIR=crazyflie-firmware

[[ -d $DIR/.git ]] || git clone --quiet "$REPO" "$DIR"
git -C "$DIR" fetch --quiet origin
git -C "$DIR" -c advice.detachedHead=false checkout --quiet "$REF"
git -C "$DIR" submodule update --init --recursive --quiet
echo "firmware at $(git -C "$DIR" describe --tags) ($REF)"
