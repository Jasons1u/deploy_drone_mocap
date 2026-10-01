#!/usr/bin/env python3
"""Flash the Crazyflie 2.1 Brushless over the radio: main (STM32) firmware only.

    python3 crazyflie/firmware/flash.py            # the mocap-estimator build (./build.sh)
    python3 crazyflie/firmware/flash.py --stock    # back to stock 2026.04 +2 (same tree, no app)

The drone must be switched on and running firmware (it is rebooted into the bootloader over the
radio). The radio chip (nRF51) and its bootloader are not touched. Build the images first:
./build.sh (mocap estimator) and ./build.sh --stock (stock, from the same pinned tree).
"""

import argparse
import sys
from pathlib import Path

import cflib.crtp
from cflib.bootloader import Bootloader, Target

from cf_mocap.config import load_config

HERE = Path(__file__).resolve().parent
IMAGES = {
    "mocap": HERE / "mocap_estimator" / "build" / "cf21bl.bin",
    "stock": HERE / "crazyflie-firmware" / "build" / "cf21bl.bin",
}
STM32_FIRMWARE = Target("cf2", "stm32", "fw", [], [])   # bootloader platform id for all CF2.x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stock", action="store_true", help="flash the stock image instead")
    ap.add_argument("--uri", help="radio URI (default: radio.uri from crazyflie/config/hover.yaml)")
    args = ap.parse_args(argv)

    image = IMAGES["stock" if args.stock else "mocap"]
    if not image.is_file():
        flag = " --stock" if args.stock else ""
        print(f"{image} not found — build it first: {HERE / 'build.sh'}{flag}", file=sys.stderr)
        return 1
    uri = args.uri or load_config().radio.uri
    print(f"Flashing {image.relative_to(HERE)} ({image.stat().st_size} bytes) to {uri} (STM32 only)")

    def progress(msg, percent):
        print(f"\r  {msg} {percent:3d}%", end="", flush=True)

    cflib.crtp.init_drivers()
    Bootloader(uri).flash_full(filename=str(image), warm=True, targets=(STM32_FIRMWARE,),
                               progress_cb=progress)
    print("\nDone; the drone is rebooting into the new firmware.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
