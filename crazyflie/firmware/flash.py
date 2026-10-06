#!/usr/bin/env python3
"""Flash the Crazyflie 2.1 Brushless STM32 over the radio: mocap_velctl build (default) or --stock."""

import argparse
import sys
from pathlib import Path

import cflib.crtp
from cflib.bootloader import Bootloader, Target

HERE = Path(__file__).resolve().parent
IMAGES = {
    "mocap_velctl": HERE / "mocap_velctl" / "build" / "cf21bl.bin",
    "stock": HERE / "crazyflie-firmware" / "build" / "cf21bl.bin",
}
STM32_FIRMWARE = Target("cf2", "stm32", "fw", [], [])   # bootloader platform id for all CF2.x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock", action="store_true", help="flash stock 2026.04 +2 (same pinned tree, no app)")
    ap.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7")
    args = ap.parse_args(argv)

    image = IMAGES["stock" if args.stock else "mocap_velctl"]
    if not image.is_file():
        print(f"{image} not found; build it first: {HERE / 'build.sh'}{' --stock' if args.stock else ''}",
              file=sys.stderr)
        return 1
    print(f"Flashing {image.relative_to(HERE)} ({image.stat().st_size} bytes) to {args.uri} (STM32 only)")

    def progress(msg, percent):
        print(f"\r  {msg} {percent:3d}%", end="", flush=True)

    cflib.crtp.init_drivers()
    Bootloader(args.uri).flash_full(filename=str(image), warm=True, targets=(STM32_FIRMWARE,),
                                    progress_cb=progress)
    print("\nDone; the drone is rebooting into the new firmware.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
