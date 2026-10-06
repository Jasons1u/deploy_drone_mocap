# Crazyflie 2.1 Brushless firmware: `mocap_velctl`

Out-of-tree app on the pinned firmware (2026.04 +2, `53142faa`):

- **Estimator** (`estimator_mocap.c`, forced by `CONFIG_ESTIMATOR_OOT`).
  - Position and velocity come from an alpha-beta filter on mocap poses streamed at 240 Hz.
  - Roll and pitch come from the IMU complementary filter.
  - Yaw is the IMU yaw plus an offset that tracks mocap.
- **Controller** (`controller_velocity.c`, selected with `stabilizer.controller = 6`).
  - Velocity setpoint (or position setpoint, through `kpXY`/`kpZ`) → acceleration `kv·(v_ref − v) + ki·∫`.
  - Acceleration → tilt and collective thrust in newtons, using `velCtl.mass`.
  - Then the stock attitude PID.
  - On this build, thrust commands are linear in newtons and battery-compensated.
  - The stock controllers (PID = 1, INDI = 3) remain selectable.

```bash
./build.sh                      # -> mocap_velctl/build/cf21bl.bin (./build.sh --stock for stock)
python3 test/validate_velctl.py # host checks of velctl_math.c: filter gains, signs, closed loop
python3 flash.py                # radio flash, STM32 only (--stock to revert)
```

Parameters (all writable at runtime):

| group | name | default | meaning |
|---|---|---|---|
| `mocapEst` | `rateHz` | 240 | mocap frame rate the host streams at |
| | `alpha`, `beta` | 0.54, 0.20 | alpha-beta gains (Kalata λ 0.3 for 0.3 mm noise) |
| | `maxDelay` | 0.7 | radio delay (periods) before a gap counts as a dropped frame |
| | `clockGain`, `minStep` | 0.01, 1.5 mm | frame-clock drift tracking; motion above which position settles a frame count |
| `velCtl` | `mass` | `CF_MASS` (39.3 g) | **set to the weighed drone, markers included** |
| | `kvX/Y/Z` | 4, 4, 6 | velocity loop [1/s]; closed loop ≈ first order with τ ≈ 1/kv |
| | `kiX/Y/Z`, `iLimXY/Z` | 0, 0, 0 | integral; with `kiZ = 0` a mass error gives `vz = −g(m/m_ctl − 1)/kvZ` |
| | `kpXY`, `kpZ`, `maxVelXY/Z` | 1.5, 1.5, 1, 1 | position setpoints only |
| | `maxTilt`, `maxAccUp/Down` | 20°, 6, 6 | limits |

Logs: `velCtl.{vRef*, aCmd*, thrustN, rollDes, pitchDes}` and `mocapEst.{ageMs, poses, drops, yawOffset}`.
