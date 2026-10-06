# Velocity-loop sysID: CF2.1 Brushless + `mocap_velctl` firmware (2026-10-01)

Flights with `hover.py --plan vsteps` and attitude kp 9. Fits are from `fit_vsteps.py`; `--cascade` fits the model used in Isaac.

## Configuration this model is valid for

The deploy side must reproduce all of it:

| setting | value |
|---|---|
| firmware | `crazyflie/firmware/mocap_velctl` (alpha-beta 0.54/0.20 @ 240 Hz, velocity controller = `stabilizer.controller` 6) |
| attitude PID | `pid_attitude.roll_kp` = `pitch_kp` = **9** (stock 6), RAM only, so set it on every connect |
| velocity loop | `velCtl.kvX/kvY/kvZ` = 4/4/6, `maxTilt` 20°, `maxAccUp/Down` 6, `mass` 0.044 kg |
| integrators | `kiXY` 1 / `kiZ` 2 during takeoff and hover (trim), then **frozen** (`ki` = 0) before velocity commands |
| setpoints | `send_velocity_world_setpoint` at 100 Hz, yaw rate 0 |
| mocap | every frame forwarded at 240 Hz, **wired Ethernet** (Wi-Fi stalls up to 136 ms) |

## Model (Isaac `VelCtlAction`, `tasks/safe_cf/mdp/velctl_action.py`)

`a_cmd = kv·(cmd(t − d_cmd) − v(t − d_fb))`, with |a_xy| ≤ a_max and a_z clamped to ±6. `a` lags `a_cmd` with time constant τ_a, and `v' = a − drag·v`.

| | xy (24 step edges) | z (12 step edges) |
|---|---|---|
| kv (firmware) | 4 | 6 |
| τ_a | 77 ms (x 75, y 79) | 41 ms |
| d_cmd | 22 ms (x 20, y 24) | 20 ms |
| d_fb (estimator) | 20 ms | 20 ms |
| a_max | 3.53 m/s² ≈ g·tan 20° | not reached (firmware ±6) |
| drag | 0 | 0.45 /s (steady-state gain 0.93) |
| edge RMS (fit) | 0.021 / max 0.047 m/s | 0.019 / max 0.029 m/s |

Isaac replay (`isaac_drone_racer/scripts/validate_velctl_rom.py`), velocity RMS against mocap. The old `VelocityAction` (brushed `mu` chain) is in brackets:
- x: 0.027, max 0.049 [0.111 / 0.282]
- y: 0.014, max 0.025 [0.112 / 0.260]
- z: 0.019, max 0.027 [0.076 / 0.222]

The new model is better on 36 of 36 edges.

## First-order view, for reference

| | delay | τ | gain | peak \|a\| (1 m/s steps) |
|---|---|---|---|---|
| x/y | ~100 ms | ~160 ms | 1.0 | 2.5–3.0 m/s² |
| z | ~46 ms | ~115 ms | 0.94 | — |

The first-order delay shrinks to ~57 ms on 1 m/s steps (tilt builds faster), which is why the cascade replaced it.

## Findings

- **Attitude kp 6 → 9:** xy delay 130 → ~97 ms, pitch lag 104 → 86 ms, accel cap +12%, less cross-coupling. Roll briefly showed a 5.8 Hz component, so going past 9 needs care.
- **Yaw torque imbalance:** in hover the yaw loop holds 7–10% of thrust as diagonal differential (m1/m3 work harder). At spin-up it saturates the motors, so a slow takeoff tips the drone. `hover.py` launches with a ≤1 s level full-thrust burst. Check the props and motors (M2 pinned at 100% while its corner dropped).
- **IMU vs body:** −2.3° roll / +1.4° pitch offset. `imu_sensors.imuPhi/imuTheta` could correct it.

## Logs (`crazyflie/logs/`)

| run | content |
|---|---|
| `20261001_200713` | x vsteps, attitude kp 6 (stock) |
| `20261001_203358` | x vsteps, kp 9 |
| `20261001_204813` | y + z vsteps, kp 9 |
