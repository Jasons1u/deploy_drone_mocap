# deploy_drone_mocap

Hover tests with OptiTrack motion capture as perception, on two vehicles:

| folder | vehicle | control | perception |
|---|---|---|---|
| `crazyflie/` | Crazyflie 2.x (cflib, plain Python) | onboard position controller, host streams setpoints | mocap → `extpose` → onboard Kalman |
| `px4/` | PX4 quad (ROS 2 Humble, uXRCE-DDS) | RL hover policy (ONNX) → offboard direct actuators | mocap → EKF2 vision fusion (or straight into the policy) |
| `mocap/` | shared | — | NatNet 3.x/4.x client (`mocap_natnet`), standard library only |

Ported from `x4/.../policy_runner` (hover path only) and `crazy_flies` (OptiTrack hover path only).

## Install (once)
```bash
cd ~/drone_ws/deploy_drone_mocap
./install.sh                  # creates the `drone_mocap` conda env (or updates it) + pip install -e ".[test]"
conda activate drone_mocap    # for cf-hover, python3 -m mocap_natnet, pytest
```
This installs `mocap_natnet`, `cf_mocap` and the `cf-hover` command into a dedicated conda env,
with the dependencies from `pyproject.toml`. The install is editable: `cf-hover` reads
`crazyflie/config/` and writes `crazyflie/logs/` in the checkout, and code edits apply
immediately. Rerun `./install.sh` after changing dependencies.

**Nothing goes into your normal Python.** cflib needs numpy 2, which breaks ROS 2 Humble's
compiled message packages (`px4_msgs` included), so the two must never mix:
- **The env can't see outside itself.** Activating it sets `PYTHONNOUSERSITE=1`, which hides
  `~/.local`, and clears `PYTHONPATH`, which hides ROS. Conda prints
  `overwriting variable ['PYTHONPATH']` on activation; that's expected.
- **`pip install` refuses anywhere else** (`tools/conda_env_guard.py`). That covers the system
  Python, conda base, and any conda pip that can see `~/.local`, before anything is installed.
  The last case matters because a conda python 3.10 shares `~/.local/lib/python3.10` with the
  system Python, and pip would uninstall packages from there.

**The PX4/ROS side does not use this env.** It runs on the system Python, built with colcon (see
[PX4](#px4)), so run `conda deactivate` before `colcon build` or `ros2 launch`. It needs only
`onnxruntime`, which is already installed there.

Don't add a `setup.py` at the repo root, or colcon will treat the whole repo as a single package.

---

## Conventions (read once)

- **World frame = the mocap frame, Z up.** x/y follow the Motive ground plane. PX4's local NED
  frame is this frame in NED (PX4 "North" = mocap +Y).
- **Body frame = FLU.** Create the rigid body in Motive **with the drone's nose along +X**, level.
- **Yaw** is ENU: 0 = mocap +X, counter-clockwise positive.
- **Height** is from the floor: put the Motive ground plane on the floor. The Crazyflie plan and
  the PX4 policy's `ground_prox` input both depend on it.

### Motive settings
- Streaming → **Broadcast Frame Data on**, transmission **Multicast** (`239.255.42.99`), local
  interface = the network the drone PC is on.
- **Up Axis: Z Up** (recommended), or set `up_axis: y` in the configs.
- Note the rigid body **Streaming ID**; it is `rigid_body_id` in the configs.

### Check mocap first (no drone needed)
```bash
conda activate drone_mocap
python3 -m mocap_natnet --server <MOTIVE_IP> --body 1
```
Check that the rate matches Motive, z is roughly marker height when on the floor, that moving
the drone nose-forward increases x at yaw ≈ 0, and that tilting it right gives roll > 0.
To bench-test without OptiTrack: `python3 -m mocap_natnet.fake_server --body 1 --pos 0 0 0.02`
in another terminal and use `--server 127.0.0.1`.

---

## Crazyflie

### Setup
The Crazyradio udev rules (`/etc/udev/rules.d/99-bitcraze.rules`) and the `plugdev` group are
already set up on this machine. Only one program can hold the radio: close `cfclient` first.

### Configure
Edit `crazyflie/config/hover.yaml`, at least `mocap.server_ip`. Any key can also be overridden per
run: `cf-hover --set flight.hover_height=0.6`.

### Run
```bash
conda activate drone_mocap
cf-hover --scan              # radio + drone visible? (expect radio://0/80/2M)
cf-hover --check-link        # connect + live roll/pitch/battery for 10 s: tilt it by hand (no mocap, no motors)
cf-hover --gamepad-test      # confirm the deadman axis (Xbox LT = 2) and land button (A = 0)
cf-hover --preflight-only    # full connect + Kalman convergence + checks, no motors
cf-hover --calibrate-yaw     # drone level, NOSE ALONG MOCAP +X: measure the rigid body's yaw offset (mocap only)
cf-hover --calibrate-tilt    # drone still: measure rigid-body vs IMU tilt, save to config (redo if the body changes)
cf-hover                     # fly: mocap is the state (default)
cf-hover --kalman            # fly: onboard Kalman filter (fed by mocap) is the state
cf-hover --kalman --filter-test   # carry the drone for 30 s: Kalman vs mocap error (never arms)
python3 crazyflie/scripts/plot_flight.py    # plot the newest log in crazyflie/logs/
python3 crazyflie/scripts/motor_test.py     # slider per motor (full range) + live PWM / battery / yaw plot
```

**Motor test** (`scripts/motor_test.py`). One slider per motor over the full PWM range (optional
cap: `--max-pwm`; lift-off is about 36000). Click ENABLE to spin the motors; STOP, the space bar or closing
the window stops them. It plots, all as reported by the drone:
- the PWM each motor actually gets (`motor.mN`),
- the battery sag (`pm.vbat`),
- the yaw rate (`gyro.z`).

To check spin directions, power one diagonal pair (M1+M3), then the other (M2+M4). The drone
should try to yaw in opposite directions for the two pairs. The Crazyflie has no thrust sensor;
for grams, stand it on a kitchen scale. **Warning:** the override (`motorPowerSet`) works below
every firmware safety net. If the radio link drops, the motors keep running at the last values.
Test with the props off or the drone strapped down, and keep a hand on the power switch.

**State source** (`flight.state_source`). The drone is a **Crazyflie 2.1 Brushless**.

| mode | firmware on the drone | what flies |
|---|---|---|
| **`mocap`** (default) | `crazyflie/firmware` (mocap passthrough estimator) | the host streams mocap (`extpose`); the drone's state *is* mocap; the **onboard INDI** controller flies position and attitude |
| `kalman` (`--kalman`) | stock | mocap → onboard Kalman filter → onboard controller |
| `host` | either | host-side controller on raw mocap (`cf_mocap/mocap_control.py`); the drone only stabilises attitude |

`flight.controller` picks the onboard controller: `indi` (default), `pid`, `mellinger`,
`brescianini` or `lee`. `cf-hover` detects which firmware is on the drone and refuses a mode that
doesn't match it.

### Firmware (`crazyflie/firmware/`)
```bash
crazyflie/firmware/build.sh                  # fetches the pinned firmware tree, builds in bitcraze/builder (Docker)
python3 crazyflie/firmware/flash.py          # flash the mocap-estimator build over the radio (STM32 only)
python3 crazyflie/firmware/flash.py --stock  # back to stock (built from the same tree; see flash.py)
```
- **What it is.** An out-of-tree estimator (`mocap_estimator/src/estimator_mocap.c`), about 150
  lines of C. It writes position straight from each mocap packet, velocity from a filtered
  difference of those positions, roll/pitch from the IMU's complementary filter, and yaw from the
  IMU corrected towards mocap yaw.
- **Why roll/pitch stay on the IMU.** The attitude loop runs at 500 Hz, while mocap reaches the
  drone at about 100 Hz with radio delay.
- **Tuning.** Parameters `mocapEst.velAlpha`, `mocapEst.velMinDtMs` and `mocapEst.yawGain`.
  `mocapEst.ageMs` (how long since the drone last got mocap) is watched by `cf-hover`, which cuts
  the motors if it goes stale.
- **Pinned version.** It's built against exactly what the drone shipped with, `2026.04 +2`
  (`53142faa`), for `cf21bl`. That predates bi-directional DShot and the RPM arming check, so the
  estimator is the only change.
- **This build always uses the passthrough.** With `CONFIG_ESTIMATOR_OOT` the firmware forces the
  out-of-tree estimator, so `--kalman` needs the stock firmware flashed back.

What `cf-hover` does:
1. Connects to mocap, then the drone. It checks the self-test and that the firmware matches the
   mode, then sets the estimator and controller.
   - `mocap` and `kalman` both stream `extpose` at 100 Hz.
   - `mocap` also waits until the drone confirms it is receiving mocap.
   - `kalman` resets the Kalman filter and waits for it to converge.
2. Runs the preflight checks: mocap fresh, inside the geofence, level, battery ≥ 3.6 V, and the
   drone agrees with mocap. That's the onboard position within 5 cm in `mocap`/`kalman` modes, or
   the IMU roll/pitch within 5° in `host` mode. The supervisor must also report "can be armed".
3. **Hold LT.** It re-checks, captures the takeoff XY and yaw, arms, then flies takeoff → hover
   (`hover_duration`) → land.

**Tuning the mocap controller** (`host_control:` in `config/hover.yaml`):
- **`hover_thrust`:** only a starting guess. The z integrator finds the real value within a few
  seconds; watch `cmd_thrust` in the log and put the settled value back into the config.
- **Gains:** checked in a simulation with 30 ms latency, 80 ms attitude lag and a hover thrust
  that's 7% off (`crazyflie/test/test_mocap_control.py`). Raise `kp_*`/`kv_*` only after a clean
  hover.

| event | result |
|---|---|
| A button, `hover_duration` elapsed, Ctrl+C | controlled landing |
| LT released, mocap lost > 0.2 s, geofence, tilt > 35°, Kalman ≠ mocap by > 15 cm (`--kalman`), Ctrl+C while landing | **E-STOP** (motors cut) |
| radio link lost | firmware levels out after 0.5 s and cuts motors after 2 s |

Every flight and filter test writes a CSV in `crazyflie/logs/` at 50 Hz, with one row per control
tick:
- setpoint
- mocap and Kalman side by side: position, velocity, roll/pitch/yaw, all in the same frame
  (right-handed, pitch > 0 = nose down)
- the host controller's commands `cmd_roll/pitch/yawrate/thrust` (mocap mode)
- motor PWM `m1..m4`
- battery

With `--kalman`, `est_* − mocap_*` is the onboard filter's error. In mocap mode the `est_*`
position/velocity columns are empty, since the drone has no position estimate.

**Why `--calibrate-tilt` matters.** The onboard Kalman filter trusts mocap's orientation tightly,
and it has no accelerometer-bias state. If the Motive rigid body's axes are tilted even 1–2°
relative to the IMU, the filter removes gravity along the wrong axis. The leftover shows up as a
constant false velocity (12 cm/s at rest was measured here, with a 1.9° roll / 0.7° pitch
offset). The calibration measures that offset from the accelerometer at rest, and every mocap
orientation is rotated by it (`mocap.body_tilt_offset_deg`). Rerun it whenever you recreate the
rigid body or move the markers.

**Why `--calibrate-yaw` matters.** Every controller turns "go that way" into a tilt using the
heading from mocap. If the rigid body's x axis isn't the drone's nose, every correction pushes
sideways by that angle, and the position error spirals outward. Here the body was 74.5° off, and
the flight log's heading test measured about −68°. Place the drone nose along +X and run
`--calibrate-yaw`; the tilt offset is re-expressed in the new frame automatically.

**What the filter test can and can't tell you.** With the motors off, the firmware's Kalman filter
runs its *not flying* model. That model integrates the full accelerometer and keeps pulling the
attitude toward level, with yaw toward `kalman.initialYaw`. The further you hold the drone from
level, the worse it gets: about 70 cm/s of velocity error at a 10° tilt was seen here. In flight,
the filter switches to a thrust + drag model with neither of those.
- **Use the filter test on the table:** the still-speed should be about 0 after `--calibrate-tilt`.
- **Judge in-flight estimate quality from a flight log:** compare `est_v*` with `mocap_v*`.

---

## PX4

### Build (once, and after code changes)
On the system Python, with conda deactivated. `px4_msgs` comes from the X4 workspace:
```bash
source /opt/ros/humble/setup.bash
source ~/x4/ros2_ws/install/setup.bash
cd ~/drone_ws/deploy_drone_mocap && colcon build      # builds mocap/ and px4/ (crazyflie/ is COLCON_IGNOREd)
source install/setup.bash
```

### PX4 parameters (EKF2 external vision)
With no GPS indoors, PX4 needs mocap for its own position. That is required both for
`state_source: px4` and for a POSITION-mode takeover by the pilot.

| param | value | why |
|---|---|---|
| `EKF2_EV_CTRL` | 11 (hpos + vpos + yaw) | fuse mocap position and heading |
| `EKF2_HGT_REF` | 3 (vision) | height from mocap, so z = 0 is the floor |
| `EKF2_GPS_CTRL` | 0 | no GPS indoors |
| `EKF2_MAG_TYPE` | none (5) | heading comes from mocap, not the magnetometer |
| `EKF2_EV_DELAY` | measure; start ~10–20 ms | Motive + network latency |
| `EKF2_EV_NOISE_MD` | 0 | use the variances the bridge reports |

Also check that `/fmu/in/vehicle_visual_odometry` is in the uXRCE `dds_topics.yaml` (it is in
stock PX4).

### Run
The bridge must run on a machine **wired** to the Motive network, ideally the one running the
MicroXRCEAgent. PX4 re-stamps incoming messages with the agent's clock, so if the bridge runs
elsewhere, keep the clocks in sync with chrony.

```bash
# everything on one machine
ros2 launch px4_mocap_hover hover.launch.py server_ip:=<MOTIVE_IP>

# split: ground station runs the bridge, the onboard computer runs the policy
ros2 launch px4_mocap_hover hover.launch.py with_policy:=false server_ip:=<MOTIVE_IP>
ros2 launch px4_mocap_hover hover.launch.py with_bridge:=false
```

Before arming, check:
- `ros2 topic hz /mocap/odom` shows the Motive frame rate.
- No `PX4 position differs from mocap` warning from `hover_node`. If there is one, EKF2 is not
  fusing vision.
- No `below the floor` warning from the bridge. If there is one, the up axis or ground plane is wrong.

Flight:
1. Take off in POSITION mode to about 1 m (the policy was trained for targets at 0.5–3 m).
2. Switch to **OFFBOARD**. The node captures the current position and yaw as the target, so
   nothing jumps.
3. Move the target only with **small relative nudges**:
   ```bash
   ros2 topic pub --once /hover/nudge geometry_msgs/msg/Vector3 '{z: 0.2}'     # ENU m, max 0.5
   ros2 topic pub --once /hover/nudge_yaw std_msgs/msg/Float32 '{data: 0.3}'   # rad CCW, max 0.5
   ```
4. Switch back to POSITION at any time to take over.

If state goes stale for more than 100 ms while engaged, the node sends NaN to all motors, which
PX4 treats as disarm. The vehicle drops rather than flying blind.

`state_source:=mocap` feeds the policy position, velocity and attitude straight from mocap, with
only the gyro from PX4. EKF2 vision fusion is still needed for the pilot's POSITION-mode fallback.

Telemetry: `/hover/debug` (every tick: status, state, obs, raw action, motor command, timing) and
`/hover/event` (latched transitions). Record with `ros2 bag record /hover/debug /hover/event /mocap/odom /fmu/out/vehicle_odometry`.

Config: `px4/px4_mocap_hover/config/{hover,mocap}.yaml`. `policy_onnx: hover.onnx` selects the
36-dim policy (with ground proximity); the default `new_hover.onnx` is the 35-dim retrain.

---

## Tests
```bash
conda activate drone_mocap
python3 -m pytest -q mocap/test crazyflie/test
(cd px4/px4_mocap_hover && python3 -m pytest -q test)
```
They cover:
- the NatNet parser for NatNet 3.0–4.2, and multicast loopback against the fake server
- frame conventions
- the Crazyflie flight plan, safety checks, and flight loop, including every E-STOP path, using fakes
- the policy obs layout and motor map, for both ONNX files

The policy obs and motor output were also checked against the original `policy_runner` code on
random states: they agree to within 3e-6.

## Not ported
From `crazy_flies`: baro/attitude hover, the open-loop test, the drag velocity estimator, sysid,
and the firmware checkout.
From `policy_runner`: trajectory tracking, figure-8, the plant/param witness plumbing, and the
large debug msgs.
Both source repos are unchanged if you need any of these.
