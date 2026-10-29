# deploy_drone_mocap

Crazyflie 2.1 Brushless flights in the OptiTrack arena: our own firmware app on the drone, and host scripts that fly it from Motive data, run the Isaac-trained `safe_cf` policy, and simulate it on the desktop.

```
Motive ──NatNet──► cfdeploy/natnet.py ──► crazyflie/*.py (host, 100 Hz) ──radio──► firmware mocap_velctl
policy/models/*.npz + cfdeploy/safe_cf_obs.py ──► fly_policy.py / sim_policy.py ◄──UDP──► crazyflie/viz (ROS 2 + RViz)
```

## Layout

| Path | What it is |
|---|---|
| `config/deploy.yaml` | **Every setting** (mocap, radio, drone, gamepad, limits, obstacles, sim, drone model, training constants). |
| `cfdeploy/` | Shared library: config, NatNet client, radio link + telemetry, gamepad, velocity filter, safety, flight phases, obstacles, targets, policy obs, drone model, RViz link, logs. |
| `crazyflie/sysid_flight.py` | Hover check (`--plan hover`) and velocity-step sysID flights (`--plan vsteps`). |
| `crazyflie/fit_vsteps.py` | Fits the drone's velocity response from vsteps logs (`--cascade` = the model Isaac uses). Results: `crazyflie/SYSID.md`. |
| `crazyflie/fly_rates.py` | Rate-mode flights: props-off sign bench, rate/thrust doublet sysID, mj-nlp flip tape (conversions: `cfdeploy/rate_tape.py`; firmware loop model + tape compensation: `cfdeploy/rate_loop.py`). |
| `crazyflie/fit_rates.py` | Fits the firmware rate loop (Ixx, Iyy, motor lag, delay) and the thrust lag from rsteps logs; prints the mj-nlp XML values. |
| `crazyflie/fly_policy.py` | Real flights: manual sticks, RB hands control to the policy. |
| `crazyflie/sim_policy.py` | Desktop sim of a policy flight with draggable target and obstacles in RViz (sim world: `cfdeploy/sim.py`). |
| `crazyflie/analyze_flight.py` | Flight-log diagnosis: launches and radio delays. |
| `crazyflie/plot_safety.py` | Safety plots of policy flights: paths, clearances, training cost r1, SafePPO cost critic. |
| `crazyflie/plot_dodges.py` | Showcase of policy flights: the drone dodging hand-moved obstacles (figure + a video per dodge). |
| `crazyflie/viz/` | ROS 2 bridge, RViz layout, `run_rviz.sh`. |
| `crazyflie/firmware/` | The on-drone app (estimator + velocity controller); see its README. |
| `policy/` | `export_policy.py` (Isaac checkpoint → `.npz`), `evaluate_policy.py` (sim comparison) and `models/` (`safecf_moving_good.npz` = current, `_bad` = kept for the record). |

Nothing is installed: every script adds the repo root to its path. Run the Python scripts in the `drone` conda env; the RViz side uses the system Python with ROS 2 Humble (`run_rviz.sh` does that for you).

## Settings

Edit `config/deploy.yaml`, or override a value for one run:

```bash
--set sticks.max_xy=0.3 --set gamepad.buttons.rb=4     # any key, YAML value
--param pid_attitude.roll_kp=8                          # extra firmware parameter (RAM) for this run
```

The `training` section must match the Isaac `safe_cf` env the policy was trained in; the `drone_model` section is the sysID fit (the same numbers as Isaac's `VelCtlActionCfg`).

## Flying

All flights: **hold LT** to arm and fly (release = motors off), SPACE in the terminal = E-STOP, Ctrl+C = land. Mocap dropouts shorter than `safety.mocap_timeout` (0.1 s) ride through on the last tracked pose; longer ones E-STOP. Lift-off is a short level burst at 65% of max thrust (about 1.2× hover), handed to the controller 3 cm up, then a climb ramp. `--dry-run` streams mocap and runs the preflight checks without arming.

```bash
conda run --no-capture-output -n drone python3 crazyflie/sysid_flight.py                 # hover check
conda run --no-capture-output -n drone python3 crazyflie/sysid_flight.py --plan vsteps --axes x
conda run --no-capture-output -n drone python3 crazyflie/fit_vsteps.py --cascade crazyflie/logs/<run> ...
```

Policy flights (`fly_policy.py`):

```bash
conda run --no-capture-output -n drone python3 crazyflie/fly_policy.py --height 1.2 --loop --viz   # --viz: starts RViz, drag the target
```

- **Sticks:** right = horizontal velocity (heading-relative), left = climb / yaw.
- **RB:** manual ↔ policy. The policy flies to the target (the launch spot, or where you drag it in RViz) and only engages inside the trained region, slow, with every obstacle tracked.
- **Back to manual** on RB, a stick push, a lost obstacle, or a braking point past the soft arena boundary.
- **A** lands; with `--loop`, **Y** flies again (restarting a locked drone), **B** quits.
- **Obstacles:** Motive rigid bodies listed under `obstacles.tracked` (25 and 26); keep them at z 0.5–2.0 m within 1.5 m of the centre (where training put them). All must be tracked to engage; losing any one hands back to manual.
- **Radio:** poses go out newest-first from their own thread (never queued); telemetry is lighter in policy flights (`telemetry.flight`). The terminal warns `radio: telemetry N ms late` when the downlink backs up, and the estimate E-STOP compares against mocap at the drone's own sample time. Each run logs `link.csv` (link quality, ping latency, rates) and `extpose.csv` (every pose sent, how long cflib blocked).
- **Two-obstacle policies (33 inputs, Isaac `Moving2`)** are picked up from the policy file and see the two nearest; with only one obstacle the second slot reads empty, as in their one-obstacle training episodes.
- **Two obstacles with a single-obstacle policy:** the policy only sees the closest; with both about equally far on opposite sides it flips between them and the commands jump. That's the trained behaviour, not a deploy bug.

## Rate mode: bench, rate sysID, flip

`fly_rates.py` sets `flightmode.stabModeRoll/Pitch = 0` after the climb, so legacy setpoints carry body rates (deg/s) and thrust straight to the rate PID. It restores angle mode before landing and on exit. Run these in order, each one only after the previous passes:

```bash
./crazyflie/firmware/build.sh && python3 crazyflie/firmware/flash.py                          # rate pass-through + estimator gates
python3 tests/test_rate_mode.py                                                               # offline: conventions, compensation, state machine, fit
conda run --no-capture-output -n drone python3 crazyflie/fly_rates.py --plan bench            # PROPS OFF: PASS table per axis and sign
conda run --no-capture-output -n drone python3 crazyflie/sysid_flight.py                      # normal hover on the new firmware
conda run --no-capture-output -n drone python3 crazyflie/fly_rates.py --plan rsteps           # ~70 s at 1 m: 30 doublets
conda run --no-capture-output -n drone python3 crazyflie/fit_rates.py crazyflie/logs/<run>_rsteps
conda run --no-capture-output -n drone python3 crazyflie/fly_rates.py --plan flip --tape <mj-nlp>/examples/crazyflie_flip/crazyflie_flip_<stamp>.npz --dry-run
conda run --no-capture-output -n drone python3 crazyflie/fly_rates.py --plan flip --tape <same .npz>   # sticks, X = flip here, A = land
```

- **Bench** (props off, drone on the floor; no mocap needed): arms, then spins the motors at `bench_thrust` for each of ±p, ±q, ±r in turn. It passes when the logged `velCtl.rateP/Q/R` equals the command, `thrustN` equals the bench thrust, and the matching motor pair speeds up (`ALLOC`). This checks the whole sign chain, host to cflib to commander to rate PID to mixer.
- **rsteps**: hover, trim the hover thrust, then rate doublets on p, q, r and thrust doublets on T. These are each flown 3 times, at most 29 deg of tilt. Before each doublet the rate and motor log blocks restart at a random phase, so the 10 ms samples don't stay locked to the 10 ms setpoints.
- **Fit**: `fit_rates.py` fits the roll and pitch doublets with `cfdeploy/rate_loop.py`. That is the stock rate PID (cf21bl gains, D on the gyro, 500 Hz, 80 Hz gyro filter), the legacy mixer's int16 clamp and motor cap, and first-order motors. The fit gives Ixx, Iyy, one motor lag and the delay. A first-order rate model misses the PID's ~20% overshoot (0.88 vs 0.14 rad/s rms on the 2026-10-28 run). It also fits the collective lag on the mocap vertical velocity as a check (about the motor lag plus mocap latency). It prints the XML lines for mj-nlp `models/crazyflie/crazyflie.xml` (inertia, `kv`, `motor_lag`); paste them in and re-solve the flip.
- **flip**: climb to `flip_height`, then fly it on the sticks as in fly_policy's manual mode (right = horizontal, left = climb / yaw). **X** flips centred where it is, at that height, about its current heading's pitch axis: it holds there for `flip_settle` and trims. Then it streams the tape at 100 Hz, with thrust scaled by the trimmed hover thrust. It holds `recover` at the X point and goes back to the sticks; X again flies another flip, **A** lands. X is refused (printed) where the tape's path plus `flip_margin` would leave the soft arena limits; the E-STOP limit is the arena, as in fly_policy. The airborne tilt guard and the mocap checks are off from the tape's start until `guard_extra` after it: the rate loop runs on the gyro alone, and the markers face the floor while inverted. If mocap is still lost then, the drone holds level in angle mode at the trimmed hover thrust for up to `blind_hold`. It resumes when mocap returns and E-STOPs if it doesn't. A land request (A, Ctrl+C) during the tape waits for the tape to end. Deadman, SPACE and the arena limit still E-STOP mid-tape.
- **Compensation**: the planner flies an instant P rate loop, so `fly_rates.py` rewrites the tape with `rate_loop.compensate` before flying. It inverts the fitted loop so the real PID and motors reproduce the planned p and q rates and their angle. Thrust is inverted through the motor lag the same way. The inertia and motor lag come from the tape, which mj-nlp saves with the model it solved.
- mj-nlp `python tests/test_crazyflie.py deploy` flies these exact packets through `rate_loop.RateLoop` in MuJoCo, compensated and not. It also flies them under inertia, motor-lag, delay, mass, jitter and drop errors.

## Simulating

```bash
conda run --no-capture-output -n drone python3 crazyflie/sim_policy.py      # also starts RViz
```

Drag the green ball (target, sent on release) or the orange shell (obstacle, follows live). Keys in the terminal: `r` reset drone, `o` reset obstacles, `p` pause, `q` quit. Options: `--randomize` (model drawn from the training ranges), `--speed 0.5`, `--obstacle X Y Z R`, `--no-rviz`. Logs go to `crazyflie/logs/<time>_sim/`.

## Evaluating a policy or a flight

```bash
conda run -n drone python3 policy/evaluate_policy.py policy/models/a.npz policy/models/b.npz   # side-by-side sim tests
conda run -n drone python3 crazyflie/analyze_flight.py crazyflie/logs/<run> ...                  # launches + radio delays
conda run -n drone python3 crazyflie/plot_safety.py crazyflie/logs/<run or folder of runs>       # safety plots -> <folder>/plots
conda run -n drone python3 crazyflie/plot_dodges.py crazyflie/logs/<run or folder of runs>       # dodge showcase -> <folder>/plots
```

`evaluate_policy.py` runs every policy through the same sim checks (parking near a target, an obstacle flown through the hold point, a recorded two-obstacle flight replayed, hover calm, stand-off from a nearby obstacle) and prints one table per check. `analyze_flight.py` reports each launch (burst, tilt, which side dropped, battery sag) and the radio (telemetry delay from the drone's timestamps, link statistics, pose sends).

## Exporting a policy

```bash
conda run -n drone python3 policy/export_policy.py <isaac run>/model_1499.pt policy/models/<name>.npz \
    --critic-dim 100 --task Isaac_SafeCF_SAFEPPO_CF21BL
```

Then check the observation chain against Isaac with `scripts/verify_safecf_deploy_obs.py --policy <npz>` in the IsaacLab repo.
