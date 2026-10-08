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
| `crazyflie/fly_policy.py` | Real flights: manual sticks, RB hands control to the policy. |
| `crazyflie/sim_policy.py` | Desktop sim of a policy flight with draggable target and obstacles in RViz. |
| `crazyflie/viz/` | ROS 2 bridge, RViz layout, `run_rviz.sh`. |
| `crazyflie/firmware/` | The on-drone app (estimator + velocity controller); see its README. |
| `policy/` | `export_policy.py` (Isaac checkpoint → `.npz`) and `models/` (`safecf_moving_good.npz` = current, `_bad` = kept for the record). |

Nothing is installed: every script adds the repo root to its path. Run the Python scripts in the `drone` conda env; the RViz side uses the system Python with ROS 2 Humble (`run_rviz.sh` does that for you).

## Settings

Edit `config/deploy.yaml`, or override a value for one run:

```bash
--set sticks.max_xy=0.3 --set gamepad.buttons.rb=4     # any key, YAML value
--param pid_attitude.roll_kp=8                          # extra firmware parameter (RAM) for this run
```

The `training` section must match the Isaac `safe_cf` env the policy was trained in; the `drone_model` section is the sysID fit (the same numbers as Isaac's `VelCtlActionCfg`).

## Flying

All flights: **hold LT** to arm and fly (release = motors off), SPACE in the terminal = E-STOP, Ctrl+C = land. Lift-off is a short level burst at 65% of max thrust (about 1.2× hover), handed to the controller 3 cm up, then a climb ramp. `--dry-run` streams mocap and runs the preflight checks without arming.

```bash
conda run --no-capture-output -n drone python3 crazyflie/sysid_flight.py                 # hover check
conda run --no-capture-output -n drone python3 crazyflie/sysid_flight.py --plan vsteps --axes x
conda run --no-capture-output -n drone python3 crazyflie/fit_vsteps.py --cascade crazyflie/logs/<run> ...
```

Policy flights (`fly_policy.py`):

```bash
crazyflie/viz/run_rviz.sh        # optional: RViz, drag the target
conda run --no-capture-output -n drone python3 crazyflie/fly_policy.py --height 1.2 --loop --viz
```

- **Sticks:** right = horizontal velocity (heading-relative), left = climb / yaw.
- **RB:** manual ↔ policy. The policy flies to the target (the launch spot, or where you drag it in RViz) and only engages inside the trained region, slow, with every obstacle tracked.
- **Back to manual** on RB, a stick push, a lost obstacle, or a braking point past the soft arena boundary.
- **A** lands; with `--loop`, **Y** flies again (restarting a locked drone), **B** quits.
- **Obstacles:** Motive rigid bodies listed under `obstacles.tracked` (25 and 26); keep them at z 0.5–2.0 m within 1.5 m of the centre (where training put them). All must be tracked to engage; losing any one hands back to manual.
- **Radio:** poses go out newest-first from their own thread (never queued); telemetry is lighter in policy flights (`telemetry.flight`). The terminal warns `radio: telemetry N ms late` when the downlink backs up, and the estimate E-STOP compares against mocap at the drone's own sample time. Each run logs `link.csv` (link quality, ping latency, rates) and `extpose.csv` (every pose sent, how long cflib blocked).
- **Two-obstacle policies (33 inputs, Isaac `Moving2`)** are picked up from the policy file and see the two nearest; with only one obstacle the second slot reads empty, as in their one-obstacle training episodes.
- **Two obstacles with a single-obstacle policy:** the policy only sees the closest; with both about equally far on opposite sides it flips between them and the commands jump. That's the trained behaviour, not a deploy bug.

## Simulating

```bash
conda run --no-capture-output -n drone python3 crazyflie/sim_policy.py      # also starts RViz
```

Drag the green ball (target, sent on release) or the orange shell (obstacle, follows live). Keys in the terminal: `r` reset drone, `o` reset obstacles, `p` pause, `q` quit. Options: `--randomize` (model drawn from the training ranges), `--speed 0.5`, `--obstacle X Y Z R`, `--no-rviz`. Logs go to `crazyflie/logs/<time>_sim/`.

## Exporting a policy

```bash
conda run -n drone python3 policy/export_policy.py <isaac run>/model_1499.pt policy/models/<name>.npz \
    --critic-dim 96 --task Isaac_SafeCF_Moving_SAFEPPO_CF21BL
```

Then check the observation chain against Isaac with `scripts/verify_safecf_deploy_obs.py --policy <npz>` in the IsaacLab repo.
