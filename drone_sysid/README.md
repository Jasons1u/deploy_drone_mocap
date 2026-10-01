# Crazyflie sysID for the safe_cf training sim

Flies identification tests through the same deploy chain the policy uses (mocap velocity → `mu = KPV·(uv − v) + 3.0` → appchannel → Ryan's firmware), then fits the parameters of the sim's velocity model (`tasks/safe_cf/mdp/velocity_action.py`). The firmware is not modified.

## Setup (once)

Mocap comes straight from Motive over NatNet; no ROS needed. Flights use the repo's conda env, and
the **same mocap and radio config as `cf-hover`** (`crazyflie/config/hover.yaml`): `mocap.server_ip`,
`rigid_body_id`, `up_axis`, and the `body_yaw_offset_deg` / `body_tilt_offset_deg` corrections from
`cf-hover --calibrate-yaw` / `--calibrate-tilt`. Use `--config` for a different file, or `--uri` to
override the radio.

```bash
./install.sh && conda activate drone_mocap        # from the repo root
cd drone_sysid
```

**Firmware:** these flights need **Ryan's firmware** (the appchannel `mu` controller). The repo's
mocap-estimator build (`crazyflie/firmware`) doesn't implement the appchannel, so `fly.py` refuses
to run if it detects it.

## Before every flight

1. Battery at least 4.0 V resting. Arming is refused below 3.70 V, and the flight lands below 3.30 V under load.
2. Mocap streaming this drone's rigid body:
   ```bash
   python3 check_mocap.py         # must print LIVE
   ```
3. Bench check with props off. This runs the whole pipeline, and the motors never arm:
   ```bash
   python3 fly.py vsteps --dry-run
   ```

## Fly

Press SPACE (or q) in the terminal to kill the motors. Each live run asks you to type `fly` before arming.

| plan | what it measures | duration |
|---|---|---|
| `hover` | hover point `mu_z` and trim (30 s hold at 0.8 m) | ~45 s |
| `vsteps` | gain, drag, response time and delay per axis (mirrored ±0.2/0.3/0.4 m/s steps in xy, ±0.10/0.18 m/s in z) | ~85 s |
| `thrust` | `alpha_z`, `b_z` and the vertical ceiling (open-loop `mu_z` 2→8 for 0.4 s each, from a settled hover at 0.5 m) | ~30 s |

```bash
python3 fly.py hover
python3 fly.py vsteps
python3 fly.py thrust
```

Logs go to `logs/sysid_<plan>_<time>.csv`, plus a `.json` sidecar with the settings (including the
mocap server, body and offsets) and events.

## Fit

```bash
python3 fit.py logs/sysid_*.csv --out fit.json
```

It prints two kinds of estimate:
- **classic:** step fits, mirrored-pair gain and trim, thrust slope and ceiling
- **model replay:** the sim's own model, with one shared delay, fitted to the logged velocities

It ends with suggested `VelocityActionCfg` values. Run several batteries or flights to get ranges for the randomization.

## Self-test without hardware

```bash
python3 validate_sim.py
```

This flies every plan against a simulated drone with known parameters, checks that the fit recovers them, and fault-injects every safety guard: low battery, stale/frozen/jumping mocap, geofence, kill key and dry run.

Note: on real thrust-ID data the replay fit and the thrust-slope fit disagree on `alpha_z`. The replay fit finds an effective delay of about 70 ms, against the sim's 20 ms. The sim's pure delay-plus-gain model likely misses motor or attitude lag.
