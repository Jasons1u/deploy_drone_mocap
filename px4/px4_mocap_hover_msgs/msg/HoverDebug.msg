# Per-tick record from hover_node. Published on EVERY control tick, including gated and stale
# ticks, so a gap in this topic means the node stopped. `status` says why a tick commanded nothing.

std_msgs/Header header

uint8 STATUS_OK=0                  # policy ran, motors commanded
uint8 STATUS_NO_VEHICLE_STATUS=1   # no VehicleStatus received yet
uint8 STATUS_NOT_OFFBOARD=2        # waiting for ARMED + OFFBOARD (heartbeat only)
uint8 STATUS_STALE=3               # state inputs stale: all-NaN motors (PX4 disarms)
uint8 STATUS_NONFINITE=4           # policy returned NaN/Inf: all-NaN motors (PX4 disarms)
uint8 status
bool commanded                     # ActuatorMotors carried policy output this tick
uint8 nav_state                    # VehicleStatus.nav_state (255 = not received)
uint8 arming_state                 # VehicleStatus.arming_state (255 = not received)

# loop health
float32 loop_dt_ms                 # measured period (nominal 1000 / control_hz)
float32 inf_ms                     # ONNX inference only
float32 state_age_ms               # oldest state input in use (what the stale check compares)

# state fed to the policy: ENU world / FLU body, quaternions [w, x, y, z]
float32[3] pos_enu
float32[3] vel_enu
float32[4] quat_flu_enu
float32[3] ang_vel_flu
float32[3] mocap_pos_enu           # raw mocap position (NaN if no mocap), to audit the PX4 frame

# setpoint frame
float32[3] target_enu
float32 target_yaw                 # [rad] ENU yaw (0 = world +X, CCW positive)

# policy
float32[] obs
float32[4] raw                     # unclipped policy output
float32[4] motor_cmd               # [0, 1] published to /fmu/in/actuator_motors
