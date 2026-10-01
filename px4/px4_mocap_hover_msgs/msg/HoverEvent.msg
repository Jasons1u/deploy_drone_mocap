# A discrete transition in hover_node. Latched (TRANSIENT_LOCAL) so a late recorder still gets them.

std_msgs/Header header

uint8 NODE_STARTED=0
uint8 OFFBOARD_ACQUIRED=1
uint8 OFFBOARD_RELEASED=2
uint8 ENGAGE=3             # first commanded tick: target captured at the handover pose
uint8 SETPOINT_CHANGED=4   # nudge applied
uint8 STALE=5              # state went stale (one event per episode)
uint8 STALE_RECOVERED=6
uint8 NONFINITE_OUTPUT=7   # ONNX returned NaN/Inf
uint8 event

string reason
float32[3] pos_enu
float32[3] target_enu
float32 target_yaw
