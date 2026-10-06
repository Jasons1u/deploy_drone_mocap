// Out-of-tree controller: velocity (or position) setpoint -> acceleration -> tilt + thrust [N] -> stock attitude PID.

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "attitude_controller.h"
#include "controller.h"
#include "log.h"
#include "math3d.h"
#include "param.h"
#include "platform_defaults.h"
#include "power_distribution.h"
#include "stabilizer_types.h"

#include "velctl_math.h"

#define ATTITUDE_UPDATE_DT (1.0f / ATTITUDE_RATE)
#define AIRBORNE_M 0.05f   // above the engage height, the horizontal integrators start

// --- tunables ---
static float mass = CF_MASS;      // [kg] set to the weighed drone incl. markers
static float kpXY = 1.5f;         // [1/s] position error -> velocity (position setpoints only)
static float kpZ = 1.5f;
static float maxVelXY = 1.0f;     // [m/s] position-mode velocity clamp
static float maxVelZ = 1.0f;
static float maxTiltDeg = 20.0f;
static float yawTrim = 0.0f;      // yaw mixer feedforward as a fraction of thrust (per-drone torque imbalance)
static float yawLim = 0.15f;      // max |yaw| mixer share (fraction of full scale) so roll/pitch keep authority

static velCtlGains_t gains = {
  .kv = {4.0f, 4.0f, 6.0f},
  .ki = {0.0f, 0.0f, 0.0f},
  .iLimit = {1.0f, 1.0f, 2.0f},
  .maxAccUp = 6.0f,
  .maxAccDown = 6.0f,
};

static velCtlState_t ctl;
static attitude_t attitudeDesired;   // legacy convention (pitch inverted), deg
static attitude_t rateDesired;
static bool engaged;
static float engageZ;

// --- logs ---
static float vRefLog[3];
static float aCmdLog[3];
static float thrustN;
static float yawCmd;

static float capAngle(float angle)
{
  while (angle > 180.0f) { angle -= 360.0f; }
  while (angle < -180.0f) { angle += 360.0f; }
  return angle;
}

static float clampf(float v, float lo, float hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

static void updateYaw(const setpoint_t *setpoint, const state_t *state)
{
  if (setpoint->mode.yaw == modeVelocity) {
    attitudeDesired.yaw = capAngle(attitudeDesired.yaw + setpoint->attitudeRate.yaw * ATTITUDE_UPDATE_DT);
    const float yawMaxDelta = attitudeControllerGetYawMaxDelta();
    if (yawMaxDelta != 0.0f) {
      const float delta = capAngle(attitudeDesired.yaw - state->attitude.yaw);
      if (delta > yawMaxDelta) {
        attitudeDesired.yaw = state->attitude.yaw + yawMaxDelta;
      } else if (delta < -yawMaxDelta) {
        attitudeDesired.yaw = state->attitude.yaw - yawMaxDelta;
      }
    }
  } else if (setpoint->mode.yaw == modeAbs) {
    attitudeDesired.yaw = setpoint->attitude.yaw;
  }
  attitudeDesired.yaw = capAngle(attitudeDesired.yaw);
}

static float axisVelRef(stab_mode_t mode, float posRef, float velRef, float pos, float kp, float vmax)
{
  if (mode == modeAbs) {
    return clampf(kp * (posRef - pos) + velRef, -vmax, vmax);
  }
  return velRef;
}

static void cutPower(control_t *control, const state_t *state)
{
  control->thrust = 0;
  control->roll = 0;
  control->pitch = 0;
  control->yaw = 0;
  thrustN = 0.0f;
  yawCmd = 0.0f;
  engaged = false;
  velCtlReset(&ctl);
  attitudeControllerResetAllPID(state->attitude.roll, state->attitude.pitch, state->attitude.yaw);
  attitudeDesired.yaw = state->attitude.yaw;
}

void controllerOutOfTreeInit(void)
{
  attitudeControllerInit(ATTITUDE_UPDATE_DT);
  velCtlReset(&ctl);
  engaged = false;
}

bool controllerOutOfTreeTest(void)
{
  return attitudeControllerTest();
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors,
                         const state_t *state, const stabilizerStep_t stabilizerStep)
{
  control->controlMode = controlModeLegacy;
  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    return;
  }

  const bool manualThrust = setpoint->mode.z == modeDisable;
  const bool manualTilt = setpoint->mode.x == modeDisable || setpoint->mode.y == modeDisable;
  if (manualThrust && setpoint->thrust == 0.0f) {
    cutPower(control, state);
    return;
  }
  if (!engaged) {
    engaged = true;
    engageZ = state->position.z;
  }

  updateYaw(setpoint, state);

  const float pos[3] = {state->position.x, state->position.y, state->position.z};
  const float vel[3] = {state->velocity.x, state->velocity.y, state->velocity.z};
  float vSp[3] = {setpoint->velocity.x, setpoint->velocity.y, setpoint->velocity.z};
  if (setpoint->velocity_body) {
    const float yawRad = radians(state->attitude.yaw);
    const float c = cosf(yawRad), s = sinf(yawRad);
    const float bx = vSp[0], by = vSp[1];
    vSp[0] = c * bx - s * by;
    vSp[1] = s * bx + c * by;
  }
  vRefLog[0] = axisVelRef(setpoint->mode.x, setpoint->position.x, vSp[0], pos[0], kpXY, maxVelXY);
  vRefLog[1] = axisVelRef(setpoint->mode.y, setpoint->position.y, vSp[1], pos[1], kpXY, maxVelXY);
  vRefLog[2] = axisVelRef(setpoint->mode.z, setpoint->position.z, vSp[2], pos[2], kpZ, maxVelZ);

  gains.maxTiltRad = radians(maxTiltDeg);
  gains.iLimit[1] = gains.iLimit[0];
  const bool airborne = state->position.z > engageZ + AIRBORNE_M;
  velCtlAccel(&ctl, &gains, vRefLog, vel, ATTITUDE_UPDATE_DT, airborne && !manualTilt, !manualThrust, aCmdLog);

  float zbDes[3];
  float lift;
  velCtlAccelToZb(&gains, aCmdLog, zbDes, &lift);

  if (manualTilt) {
    attitudeDesired.roll = setpoint->attitude.roll;
    attitudeDesired.pitch = setpoint->attitude.pitch;
  } else {
    float rollRad, pitchRad;
    velCtlZbToRollPitch(zbDes, radians(state->attitude.yaw), &rollRad, &pitchRad);
    attitudeDesired.roll = degrees(rollRad);
    attitudeDesired.pitch = -degrees(pitchRad);   // legacy: positive = nose up
  }

  const struct quat q = mkquat(state->attitudeQuaternion.x, state->attitudeQuaternion.y,
                               state->attitudeQuaternion.z, state->attitudeQuaternion.w);
  const struct vec zb = mcolumn(quat2rotmat(q), 2);
  const float zbNow[3] = {zb.x, zb.y, zb.z};
  if (manualThrust) {
    control->thrust = setpoint->thrust;
    thrustN = control->thrust / UINT16_MAX * powerDistributionGetMaxThrust();
  } else {
    thrustN = velCtlThrustN(mass, zbDes, lift, zbNow);
    control->thrust = clampf(thrustN / powerDistributionGetMaxThrust() * UINT16_MAX, 0.0f, UINT16_MAX);
  }

  attitudeControllerCorrectAttitudePID(state->attitude.roll, state->attitude.pitch, state->attitude.yaw,
                                       attitudeDesired.roll, attitudeDesired.pitch, attitudeDesired.yaw,
                                       &rateDesired.roll, &rateDesired.pitch, &rateDesired.yaw);
  attitudeControllerCorrectRatePID(sensors->gyro.x, -sensors->gyro.y, sensors->gyro.z,
                                   rateDesired.roll, rateDesired.pitch, rateDesired.yaw);
  attitudeControllerGetActuatorOutput(&control->roll, &control->pitch, &control->yaw);
  const float yawMax = yawLim * UINT16_MAX;
  yawCmd = clampf(-(float)control->yaw + yawTrim * control->thrust, -yawMax, yawMax);
  control->yaw = (int16_t)clampf(yawCmd, INT16_MIN, INT16_MAX);
}

PARAM_GROUP_START(velCtl)
PARAM_ADD(PARAM_FLOAT, mass, &mass)
PARAM_ADD(PARAM_FLOAT, kpXY, &kpXY)
PARAM_ADD(PARAM_FLOAT, kpZ, &kpZ)
PARAM_ADD(PARAM_FLOAT, kvX, &gains.kv[0])
PARAM_ADD(PARAM_FLOAT, kvY, &gains.kv[1])
PARAM_ADD(PARAM_FLOAT, kvZ, &gains.kv[2])
PARAM_ADD(PARAM_FLOAT, kiX, &gains.ki[0])
PARAM_ADD(PARAM_FLOAT, kiY, &gains.ki[1])
PARAM_ADD(PARAM_FLOAT, kiZ, &gains.ki[2])
PARAM_ADD(PARAM_FLOAT, iLimXY, &gains.iLimit[0])
PARAM_ADD(PARAM_FLOAT, iLimZ, &gains.iLimit[2])
PARAM_ADD(PARAM_FLOAT, maxVelXY, &maxVelXY)
PARAM_ADD(PARAM_FLOAT, maxVelZ, &maxVelZ)
PARAM_ADD(PARAM_FLOAT, maxTilt, &maxTiltDeg)
PARAM_ADD(PARAM_FLOAT, maxAccUp, &gains.maxAccUp)
PARAM_ADD(PARAM_FLOAT, maxAccDown, &gains.maxAccDown)
PARAM_ADD(PARAM_FLOAT, yawTrim, &yawTrim)
PARAM_ADD(PARAM_FLOAT, yawLim, &yawLim)
PARAM_GROUP_STOP(velCtl)

LOG_GROUP_START(velCtl)
LOG_ADD(LOG_FLOAT, vRefX, &vRefLog[0])
LOG_ADD(LOG_FLOAT, vRefY, &vRefLog[1])
LOG_ADD(LOG_FLOAT, vRefZ, &vRefLog[2])
LOG_ADD(LOG_FLOAT, aCmdX, &aCmdLog[0])
LOG_ADD(LOG_FLOAT, aCmdY, &aCmdLog[1])
LOG_ADD(LOG_FLOAT, aCmdZ, &aCmdLog[2])
LOG_ADD(LOG_FLOAT, thrustN, &thrustN)
LOG_ADD(LOG_FLOAT, yawCmd, &yawCmd)
LOG_ADD(LOG_FLOAT, rollDes, &attitudeDesired.roll)
LOG_ADD(LOG_FLOAT, pitchDes, &attitudeDesired.pitch)
LOG_GROUP_STOP(velCtl)
