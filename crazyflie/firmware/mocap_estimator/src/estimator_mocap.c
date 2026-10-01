/*
 * Mocap passthrough estimator (out-of-tree estimator for the Crazyflie firmware).
 *
 * The onboard controller (INDI, PID, ...) flies on `state`. This estimator fills it from motion
 * capture directly instead of fusing it in a Kalman filter:
 *
 *   position     the latest extpose/extpos packet, as is                       (mocap)
 *   velocity     finite difference of those positions, low-pass filtered        (mocap)
 *   roll, pitch  the stock complementary filter (gyro + accelerometer)          (IMU)
 *   yaw          complementary filter yaw + an offset that tracks mocap yaw     (IMU + mocap)
 *
 * Roll/pitch stay on the IMU on purpose: the attitude loop runs at 500 Hz, while mocap reaches the
 * drone at ~100 Hz with radio delay. Yaw needs mocap because the gyro alone drifts.
 *
 * With CONFIG_ESTIMATOR_OOT the firmware forces this estimator (estimator.c), so it is the only one
 * on this build. Nothing here is flight-critical logic beyond copying: if mocap stops, position and
 * velocity freeze; the host watches mocapEst.ageMs and cuts the motors.
 *
 * Params (mocapEst.*): velAlpha, velMinDtMs, yawGain.  Logs (mocapEst.*): ageMs, poses, yawOffset.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "estimator.h"
#include "log.h"
#include "param.h"
#include "sensfusion6.h"
#include "stabilizer_types.h"
#include "usec_time.h"

#define ATTITUDE_DT (1.0f / ATTITUDE_RATE)
#define MAX_POSE_GAP_US 100000   // longer gaps restart the velocity estimate instead of averaging
#define DEG_PER_RAD 57.29578f

// --- tunables ---
static float velAlpha = 0.3f;     // velocity low-pass: weight of each new finite difference
static uint16_t velMinDtMs = 8;   // difference against a sample at least this old (radio bunching)
static float yawGain = 0.05f;     // per mocap sample: how fast the yaw offset follows mocap

// --- inputs ---
static Axis3f gyro;
static Axis3f acc;

// --- mocap-derived state ---
static point_t position;
static velocity_t velocity;
static bool havePose;
static point_t diffPos;           // reference sample for the finite difference
static uint64_t diffUs;
static uint64_t lastPoseUs;
static uint32_t poses;

// --- yaw alignment ---
static float yawOffsetDeg;        // added to the IMU yaw so it matches mocap yaw
static bool yawAligned;
static float imuYawDeg;           // latest complementary-filter yaw

static uint16_t ageMs;            // log: time since the last mocap packet

static float wrap180(float deg)
{
  while (deg > 180.0f) { deg -= 360.0f; }
  while (deg < -180.0f) { deg += 360.0f; }
  return deg;
}

static void onPosition(const float pos[3], uint64_t now)
{
  if (havePose && now - lastPoseUs > MAX_POSE_GAP_US) {
    havePose = false;                           // dropout: start the difference over
    velocity = (velocity_t){0};
  }
  if (!havePose) {
    diffPos = (point_t){.x = pos[0], .y = pos[1], .z = pos[2]};
    diffUs = now;
    havePose = true;
  } else if (now - diffUs >= (uint64_t)velMinDtMs * 1000u) {
    const float dt = (float)(now - diffUs) * 1e-6f;
    velocity.x += velAlpha * ((pos[0] - diffPos.x) / dt - velocity.x);
    velocity.y += velAlpha * ((pos[1] - diffPos.y) / dt - velocity.y);
    velocity.z += velAlpha * ((pos[2] - diffPos.z) / dt - velocity.z);
    diffPos = (point_t){.x = pos[0], .y = pos[1], .z = pos[2]};
    diffUs = now;
  }
  position = (point_t){.x = pos[0], .y = pos[1], .z = pos[2]};
  lastPoseUs = now;
  poses++;
}

static void onPose(const poseMeasurement_t *pose, uint64_t now)
{
  onPosition(pose->pos, now);

  const quaternion_t *q = &pose->quat;
  const float mocapYawDeg = DEG_PER_RAD * atan2f(2.0f * (q->w * q->z + q->x * q->y),
                                                 1.0f - 2.0f * (q->y * q->y + q->z * q->z));
  const float err = wrap180(mocapYawDeg - (imuYawDeg + yawOffsetDeg));
  yawOffsetDeg = wrap180(yawOffsetDeg + (yawAligned ? yawGain * err : err));
  yawAligned = true;
}

void estimatorOutOfTreeInit(void)
{
  sensfusion6Init();
  position = (point_t){0};
  velocity = (velocity_t){0};
  havePose = false;
  yawAligned = false;
  yawOffsetDeg = 0.0f;
  poses = 0;
}

bool estimatorOutOfTreeTest(void)
{
  return sensfusion6Test();
}

void estimatorOutOfTree(state_t *state, const stabilizerStep_t stabilizerStep)
{
  const uint64_t now = usecTimestamp();
  measurement_t m;
  while (estimatorDequeue(&m)) {
    switch (m.type) {
      case MeasurementTypeGyroscope:
        gyro = m.data.gyroscope.gyro;
        break;
      case MeasurementTypeAcceleration:
        acc = m.data.acceleration.acc;
        break;
      case MeasurementTypePose:
        onPose(&m.data.pose, now);
        break;
      case MeasurementTypePosition:
        onPosition(m.data.position.pos, now);
        break;
      default:
        break;
    }
  }

  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    return;
  }

  sensfusion6UpdateQ(gyro.x, gyro.y, gyro.z, acc.x, acc.y, acc.z, ATTITUDE_DT);

  // Attitude in the legacy CF2 convention (pitch inverted), as the controllers expect.
  float roll, pitch, yaw;
  sensfusion6GetEulerRPY(&roll, &pitch, &yaw);
  imuYawDeg = yaw;
  state->attitude.roll = roll;
  state->attitude.pitch = pitch;
  state->attitude.yaw = wrap180(yaw + yawOffsetDeg);

  // Quaternion (not legacy-adjusted), rotated by the same yaw offset about world z.
  float qx, qy, qz, qw;
  sensfusion6GetQuaternion(&qx, &qy, &qz, &qw);
  const float half = 0.5f * yawOffsetDeg / DEG_PER_RAD;
  const float cz = cosf(half), sz = sinf(half);
  state->attitudeQuaternion.w = cz * qw - sz * qz;
  state->attitudeQuaternion.x = cz * qx - sz * qy;
  state->attitudeQuaternion.y = cz * qy + sz * qx;
  state->attitudeQuaternion.z = cz * qz + sz * qw;

  state->acc.z = sensfusion6GetAccZWithoutGravity(acc.x, acc.y, acc.z);
  state->position = position;
  state->velocity = velocity;

  const uint64_t age = havePose ? (now - lastPoseUs) / 1000u : UINT16_MAX;
  ageMs = age > UINT16_MAX ? UINT16_MAX : (uint16_t)age;
}

PARAM_GROUP_START(mocapEst)
PARAM_ADD(PARAM_FLOAT, velAlpha, &velAlpha)
PARAM_ADD(PARAM_UINT16, velMinDtMs, &velMinDtMs)
PARAM_ADD(PARAM_FLOAT, yawGain, &yawGain)
PARAM_GROUP_STOP(mocapEst)

LOG_GROUP_START(mocapEst)
LOG_ADD(LOG_UINT16, ageMs, &ageMs)
LOG_ADD(LOG_UINT32, poses, &poses)
LOG_ADD(LOG_FLOAT, yawOffset, &yawOffsetDeg)
LOG_GROUP_STOP(mocapEst)
