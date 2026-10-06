// Out-of-tree estimator: position/velocity from an alpha-beta filter on mocap, roll/pitch from the IMU, yaw IMU + mocap.

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "estimator.h"
#include "log.h"
#include "param.h"
#include "sensfusion6.h"
#include "stabilizer_types.h"
#include "usec_time.h"

#include "velctl_math.h"

#define ATTITUDE_DT (1.0f / ATTITUDE_RATE)
#define MAX_POSE_GAP_US 100000   // longer gaps restart the filter
#define DEG_PER_RAD 57.29578f

// --- tunables ---
static float rateHz = 240.0f;     // nominal mocap frame rate the host streams at
static float alpha = 0.54f;       // alpha-beta position gain (Kalata lambda 0.3; test/validate_velctl.py)
static float beta = 0.20f;        // alpha-beta velocity gain
static float maxDelay = 0.7f;     // [periods] radio delay tolerated before a gap counts as a dropped frame
static float clockGain = 0.01f;   // per frame: how fast the slot clock follows later arrivals (clock drift)
static float minStep = 0.0015f;   // [m] per-frame motion above which the position decides ambiguous frame counts
static float yawGain = 0.05f;     // per mocap sample: how fast the yaw offset follows mocap

// --- inputs ---
static Axis3f gyro;
static Axis3f acc;

// --- mocap-derived state ---
static abFilter_t ab;
static frameClock_t frameClock;
static uint64_t lastPoseUs;
static uint32_t poses;
static uint32_t drops;

// --- yaw alignment ---
static float yawOffsetDeg;
static bool yawAligned;
static float imuYawDeg;

static uint16_t ageMs;

static float wrap180(float deg)
{
  while (deg > 180.0f) { deg -= 360.0f; }
  while (deg < -180.0f) { deg += 360.0f; }
  return deg;
}

static void onPosition(const float pos[3], uint64_t now)
{
  const float period = 1.0f / rateHz;
  if (!ab.init || now - lastPoseUs > MAX_POSE_GAP_US) {
    abReset(&ab, pos);
    frameClock.lag = 0.0f;
  } else {
    const int n = abFramesElapsed(&ab, &frameClock, pos, (float)(now - lastPoseUs) * 1e-6f, period, maxDelay, clockGain,
                                  minStep);
    drops += (uint32_t)(n - 1);
    abUpdate(&ab, pos, (float)n * period, alpha, beta);
  }
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
  ab.init = false;
  yawAligned = false;
  yawOffsetDeg = 0.0f;
  poses = 0;
  drops = 0;
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
  state->position = (point_t){.x = ab.p[0], .y = ab.p[1], .z = ab.p[2]};
  state->velocity = (velocity_t){.x = ab.v[0], .y = ab.v[1], .z = ab.v[2]};

  const uint64_t age = ab.init ? (now - lastPoseUs) / 1000u : UINT16_MAX;
  ageMs = age > UINT16_MAX ? UINT16_MAX : (uint16_t)age;
}

PARAM_GROUP_START(mocapEst)
PARAM_ADD(PARAM_FLOAT, rateHz, &rateHz)
PARAM_ADD(PARAM_FLOAT, alpha, &alpha)
PARAM_ADD(PARAM_FLOAT, beta, &beta)
PARAM_ADD(PARAM_FLOAT, maxDelay, &maxDelay)
PARAM_ADD(PARAM_FLOAT, clockGain, &clockGain)
PARAM_ADD(PARAM_FLOAT, minStep, &minStep)
PARAM_ADD(PARAM_FLOAT, yawGain, &yawGain)
PARAM_GROUP_STOP(mocapEst)

LOG_GROUP_START(mocapEst)
LOG_ADD(LOG_UINT16, ageMs, &ageMs)
LOG_ADD(LOG_UINT32, poses, &poses)
LOG_ADD(LOG_UINT32, drops, &drops)
LOG_ADD(LOG_FLOAT, yawOffset, &yawOffsetDeg)
LOG_GROUP_STOP(mocapEst)
