// Firmware-independent math for the mocap estimator and velocity controller (also built on the host for tests).
#pragma once

#include <stdbool.h>

#define VELCTL_G 9.81f

typedef struct {
  float p[3];
  float v[3];
  bool init;
} abFilter_t;

void abReset(abFilter_t *f, const float z[3]);

// One alpha-beta step: predict over dt, correct with position measurement z.
void abUpdate(abFilter_t *f, const float z[3], float dt, float alpha, float beta);

typedef struct {
  float lag;   // [s] last arrival minus its frame slot (radio queueing delay estimate)
} frameClock_t;

// Frames since the previous packet: slot-clock guess (radio delay <= maxDelay periods); once one frame of motion
// exceeds minStep [m], the count (+-1) that best predicts z wins.
int abFramesElapsed(const abFilter_t *f, frameClock_t *c, const float z[3], float gapS, float periodS,
                    float maxDelay, float clockGain, float minStep);

typedef struct {
  float kv[3];       // [1/s]   velocity error -> acceleration
  float ki[3];       // [1/s^2] velocity error integral -> acceleration
  float iLimit[3];   // [m/s^2] integral clamp
  float maxTiltRad;
  float maxAccUp;    // [m/s^2] above gravity
  float maxAccDown;  // [m/s^2] below gravity (positive number)
} velCtlGains_t;

typedef struct {
  float integ[3];
} velCtlState_t;

void velCtlReset(velCtlState_t *s);

// Velocity loop: world velocity reference -> world acceleration command (gravity excluded), clamped.
void velCtlAccel(velCtlState_t *s, const velCtlGains_t *g, const float vRef[3], const float v[3],
                 float dt, bool integrateXY, bool integrateZ, float aOut[3]);

// World acceleration command (gravity excluded) -> desired body z axis with tilt and vertical limits applied.
void velCtlAccelToZb(const velCtlGains_t *g, const float a[3], float zbDes[3], float *liftOut);

// Desired body z axis + yaw -> right-handed ZYX roll/pitch [rad] (pitch > 0 = nose down).
void velCtlZbToRollPitch(const float zbDes[3], float yawRad, float *roll, float *pitch);

// Collective thrust [N] projected on the current body z axis.
float velCtlThrustN(float mass, const float zbDes[3], float lift, const float zbNow[3]);
