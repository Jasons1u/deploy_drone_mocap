#include "velctl_math.h"

#include <math.h>

#define AMBIGUOUS_FRAC 0.3f   // arrivals this far (in periods) past a slot could be a late packet or a dropped frame

static float clampf(float v, float lo, float hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

void abReset(abFilter_t *f, const float z[3])
{
  for (int i = 0; i < 3; i++) {
    f->p[i] = z[i];
    f->v[i] = 0.0f;
  }
  f->init = true;
}

void abUpdate(abFilter_t *f, const float z[3], float dt, float alpha, float beta)
{
  for (int i = 0; i < 3; i++) {
    const float pred = f->p[i] + f->v[i] * dt;
    const float r = z[i] - pred;
    f->p[i] = pred + alpha * r;
    f->v[i] += (beta / dt) * r;
  }
}

static float residual2(const abFilter_t *f, const float z[3], float dt)
{
  float r2 = 0.0f;
  for (int i = 0; i < 3; i++) {
    const float r = z[i] - (f->p[i] + f->v[i] * dt);
    r2 += r * r;
  }
  return r2;
}

int abFramesElapsed(const abFilter_t *f, frameClock_t *c, const float z[3], float gapS, float periodS,
                    float maxDelay, float clockGain, float minStep)
{
  const float x = (c->lag + gapS) / periodS;
  const int guess = (int)floorf(x - maxDelay) + 1;
  int n = guess < 1 ? 1 : guess;
  const float step2 = (f->v[0] * f->v[0] + f->v[1] * f->v[1] + f->v[2] * f->v[2]) * periodS * periodS;
  if (step2 > minStep * minStep && x - floorf(x) >= AMBIGUOUS_FRAC) {
    const int base = n;
    float best = residual2(f, z, (float)base * periodS);
    for (int cand = base - 1; cand <= base + 1; cand += 2) {
      if (cand >= 1) {
        const float r2 = residual2(f, z, (float)cand * periodS);
        if (r2 < best) {
          best = r2;
          n = cand;
        }
      }
    }
  }
  const float lag = x - (float)n;
  c->lag = (lag < 0.0f ? 0.0f : lag * (1.0f - clockGain)) * periodS;
  return n;
}

void velCtlReset(velCtlState_t *s)
{
  for (int i = 0; i < 3; i++) {
    s->integ[i] = 0.0f;
  }
}

void velCtlAccel(velCtlState_t *s, const velCtlGains_t *g, const float vRef[3], const float v[3],
                 float dt, bool integrateXY, bool integrateZ, float aOut[3])
{
  for (int i = 0; i < 3; i++) {
    const float err = vRef[i] - v[i];
    const bool integrate = (i < 2) ? integrateXY : integrateZ;
    if (integrate) {
      s->integ[i] = clampf(s->integ[i] + g->ki[i] * err * dt, -g->iLimit[i], g->iLimit[i]);
    }
    aOut[i] = g->kv[i] * err + s->integ[i];
  }
}

void velCtlAccelToZb(const velCtlGains_t *g, const float a[3], float zbDes[3], float *liftOut)
{
  const float lift = clampf(a[2], -g->maxAccDown, g->maxAccUp) + VELCTL_G;
  float ax = a[0];
  float ay = a[1];
  const float maxH = lift * tanf(g->maxTiltRad);
  const float h = sqrtf(ax * ax + ay * ay);
  if (h > maxH && h > 0.0f) {
    ax *= maxH / h;
    ay *= maxH / h;
  }
  const float n = sqrtf(ax * ax + ay * ay + lift * lift);
  zbDes[0] = ax / n;
  zbDes[1] = ay / n;
  zbDes[2] = lift / n;
  *liftOut = n;
}

void velCtlZbToRollPitch(const float zbDes[3], float yawRad, float *roll, float *pitch)
{
  const float c = cosf(yawRad);
  const float s = sinf(yawRad);
  const float xh = c * zbDes[0] + s * zbDes[1];
  const float yh = -s * zbDes[0] + c * zbDes[1];
  *roll = asinf(clampf(-yh, -1.0f, 1.0f));
  *pitch = atan2f(xh, zbDes[2]);
}

float velCtlThrustN(float mass, const float zbDes[3], float lift, const float zbNow[3])
{
  const float proj = zbDes[0] * zbNow[0] + zbDes[1] * zbNow[1] + zbDes[2] * zbNow[2];
  return proj > 0.0f ? mass * lift * proj : 0.0f;
}
