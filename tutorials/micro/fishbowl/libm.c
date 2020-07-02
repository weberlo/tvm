#include <libm.h>

float expf(float x)
{
  int n = 10;
  float sum = 1.0f; // initialize sum of series

  for (int i = n - 1; i > 0; --i )
    sum = 1 + x * sum / i;

  return sum;
}
