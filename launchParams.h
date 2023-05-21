#pragma once

#include <stdint.h>
#include <vector_types.h>

struct LaunchParams {
  uint32_t* colorBuffer;
  int2      fbSize;
};
