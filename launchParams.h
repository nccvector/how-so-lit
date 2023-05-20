#pragma once

#include <stdint.h>
#include <vector_types.h>

namespace optix7tutorial {
  struct LaunchParams
  {
    uint32_t *colorBuffer;
    int2      fbSize;
  };

}
