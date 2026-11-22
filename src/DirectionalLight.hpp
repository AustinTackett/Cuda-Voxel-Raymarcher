#pragma once

#include <cuda_runtime.h>

struct DirectionalLight {
    float3 direction;
    float3 Radiance; // Per RGB channel
};
