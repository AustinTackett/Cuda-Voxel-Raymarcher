#pragma once

#include <cuda_runtime.h>

/*
    Axis aligned bounding box
*/
struct Aabb
{
    float3 min;
    float3 max;
};
