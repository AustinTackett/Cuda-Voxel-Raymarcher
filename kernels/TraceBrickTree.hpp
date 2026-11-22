# pragma once

#include <cuda_runtime.h>
#include "CudaBrickTree.hpp"

/*
    Ray 
*/
struct Ray 
{
    float3 o;
    float3 d;    
};

struct AabbHit
{
    bool   isHit;
    float  tMin;
    float  tMax;
};

struct BrickTreeHit
{
    bool    isHit;
    uint8_t material;
    float3  position;
    float3  normal;
    float   t;
};

// Default tracing ignores the background material.
__device__ BrickTreeHit TraceBrickTree(const CudaBrickTree& tree, Ray ray);

__device__ BrickTreeHit TraceBrickTree(const CudaBrickTree& tree, Ray ray, uint ignoreMaterial);
