# pragma once

#include <cuda_runtime.h>

#include "BrickTree.hpp" 

struct Ray 
{
    float3 o;
    float3 d;    
    float  t;
};

struct AabbHit
{
    bool   isHit;
    float  tMin;
    float  tMax;
};

struct VoxelHit
{
    bool    isHit;
    uint8_t pallete;
};

__device__ VoxelHit TraceVoxel(Ray& ray, BrickTreeNode<uint8_t>* nodes, Brick<uint8_t>* bricks);