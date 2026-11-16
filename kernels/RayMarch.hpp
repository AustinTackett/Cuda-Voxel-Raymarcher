#pragma once

#include <CRRT.hpp>
#include <cuda_runtime.h>
#include "TraceBrickTree.hpp"
#include "Camera.hpp"

__global__ void RayMarch(CRRT::FrameView frame, Camera camera, BrickTreeNode<uint8_t>* nodes, Brick<uint8_t> *bricks, uchar4* pallete); 