#pragma once

#include <CRRT.hpp>

#include <cuda_runtime.h>

#include "CudaBrickTree.hpp"
#include "Camera.hpp"
#include "CudaLights.hpp"

__global__ void RayMarch(CRRT::FrameView frame, Camera camera, CudaBrickTree tree, CudaLights lights);
