#pragma once

#include <CRRT.hpp>
#include "CudaBrickTree.hpp"
#include "Camera.hpp"
#include "CudaLights.hpp"

/*
    Accessors for Cuda kernel launching in host code
*/
void Draw(CRRT::FrameView frame, const CudaBrickTree& tree, const Camera& camera, const CudaLights& lights);
