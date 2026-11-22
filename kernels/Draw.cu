#include "Draw.hpp"
#include "RayMarch.hpp"

#include <stdexcept>
#include <string>

/*
    Draw to bricktree to window (launchable from host)
*/
void Draw(CRRT::FrameView frame, const CudaBrickTree& tree, const Camera& camera, const CudaLights& lights)
{
    if (frame.width <= 0 || frame.height <= 0) return;

    dim3 blockDim(16, 16);
    dim3 gridDim((frame.width + 15) / 16, (frame.height + 15) / 16);
    RayMarch<<<gridDim, blockDim>>>(frame, camera, tree, lights);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("RayMarch launch failed: ") + cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("RayMarch execution failed: ") + cudaGetErrorString(error));
    }
}
