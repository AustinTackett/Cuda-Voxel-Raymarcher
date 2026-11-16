#include "Draw.hpp"
#include "RayMarch.hpp"

#include <stdexcept>
#include <string>

void Draw(CRRT::FrameView frame, const Camera& camera, BrickTreeNode<uint8_t>* nodes, Brick<uint8_t>* bricks, uchar4* palette)
{
    if (frame.width <= 0 || frame.height <= 0) return;

    dim3 blockDim(16, 16);
    dim3 gridDim((frame.width + 15) / 16, (frame.height + 15) / 16);
    RayMarch<<<gridDim, blockDim>>>(frame, camera, nodes, bricks, palette);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("RayMarch launch failed: ") + cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("RayMarch execution failed: ") + cudaGetErrorString(error));
    }
}
