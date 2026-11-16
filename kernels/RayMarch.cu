#include "RayMarch.hpp"

__device__ constexpr uchar4 backgroundColor = {0, 0, 0, 0};

__device__ Ray GetCameraRay(CRRT::FrameView frame, Camera camera, uint x_pixel, uint y_pixel)
{
    float aspect = static_cast<float>(frame.width) / frame.height;

    float x_clip = 2 * ((x_pixel + 0.5) / frame.width)  - 1; // [-1, 1]
    float y_clip = 2 * ((y_pixel + 0.5) / frame.height) - 1; // [-1, 1]
    
    float3 rayDir = make_float3(
        x_clip * aspect * tanf(camera.fov / 2),
        y_clip * tanf(camera.fov / 2),
        -1.0f
    );
    
    rayDir = normalize(camera.orientation.rotateVector(rayDir));

    return Ray {camera.position, rayDir, 0.0f};
}

__global__ void RayMarch(CRRT::FrameView frame, Camera camera, BrickTreeNode<uint8_t>* nodes, Brick<uint8_t> *bricks, uchar4* pallete) 
{
    uint x_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_pixel = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_pixel >= frame.width || y_pixel >= frame.height) return;

    Ray ray = GetCameraRay(frame,camera, x_pixel, y_pixel);
    VoxelHit voxelHit = TraceVoxel(ray, nodes, bricks);
    
    uchar4 pixelColor = voxelHit.isHit ? pallete[voxelHit.pallete] : backgroundColor;
    CRRT::WriteFrameRaw(frame, x_pixel, y_pixel, pixelColor);
};
