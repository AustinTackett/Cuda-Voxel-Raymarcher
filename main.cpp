#include <cstddef>
#include <cstdlib>
#include <iostream>

#include <CRRT.hpp>

#define OGT_VOX_IMPLEMENTATION
#include "ogt_vox.h"
#undef OGT_VOX_IMPLEMENTATION

#include "LoadVoxels.hpp"
#include "Camera.hpp"
#include "Draw.hpp"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <voxel-asset.vox>\n";
        return EXIT_FAILURE;
    }

    // Window Configuration
    CRRT::WindowConfig windowConfig;
    windowConfig.name       = "Ray Marching Window";
    windowConfig.initWidth  = windowConfig.resWidth  = 1920;
    windowConfig.initHeight = windowConfig.resHeight = 1080;

    // Create GLFW window
    CRRT::Errors errors = CRRT::CreateWindow(windowConfig);
    if (errors) {
        std::cerr << GetErrorsString(errors);
        return EXIT_FAILURE;
    }

    // Load Model to CPU
    BrickTree<uint8_t> brickMap;
    std::string loadError;
    if (!LoadVoxels(argv[1], brickMap, loadError)) {
        std::cerr << loadError << '\n';
        errors |= CRRT::ReleaseWindow();
        if (errors) std::cerr << GetErrorsString(errors);
        return EXIT_FAILURE;
    }

    // Load Model to GPU
    BrickTreeNode<uint8_t>* d_nodes = nullptr;
    Brick<uint8_t>* d_bricks = nullptr;
    uchar4* d_palette = nullptr;
    const size_t nodesBytes   = brickMap.nodes.size() * sizeof(BrickTreeNode<uint8_t>);
    const size_t bricksBytes  = brickMap.bricks.size() * sizeof(Brick<uint8_t>);
    const size_t paletteBytes = sizeof(brickMap.palette);
    cudaMalloc(&d_nodes, nodesBytes);
    cudaMemcpy(d_nodes, brickMap.nodes.data(), nodesBytes, cudaMemcpyHostToDevice);
    if (bricksBytes > 0) {
        cudaMalloc(&d_bricks, bricksBytes);
        cudaMemcpy(d_bricks, brickMap.bricks.data(), bricksBytes, cudaMemcpyHostToDevice);  
    }
    cudaMalloc(&d_palette, paletteBytes);
    cudaMemcpy(d_palette, brickMap.palette, paletteBytes, cudaMemcpyHostToDevice);

    auto releaseCuda = [&]() -> void {
        cudaFree(d_palette);
        cudaFree(d_bricks);
        cudaFree(d_nodes);
    };

    // Initialize Scene/Camera Parameters
    Camera camera;
    const float3 modelMin = make_float3(
        float(brickMap.offset_x), 
        float(brickMap.offset_y), 
        float(brickMap.offset_z));
    const float3 modelMax = make_float3(
        float(brickMap.grid_size_x), 
        float(brickMap.grid_size_y), 
        float(brickMap.grid_size_z));
    
    const float3 center = (modelMin + modelMax) * 0.5f;
    const float radius = length(modelMax - modelMin) * 0.5f;
    const float aspect = float(windowConfig.resWidth) / windowConfig.resHeight;
    const float halfFov = atanf(tanf(camera.fov * 0.5f) * fminf(aspect, 1.0f));
    const float distance = radius / sinf(halfFov);
    camera.position = center + normalize(make_float3(0.0f, -1.0f, 0.5f)) * distance;
    
    const float3 forward = normalize(center - camera.position);
    camera.yaw = atan2f(-forward.x, forward.y);
    camera.pitch = asinf(forward.z);
    camera.speed = fmaxf(radius * 0.5f, 3.0f);
    camera.updateOrientation();

    // Draw Call back
    auto draw = [&](CRRT::FrameView frame) {
        camera.update();
        Draw(frame, camera, d_nodes, d_bricks, d_palette);
    };

    // Main Draw Loop
    while (!CRRT::ShouldClose()) {  

        errors |= CRRT::UpdateWindow(draw);
        if (errors) {
            releaseCuda();
            errors |= CRRT::ReleaseWindow();
            std::cerr << GetErrorsString(errors);
            return EXIT_FAILURE;
        }

        if (CRRT::Input::KEY_ESCAPE) {
            CRRT::SetShouldClose();
            break;
        }

        CRRT::SwapBuffer(); 
    }

    releaseCuda();
    errors |= CRRT::ReleaseWindow();
    std::cout << GetErrorsString(errors);
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}

