#include <cstdlib>
#include <iostream>

#include <CRRT.hpp>

#define OGT_VOX_IMPLEMENTATION
#include "ogt_vox.h"
#undef OGT_VOX_IMPLEMENTATION

#include "LoadVoxels.hpp"
#include "Camera.hpp"
#include "Draw.hpp"
#include "Lights.hpp"

// Place camera at some reasonable orientation to view the model
void initCamera(Camera& camera, const BrickTree& brickMap)
{
    const float3 modelMin = make_float3(
        float(brickMap.offset_x), 
        float(brickMap.offset_y), 
        float(brickMap.offset_z));
    const float3 modelMax = make_float3(
        float(brickMap.grid_size_x - BACKGROUND_PADDING),
        float(brickMap.grid_size_y - BACKGROUND_PADDING),
        float(brickMap.grid_size_z - BACKGROUND_PADDING));
    
    const float3 center = (modelMin + modelMax) * 0.5f;
    camera.position     = make_float3(center.x, modelMin.y, center.z);
    
    camera.yaw   = 0.0f;
    camera.pitch = 0.0f;
    camera.speed = fmaxf(length(modelMax - modelMin) * 0.25f, 3.0f);
    camera.updateOrientation();
}

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

    // Load CPU data
    BrickTree brickMap;
    std::string loadError;
    if (!LoadVoxels(argv[1], brickMap, loadError)) {
        std::cerr << loadError << '\n';
        errors |= CRRT::ReleaseWindow();
        if (errors) std::cerr << GetErrorsString(errors);
        return EXIT_FAILURE;
    }
    Lights lights;
    lights.directionalLights.push_back({
        normalize(make_float3(1.0f, 1.0f, -1.0f)), 
        make_float3(1.5f)
    });

    // Initialize Scene/Camera Parameters
    Camera camera;
    initCamera(camera, brickMap);

    // GPU resources
    CudaBrickTree deviceTree;
    CudaLights deviceLights;
    auto releaseCuda = [&]() {
        bool success = ReleaseBrickTree(deviceTree);
        if (!success) std::cerr << "Brick tree release failed.\n";
        bool lightsReleased = ReleaseLights(deviceLights);
        if (!lightsReleased) std::cerr << "Lights release failed.\n";
        return success && lightsReleased;
    };

    // Upload CPU data to GPU
    if (!UploadLights(lights, deviceLights)) {
        std::cerr << "Lights upload failed.\n";
        releaseCuda();
        errors |= CRRT::ReleaseWindow();
        if (errors) std::cerr << GetErrorsString(errors);
        return EXIT_FAILURE;
    }
    if (!UploadBrickTree(brickMap, deviceTree)) {
        std::cerr << "Brick tree upload failed.\n";
        errors |= CRRT::ReleaseWindow();
        if (errors) std::cerr << GetErrorsString(errors);
        return EXIT_FAILURE;
    }

    // Draw Call back
    auto onWindowUpdate = [&](CRRT::FrameView frame) {
        camera.update();
        Draw(frame, deviceTree, camera, deviceLights);
    };

    // Main Window Update Loop
    while (!CRRT::ShouldClose()) {  

        errors |= CRRT::UpdateWindow(onWindowUpdate);
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

    bool released = releaseCuda();
    errors |= CRRT::ReleaseWindow();
    std::cout << GetErrorsString(errors);
    return errors || !released ? EXIT_FAILURE : EXIT_SUCCESS;
}

