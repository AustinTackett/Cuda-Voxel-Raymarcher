#pragma once

#include <cstdint>
#include "Lights.hpp"

/*
    Cuda Device Container for array of various light types 
*/

inline constexpr uint32_t MAX_DIRECTIONAL_LIGHTS = 5;

struct CudaLights
{
    DirectionalLight* directionalLights = nullptr;
    uint32_t directionalLightCount = 0;

    // ... other light types in the futures
};

inline bool ReleaseLights(CudaLights& lights)
{
    bool success = cudaFree(lights.directionalLights) == cudaSuccess;
    lights = {};
    return success;
}

inline bool UploadLights(Lights lights, CudaLights& deviceLights)
{
    if (deviceLights.directionalLights)
        return false;
    if (lights.directionalLights.size() > MAX_DIRECTIONAL_LIGHTS)
        return false;

    deviceLights = {};
    if (lights.directionalLights.empty())
        return true;

    size_t bytes = lights.directionalLights.size() * sizeof(DirectionalLight);
    cudaError_t error = cudaMalloc(&deviceLights.directionalLights, bytes);
    if (error == cudaSuccess)
        error = cudaMemcpy(deviceLights.directionalLights, lights.directionalLights.data(),
                           bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        ReleaseLights(deviceLights);
        return false;
    }

    deviceLights.directionalLightCount = static_cast<uint32_t>(lights.directionalLights.size());
    return true;
}

