#pragma once

#include <vector>

#include "DirectionalLight.hpp"

struct Lights
{
    std::vector<DirectionalLight> directionalLights;
    // ... other light types in the futures
};