#pragma once

#include <algorithm>
#include <CRRT.hpp>
#include "Quaternion.hpp"

/*
    Camera to move the scene (controls same as Minecraft)
*/

inline constexpr float Deg2Rad(float degrees) {
    return degrees * (3.14159265358979323846f / 180.0f);
}

struct Camera
{
    inline static const float3 WorldX = {1, 0, 0};
    inline static const float3 WorldY = {0, 1, 0};
    inline static const float3 WorldZ = {0, 0, 1};

    float3 position = {5.5f, 0.0f, -7.0f};
    Quaternion orientation;
    float yaw   = Deg2Rad(-230.0f);
    float pitch = 0.0f;
    float fov   = Deg2Rad(45.0f);
    float speed = 3.0f;
    bool altPressedLastFrame = false;
    bool isCursorHidden      = true;

    Camera() { 
        updateOrientation(); 
        CRRT::CursorVisibility(!isCursorHidden);
    }

    void updateOrientation() {
        Quaternion yawRotation, pitchRotation, zUpRotation;
        yawRotation.setRotation(WorldZ, yaw);
        pitchRotation.setRotation(WorldX, pitch);
        zUpRotation.setRotation(WorldX, Deg2Rad(90.0f));
        orientation = yawRotation.multiply(pitchRotation).multiply(zUpRotation).normalized();
    }

    void update() {
        using Input = CRRT::Input;

        const float distance = speed * static_cast<float>(CRRT::Time::DeltaTime);
        const float3 rightDir = orientation.rotateVector(make_float3(1, 0, 0));
        float3 forwardDir = orientation.rotateVector(make_float3(0, 0, -1));
        forwardDir.z = 0.0f;

        if (dot(forwardDir, forwardDir) > 1e-12f) forwardDir = normalize(forwardDir);
        else forwardDir = make_float3(0.0f);

        if (Input::KEY_W || Input::KEY_UP)        position += forwardDir * distance;
        else if (Input::KEY_S || Input::KEY_DOWN) position -= forwardDir * distance;
        
        if (Input::KEY_D || Input::KEY_RIGHT)     position += rightDir * distance;
        else if (Input::KEY_A || Input::KEY_LEFT) position -= rightDir * distance;
        
        if (Input::KEY_SPACE) position += WorldZ * distance;
        if (Input::KEY_SHIFT) position -= WorldZ * distance;

        const bool toggleCursor = Input::KEY_ALT && !altPressedLastFrame;
        if (toggleCursor) {
            isCursorHidden = !isCursorHidden;
            CRRT::CursorVisibility(!isCursorHidden);
        }
        altPressedLastFrame = Input::KEY_ALT;

        // Ignore the old mouse delta on the frame that changes cursor capture.
        if (isCursorHidden && !toggleCursor) {
            auto [width, height] = CRRT::GetSize();
            yaw -= static_cast<float>(Input::MOUSE_X_DELTA / std::max(width, 1));
            pitch -= static_cast<float>(Input::MOUSE_Y_DELTA / std::max(height, 1));
            pitch = std::clamp(pitch, -Deg2Rad(89.0f), Deg2Rad(89.0f));
            updateOrientation();
        }
    }
};
