#pragma once

#include "helper_math.h"
#include <cmath>
#include <cuda_runtime.h>
#include <vector_functions.h>

struct Quaternion 
{
    // [w, x, y, z]
    float data[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    // convert angle axis rotation to quaternions
    __host__ __device__ void setRotation(float3 axis, float theta) {
        axis = normalize(axis);
        data[0] = cosf(theta/2);
        data[1] = axis.x*sinf(theta/2);
        data[2] = axis.y*sinf(theta/2);
        data[3] = axis.z*sinf(theta/2);
    }

    // quaternion quaterion multiplication (hamilton product)
    __host__ __device__ Quaternion multiply(const Quaternion& other) const {
        const float w1 = data[0], x1 = data[1], y1 = data[2], z1 = data[3];
        const float w2 = other.data[0], x2 = other.data[1], y2 = other.data[2], z2 = other.data[3];
        return Quaternion{{
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        }};
    }

    // normalize quaternion
    __host__ __device__ Quaternion normalized() const {
        float lengthSquared = data[0]*data[0] + data[1]*data[1] + data[2]*data[2] + data[3]*data[3];
        if (lengthSquared == 0.0f) return Quaternion{};
        float inverseLength = 1.0f / sqrtf(lengthSquared);
        return Quaternion{{
            data[0]*inverseLength, data[1]*inverseLength, data[2]*inverseLength, data[3]*inverseLength
        }};
    }

    // Rotation by a unit quaternion.
    __host__ __device__ float3 rotateVector(float3 vector) const {
        float3 r = make_float3(data[1], data[2], data[3]);
        float3 t = 2.0f * cross(r, vector);
        return vector + data[0] * t + cross(r, t);
    };
};
