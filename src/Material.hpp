#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <helper_math.h>
#include "ogt_vox.h"

inline constexpr uint8_t BACKGROUND_MAT_INDEX = 0;

/*
    Material definition
*/
struct Material
{
    bool   surface      = true;
    float3 albedo       = {0.0f, 0.0f, 0.0f};
    float  roughness    = 0.5f;
    float  ior          = 1.5f;
    float  transmission = 0.0f;
    float3 sigma_a      = {0.0f, 0.0f, 0.0f}; // absorbtion
    //     sigma_s      // scattering
};

/*
    Parsing .vox file's internal material representation to a simpler one
*/
inline float srgbToLinear(float value)
{
    return value <= 0.04045f 
        ? value / 12.92f 
        : powf((value + 0.055f) / 1.055f, 2.4f);
}

inline float3 getAlbedo(uchar4 color)
{
    return make_float3(
        srgbToLinear(color.x / 255.0f),
        srgbToLinear(color.y / 255.0f),
        srgbToLinear(color.z / 255.0f));
}

inline float getRoughness(const ogt_vox_matl& material)
{
    return (material.content_flags & k_ogt_vox_matl_have_rough) != 0
        ? clamp(material.rough, 0.0f, 1.0f)
        : (material.type == ogt_matl_type_glass ? 0.0f : 0.5f);
}

inline float getIor(const ogt_vox_matl& material)
{
    return (material.content_flags & k_ogt_vox_matl_have_ior) != 0
        ? fmaxf(1.0f + material.ior, 1.0f)  // vox mat stores offset above 1.0
        : 1.5f;
}

inline float getTransmission(const ogt_vox_matl& material)
{
    float transmission = 0.0f;
    if (material.type == ogt_matl_type_glass ||
        material.type == ogt_matl_type_blend)
        transmission = (material.content_flags & k_ogt_vox_matl_have_trans) != 0
            ? clamp(material.trans, 0.0f, 1.0f)
            : (material.type == ogt_matl_type_glass ? 1.0f : 0.0f);

    return transmission;
}

inline float3 getSigma_a(const ogt_vox_matl& material, float3 albedo)
{
    if (getTransmission(material) == 0.0f)
        return make_float3(0.0f);

    const float sigma_absorbtion = (material.content_flags & k_ogt_vox_matl_have_att)
        ? fmaxf(material.att, 0.0f)
        : 0.1f;
    const float minTransmission = 1.0e-6f;
    return sigma_absorbtion * make_float3(
        -logf(clamp(albedo.x, minTransmission, 1.0f)),
        -logf(clamp(albedo.y, minTransmission, 1.0f)),
        -logf(clamp(albedo.z, minTransmission, 1.0f)));
}

inline Material ParseMaterial(const ogt_vox_matl& source, uchar4 color)
{
    Material material;
    material.albedo       = getAlbedo(color);
    material.roughness    = getRoughness(source);
    material.ior          = getIor(source);
    material.transmission = getTransmission(source);
    material.sigma_a      = getSigma_a(source, material.albedo);
    material.surface      = source.type != ogt_matl_type_media;
    return material;
}
