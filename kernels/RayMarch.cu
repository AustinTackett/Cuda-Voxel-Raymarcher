#include "Crrt.hpp"
#include "Material.hpp"
#include "RayMarch.hpp"

#include "CudaLights.hpp"
#include "TraceBrickTree.hpp"
#include "helper_math.h"

#include <cmath>
#include <vector_functions.h>

/*
    Rendering Brick Tree Data using Ray Marching Loop
*/

/*
    Constants
*/
__device__ constexpr float  EPS_RAY          = 0.001f;
__device__ constexpr float  MIN_THROUGHPUT   = 0.001f;
__device__ constexpr float  AMBIENT_LIGHT    = 0.25f;
__device__ constexpr float3 BACKGROUND_COLOR = {0.55f, 0.55f, 0.55f};
__device__ constexpr float3 GROUND_COLOR     = {0.45f, 0.45f, 0.45f};
__device__ constexpr float  EPS              = 1e-6;
__device__ constexpr float  PI               = 3.141592653589f;
__device__ constexpr uint   MAX_BOUNCES      = 6u;
__device__ constexpr uint   MAX_RAY_STACK    = 8u;

/*
    Dithering
*/

/*
    Hash Without Sine
    Copyright (c) 2014 David Hoskins
    Licensed under the MIT License.

    Source: https://www.shadertoy.com/view/4djSRW
    Adapted from GLSL to CUDA.

    License:THIRD-PARTY-NOTICES/HashWithoutSine-LICENSE.md
*/
__device__ float noise(float x, float y)
{
    float3 p3 = make_float3(x, y, x);
    p3 *= make_float3(0.1031f, 0.1030f, 0.0973f);
    p3  = fracf(p3);
    float d = dot(p3, make_float3(p3.y, p3.z, p3.x) + 33.33f);
    p3 += make_float3(d);
    float h = (p3.x + p3.y) * p3.z;
    return fracf(h);
}

// Convert linear to srgb applying some dithering to avoid banding
__device__ float3 linearToDitherSrgb(float3 color, float x_pixel, float y_pixel)
{
    float n = (noise(x_pixel, y_pixel) - 0.5f) / 255.0f;

    auto toSrgb = [](float x) -> float {
        x = fminf(fmaxf(x, 0.0f), 1.0f);
        return x <= 0.0031308f
            ? 12.92f * x
            : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
    };

    return make_float3(
        toSrgb(color.x) + n,
        toSrgb(color.y) + n,
        toSrgb(color.z) + n
    );
}

/*
    Math Utils
*/
__device__ __forceinline__ float3 refract(float3 dirEntry, float3 normal, float iorEntry, float iorExit)
{
    float eta = iorEntry / iorExit;
    float cosTheta = -dot(dirEntry, normal);
    return eta * dirEntry + 
        (eta*cosTheta - sqrt(
            max(1 - pow(eta, 2) * (1 - pow(cosTheta, 2)), 0.0f)
        )) * normal;
}

/*
    Fresnel Equations
*/

// Schlicks approximation of fresnel for color "indepedent" microfacet dielectrics
__device__ float fresnel_dielectric_microfacet(float3 l, float3 v, float iorEntry, float iorExit)
{
    float  f0    = powf((iorEntry - iorExit) / (iorEntry + iorExit), 2);
    float3 h     = normalize(l + v);
    float  vDotH = clamp(dot(v, h), 0.0f, 1.0f);

    // check that sharp grazing angles produce only reflections
    float eta = iorEntry / iorExit;
    if (eta * eta * (1 - powf(vDotH, 2)) >= 1.0f)
        return 1.0f;

    return f0 + (1 - f0) * powf(1 - vDotH, 5);
}

// Fresnel for ideal non-microfacet reflections and refractions
__device__ float fresnel_ideal(float3 n, float3 v, float iorEntry, float iorExit)
{
    float f0    = powf((iorEntry - iorExit) / (iorEntry + iorExit), 2);
    float vDotN = clamp(fabsf(dot(v, n)), 0.0f, 1.0f);
    
    // check that sharp grazing angles produce only reflections
    float eta = iorEntry / iorExit;
    if (eta * eta * (1 - vDotN * vDotN) >= 1.0f)
        return 1.0f;

    if (eta > 1.0f)
        vDotN = sqrtf(fmaxf(0.0f, 1.0f - eta * eta * (1.0f - vDotN * vDotN)));

    return f0 + (1 - f0) * powf(1 - vDotN, 5);
}

/*
    Brdf's
*/

// Complete microfacet GGX specular brdf
__device__ float3 brdf_specular_microfacet(float3 n, float3 v, float3 l, float r, float3 F)
{
    // GGX Helper: Trowbridge-Reitz microfacet normal distribution
    auto normalDistribution = [&]() -> float {
        float3 h     = normalize(l + v);
        float  alpha = max(r * r, EPS);
        float  a2    = alpha * alpha;
        float  nDotH = max(dot(n, h), 0.0f);
        return a2 / (PI * powf(nDotH * nDotH * (a2 - 1.0f) + 1.0f, 2));
    };

    // GGX Helper: Smith microfacet geometry self shadow masking
    auto maskingShadow = [&](float3 x) -> float {
        float nDotX  = max(dot(n, x), 0.0f);
        float nDotX2 = powf(nDotX, 2);
        float alpha  = max(r * r, EPS);
        float a2     = alpha * alpha;
        return (2 * nDotX) / (nDotX + sqrt(a2 + (1-a2) * nDotX2));
    };

    float nDotL = dot(n, l);
    float nDotV = dot(n, v);

    if (nDotL <= 0.0f || nDotV <= 0.0f)
        return make_float3(0.0f);

    float  D = normalDistribution();
    float  G = maskingShadow(l) * maskingShadow(v);
    return F * (D * G / (4.0f * nDotL * nDotV));
}

// Oren-Nayar microfacet diffuse brdf
__device__ float3 brdf_diffuse_microfacet(float3 n, float3 v, float3 l, float3 albedo, float r)
{
    float nDotL = dot(n, l);
    float nDotV = dot(n, v);

    if (nDotL <= 0.0f || nDotV <= 0.0f)
        return make_float3(0.0f);

    float r2 = r * r;
    float A = 1.0f - 0.5f * r2 / (r2 + 0.33f);
    float B = 0.45f * r2 / (r2 + 0.09f);
    float s = max(0.0f, dot(l, v) - nDotL * nDotV);
    float t = max(nDotL, nDotV);
    return (albedo / PI) * (A + B * s / max(t, EPS));
}

/*
    Light Sampling: Computing Incoming Radiance
*/

__device__ bool IsLightVisible(const CudaBrickTree& tree, Ray ray)
{
    BrickTreeHit hit = TraceBrickTree(
        tree, ray, BACKGROUND_MAT_INDEX
    );

    return !hit.isHit;
}

// Precompute whether light is visible at specific position 
__device__ void traceVisibility(const CudaBrickTree& tree, const CudaLights& lights, bool (&visibilities)[MAX_DIRECTIONAL_LIGHTS], 
    float3 p, float3 n)
{
    for (uint i = 0; i < MAX_DIRECTIONAL_LIGHTS; ++i)
        visibilities[i] = false;
    
    for (uint i = 0; i < lights.directionalLightCount && i < MAX_DIRECTIONAL_LIGHTS; ++i) {
        float3 lightDir = normalize(-lights.directionalLights[i].direction);
        if (dot(n, lightDir) <= 0.0f) continue;
        Ray shadowRay = {p + lightDir * EPS_RAY, lightDir};
        visibilities[i] = IsLightVisible(tree, shadowRay);
    }
}

__device__ float3 Li_light(const CudaLights& lights, const bool (&visibilities)[MAX_DIRECTIONAL_LIGHTS], uint lightIndex)
{
    if (lightIndex >= lights.directionalLightCount 
        || lightIndex >= MAX_DIRECTIONAL_LIGHTS 
        || !visibilities[lightIndex])
    {
        return make_float3(0.0f);
    }
    
    return lights.directionalLights[lightIndex].Radiance;
}

/*
    Drawing from Camera
*/
struct RayState
{
    Ray    ray;
    float3 throughput;
    uint   depth;
    uint   materialFrom;
};

__device__ Ray GetCameraRay(const CRRT::FrameView& frame, const Camera& camera, uint x_pixel, uint y_pixel)
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

    return Ray {camera.position, rayDir};
}

__global__ void RayMarch(CRRT::FrameView frame, Camera camera, CudaBrickTree tree, CudaLights lights)
{
    uint x_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_pixel = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_pixel >= frame.width || y_pixel >= frame.height) return;
    
    // Initialize ray state
    RayState      stack[MAX_RAY_STACK];
    uint          stackSize = 0;;
    
    // Initialize camera ray
    stack[stackSize++] = {
        GetCameraRay(frame, camera, x_pixel, y_pixel),
        make_float3(1.0f),
        0u,
        BACKGROUND_MAT_INDEX
    };

    // Compute outgoing radiance for this pixel
    float3 L_o = make_float3(0.0f);
    while (stackSize > 0)
    {
        RayState     state = stack[--stackSize];
        BrickTreeHit hit   = TraceBrickTree(tree, state.ray, state.materialFrom);

        if (!hit.isHit) {
            // Shade ground plane Hit
            const float groundZ = float(tree.offset_z);
            if (state.ray.d.z < 0.0f && state.ray.o.z >= groundZ) {
                const float  groundT = (groundZ - state.ray.o.z) / state.ray.d.z;
                const float3 p = state.ray.o + groundT * state.ray.d;
                const float3 n = make_float3(0, 0, 1);
                
                bool visibilities[MAX_DIRECTIONAL_LIGHTS];
                traceVisibility(tree, lights, visibilities, p, n);
                
                float3 groundLight = make_float3(AMBIENT_LIGHT);
                for (uint i = 0; i < lights.directionalLightCount && i < MAX_DIRECTIONAL_LIGHTS; ++i) {
                    const float cosine = fmaxf(dot(n, normalize(-lights.directionalLights[i].direction)), 0.0f);
                    groundLight += Li_light(lights, visibilities, i) * (cosine / PI);
                }

                // fade plane into sky
                float3 groundColor = GROUND_COLOR * groundLight;
                float fog = 1.0f - expf(-0.00005f * fmaxf(abs(p.x), abs(p.y)));
                L_o += state.throughput * lerp(groundColor, BACKGROUND_COLOR, fog);
                continue;
            }
            
            // Shade background/Sky-box hit
            L_o += state.throughput * BACKGROUND_COLOR;
            continue;
        }

        // hit material data
        const  Material matNow   = tree.materials[hit.material];
        const  Material matFrom  = tree.materials[state.materialFrom];
        const  Material& surface = hit.material == BACKGROUND_MAT_INDEX ? matFrom : matNow; // if ray leaves glass, shade the glass not air
        const  float3 albedo = surface.albedo;
        const  float  r = surface.roughness;
        const  float  t = surface.transmission;

        // hit face data
        const float3 p = hit.position;
        const float3 v = normalize(-state.ray.d);
        const float3 n = dot(hit.normal, v) > 0.0f ? normalize(hit.normal) : -1*normalize(hit.normal);

        // lights visible from hit surface
        bool visibilities[MAX_DIRECTIONAL_LIGHTS];
        traceVisibility(tree, lights, visibilities, p, n);

        // Blend of material types
        const float diffuseWeight = (1.0f - t); // microfacet diffuse dieletric material
        const float glassWeight   = t;          // ideal dieletric glass material
       
        // Integrate attenuation along path in volume
        const float  d = hit.t;
        float3 T = make_float3(
            expf(-matFrom.sigma_a.x * d),
            expf(-matFrom.sigma_a.y * d),
            expf(-matFrom.sigma_a.z * d)
        );
        state.throughput *= T;

        // Ambient contribution for the opaque portion.
        float3 L_o_lights = AMBIENT_LIGHT * albedo * diffuseWeight;

        // Diffuse and specular from opaque surfaces
        for (uint i = 0; i < lights.directionalLightCount && i < MAX_DIRECTIONAL_LIGHTS; ++i) {
            if (!visibilities[i])
                continue;

            // light direction
            float3 l = normalize(-lights.directionalLights[i].direction);

            float cosTheta = fmaxf(dot(n, l), 0.0f);
            if (cosTheta <= 0.0f || dot(n, v) <= 0.0f)
                continue;

            // Fresnel
            float  F_dielectric_microfacet = fresnel_dielectric_microfacet(l, v, matFrom.ior, matNow.ior);

            // Brdfs
            float3 f_dieletric_microfacet_diffuse  = diffuseWeight * 
                (1.0f - F_dielectric_microfacet) * brdf_diffuse_microfacet(n, v, l, albedo, r);
            float3 f_dieletric_microfacet_specular = r > EPS
                ? brdf_specular_microfacet(n, v, l, r, make_float3(F_dielectric_microfacet))
                : make_float3(0.0f);
            float3 f_surface = f_dieletric_microfacet_diffuse + f_dieletric_microfacet_specular ;

            // Integrate direct lighting
            float3 L_light = Li_light(lights, visibilities, i);

            // Summation form of direct light rendering equation
            L_o_lights += f_surface * L_light * cosTheta;
        }
        L_o += state.throughput * L_o_lights;

        // Cast Recursive rays
        if (state.depth >= MAX_BOUNCES || glassWeight <= 0.0f)
            continue;
        
        float F_ideal             = fresnel_ideal(n, v, matFrom.ior, matNow.ior);
        float reflectionWeight    = glassWeight * F_ideal;
        float refractionWeight    = glassWeight * (1.0f - F_ideal);
        const float maxThroughput = fmaxf(state.throughput.x, fmaxf(state.throughput.y, state.throughput.z));
        
        // Cast reflection ray allow for early termination
        if (maxThroughput * reflectionWeight > MIN_THROUGHPUT && stackSize < MAX_RAY_STACK)
        {
            float3 reflectDir = normalize(
                reflect(state.ray.d, n)
            );

            RayState ray_reflect = {
                {p + n * EPS_RAY, reflectDir},
                state.throughput * reflectionWeight,
                state.depth + 1,
                state.materialFrom
            };
            
            stack[stackSize++] = ray_reflect;
        }

        // Cast refraction ray allow for early termination
        if (maxThroughput * refractionWeight > MIN_THROUGHPUT && stackSize < MAX_RAY_STACK) {
            float3 refractDir = normalize(
                refract(state.ray.d, n, matFrom.ior, matNow.ior)
            );

            RayState ray_refract = {
                {p - n * EPS_RAY, refractDir},
                state.throughput * refractionWeight,
                state.depth + 1,
                hit.material
            };

            stack[stackSize++] = ray_refract;
        }   
    }
    
    float3 sRgb = linearToDitherSrgb(L_o, x_pixel, y_pixel);
    CRRT::WriteFrameSrgb(frame, x_pixel, y_pixel, make_float4(sRgb, 1.0f));
};
