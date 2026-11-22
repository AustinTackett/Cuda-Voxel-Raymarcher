#include "BrickTree.hpp"
#include "TraceBrickTree.hpp"

#include <cmath>
#include <cstdint>

#include <device_types.h>
#include <helper_math.h>
#include <math_constants.h>

/*
    Device Functions for Traversing Brick Tree
*/

/*
    Math Utils
*/
__device__ __forceinline__ bool anyLess(int3 v, int3 threshold)
{
    return v.x < threshold.x || v.y < threshold.y || v.z < threshold.z;
}

__device__ __forceinline__ bool anyGreaterEqual(int3 v, int3 threshold)
{
    return v.x >= threshold.x || v.y >= threshold.y || v.z >= threshold.z;
}

__device__ __forceinline__ uint argMax(float3 v)
{
    return (v.x >= v.y) ? ((v.x >= v.z) ? 0 : 2) : ((v.y >= v.z) ? 1 : 2);
}

/*
    Aabb intersections
*/
__device__ AabbHit AabbIntersect(Ray ray, Aabb aabb) 
{
    AabbHit hit {false, 0.0,  CUDART_INF_F};

    for (int axis = 0; axis < 3; ++axis) {
        float rayOriginAxis = axis == 0 ? ray.o.x : axis == 1 ? ray.o.y : ray.o.z;
        float rayDirectionAxis = axis == 0 ? ray.d.x : axis == 1 ? ray.d.y : ray.d.z;
        float aabbMinAxis = axis == 0 ? aabb.min.x : axis == 1 ? aabb.min.y : aabb.min.z;
        float aabbMaxAxis = axis == 0 ? aabb.max.x : axis == 1 ? aabb.max.y : aabb.max.z;

        // check if origin is in side line segment if direction is parrallel
        if (rayDirectionAxis == 0.0) {
            // Assign shared faces to exactly one node: [min, max).
            if (rayOriginAxis < aabbMinAxis || rayOriginAxis >= aabbMaxAxis) {
                return hit;
            }
            continue;
        }

        // find t of entry and exit of aabb
        float invDirection = 1.0 / rayDirectionAxis;
        float t0 = (aabbMinAxis - rayOriginAxis) * invDirection;
        float t1 = (aabbMaxAxis - rayOriginAxis) * invDirection;

        hit.tMin = fmaxf(hit.tMin, fminf(t0, t1));
        hit.tMax = fminf(hit.tMax, fmaxf(t0, t1));

        if (hit.tMin >= hit.tMax) {
            return hit;
        }
    }

    hit.isHit = true;
    return hit;
}

/*
    Voxel Intersections
*/

// index in voxel of ray brick intersection
__device__ __forceinline__ int3 GetVoxelCell(Aabb aabb, Ray ray, float t)
{
    const float3 voxelCellSize = (aabb.max - aabb.min) / float(BRICK_DIM_X);
    const float3 voxelCellf    = (ray.o + t * ray.d - aabb.min) / voxelCellSize;
    const int3 voxelCell = make_int3(
        ray.d.x < 0 ? int(ceilf(voxelCellf.x)) - 1 : int(floorf(voxelCellf.x)),
        ray.d.y < 0 ? int(ceilf(voxelCellf.y)) - 1 : int(floorf(voxelCellf.y)),
        ray.d.z < 0 ? int(ceilf(voxelCellf.z)) - 1 : int(floorf(voxelCellf.z))
    );
    // check bounds
    return clamp(voxelCell, make_int3(0), make_int3(BRICK_DIM_X - 1));
}

// normal of face of ray voxel intersection
__device__ __forceinline__ float3 GetHitNormal(float3 entryT, int3 stepDir)
{
    if (fmaxf(entryT.x, fmaxf(entryT.y, entryT.z)) < 0) return make_float3(0);
    uint faceHit = argMax(entryT);
    return faceHit == 0 ? make_float3(-stepDir.x, 0, 0) :
           faceHit == 1 ? make_float3(0, -stepDir.y, 0) :
           make_float3(0, 0, -stepDir.z);
}

/*
    Voxel Ray Traversal
*/
__device__ constexpr uint  MAX_RAY_STEPS   = 128;
__device__ constexpr int   MAX_STACK_DEPTH = 64;
__device__ constexpr float EPS = 1e-20f;

__device__ BrickTreeHit TraceBrickTree(const CudaBrickTree& tree, Ray ray)
{
    return TraceBrickTree(tree, ray, BACKGROUND_MAT_INDEX);
}

__device__ BrickTreeHit TraceBrickTree(const CudaBrickTree& tree, Ray ray, uint ignoreMaterial)
{
    // Check expected square dimensions
    constexpr uint brickDim = BRICK_DIM_X;
    static_assert(BRICK_DIM_X == BRICK_DIM_Y && BRICK_DIM_X == BRICK_DIM_Z);
    
    const BrickTreeNode& rootNode = tree.nodes[0];
    AabbHit rootHit = AabbIntersect(ray, rootNode.aabb);
    if (!rootHit.isHit) return BrickTreeHit{false, BACKGROUND_MAT_INDEX};

    uint stackPtr = 0;
    uint nodeIndexStack[MAX_STACK_DEPTH];
    float nodeTminStack[MAX_STACK_DEPTH];

    nodeIndexStack[stackPtr] = 0;
    nodeTminStack[stackPtr]  = rootHit.tMin;
    ++stackPtr;
    
    while(stackPtr > 0) {
        --stackPtr;
        uint nodeIndex = nodeIndexStack[stackPtr];
        const BrickTreeNode& node = tree.nodes[nodeIndex];

        // On background hit (air)
        if (node.type == BrickTreeNode::LEAF_NODE_BACKGROUND) {
            if (BACKGROUND_MAT_INDEX != ignoreMaterial) {
                float3 safeDir = ray.d;
                if (safeDir.x == 0.0f) safeDir.x = EPS;
                if (safeDir.y == 0.0f) safeDir.y = EPS;
                if (safeDir.z == 0.0f) safeDir.z = EPS;

                int3 stepDir = make_int3(
                    (ray.d.x > 0) - (ray.d.x < 0),
                    (ray.d.y > 0) - (ray.d.y < 0),
                    (ray.d.z > 0) - (ray.d.z < 0));
                
                float3 entryT = fminf((node.aabb.min - ray.o) / safeDir, (node.aabb.max - ray.o) / safeDir);
                float  hitTMin = nodeTminStack[stackPtr];
                float3 hitNormal = GetHitNormal(entryT, stepDir);
                return BrickTreeHit{true, BACKGROUND_MAT_INDEX, ray.o + ray.d * hitTMin, hitNormal, hitTMin};
            }
            continue;
        }

        // On brick hit
        if (node.type == BrickTreeNode::LEAF_NODE_BRICK) {
            const Brick& brick = tree.bricks[node.data.brickIndex];

            // What ray hit this brick
            float  tMin   = nodeTminStack[stackPtr];
            float3 rayPos = ray.o + tMin * ray.d;

            // Get voxel spatial data
            int3   voxelCell     = GetVoxelCell(node.aabb, ray, tMin);
            float3 voxelCellSize = (node.aabb.max - node.aabb.min) / float(brickDim);
            float3 voxelCellMin  = node.aabb.min + make_float3(voxelCell) * voxelCellSize;
            float3 voxelCellMax  = voxelCellMin + voxelCellSize;

            // Get direction with no 0 elements
            float3 safeDir = ray.d;
            if (safeDir.x == 0.0f) safeDir.x = EPS; // don't divide by zero
            if (safeDir.y == 0.0f) safeDir.y = EPS;
            if (safeDir.z == 0.0f) safeDir.z = EPS;

            // Get the t to reach each axis aligned boundary
            float3 t0 = (voxelCellMin - rayPos) / safeDir;
            float3 t1 = (voxelCellMax - rayPos) / safeDir;
            float3 t_next = tMin + fmaxf(t0, t1);
            if (ray.d.x == 0.0f) t_next.x = CUDART_INF_F; // always prioritize stepping in other directions
            if (ray.d.y == 0.0f) t_next.y = CUDART_INF_F;
            if (ray.d.z == 0.0f) t_next.z = CUDART_INF_F;
            
            // How far is a step in each axis
            float3 t_delta = fabs(voxelCellSize / safeDir);

            // ivec3( sign(ray.d.x), sign(ray.d.y), sign(ray.d.z) )
            int3 stepDir = make_int3(
                ray.d.x > 0 ? 1 : (safeDir.x < 0 ? -1 : 0),
                ray.d.y > 0 ? 1 : (safeDir.y < 0 ? -1 : 0),
                ray.d.z > 0 ? 1 : (safeDir.z < 0 ? -1 : 0)
            );
            
            // Traverse through brick one voxel at a time with DDA
            for (int j = 0; j < MAX_RAY_STEPS; j++) {
                uint brickIndex = voxelCell.x +
                    voxelCell.y * brickDim +
                    voxelCell.z * brickDim * brickDim;
                
                uint8_t voxelData = brick.data[brickIndex];
                if (voxelData != ignoreMaterial)
                {
                    float3 hitVoxelCellMin = node.aabb.min + make_float3(voxelCell) * voxelCellSize;
                    float3 hitVoxelCellMax = hitVoxelCellMin + voxelCellSize;
                    
                    float3 entryT  = fminf((hitVoxelCellMin - ray.o) / safeDir, (hitVoxelCellMax - ray.o) / safeDir);
                    float  hitTMin = fmaxf(0.0f, fmaxf(entryT.x, fmaxf(entryT.y, entryT.z)));

                    float3 hitPosition = ray.o + ray.d * hitTMin;
                    float3 hitNormal   = GetHitNormal(entryT, stepDir);

                    return BrickTreeHit{true, voxelData, hitPosition, hitNormal, hitTMin};
                };

                // Edge-only contacts are not hits.
                const float nextT = fminf(t_next.x, fminf(t_next.y, t_next.z));
                if (t_next.x == nextT) {
                    t_next.x += t_delta.x;
                    voxelCell.x += stepDir.x;
                }
                if (t_next.y == nextT) {
                    t_next.y += t_delta.y;
                    voxelCell.y += stepDir.y;
                }
                if (t_next.z == nextT) {
                    t_next.z += t_delta.z;
                    voxelCell.z += stepDir.z;
                }

                // Ray terminates when the next step leaves the brick
                bool outOfBounds = anyLess(voxelCell, {0, 0, 0}) ||
                    anyGreaterEqual(voxelCell, {brickDim, brickDim, brickDim});
                if (outOfBounds) {
                    break;
                }
            }
            
            continue;
        }

        // Add child nodes that intersect ray
        if (node.type == BrickTreeNode::INTERNAL_NODE) {
            // intersect all children
            uint  hitCount = 0;
            float tMins[8];
            uint  childHitIndices[8];
            for (int i = 0; i < 8; ++i) {
                uint childIndex = node.children[i];
                const BrickTreeNode& childNode = tree.nodes[childIndex];
                AabbHit aabbHit = AabbIntersect(ray,childNode.aabb);
                if (!aabbHit.isHit) continue;

                // Add hits in sorted list by ascending tMin (insertion sort)
                int j = hitCount;
                while (j > 0 && tMins[j-1] > aabbHit.tMin) {
                    tMins[j] = tMins[j - 1];
                    childHitIndices[j] = childHitIndices[j - 1];
                    --j;
                }

                tMins[j] = aabbHit.tMin;
                childHitIndices[j] = childIndex;

                ++hitCount;
            }

            if (stackPtr + hitCount > MAX_STACK_DEPTH) {
                // STACK OVERFLOW !
                continue;
            }

            // add hits in order of smallest tMin
            while (hitCount > 0) {
                nodeIndexStack[stackPtr] = childHitIndices[hitCount - 1];
                nodeTminStack[stackPtr]  = tMins[hitCount - 1];
                ++stackPtr;
                --hitCount;
            }

            continue;
        }
    }

    return BrickTreeHit{false, BACKGROUND_MAT_INDEX};
}


