#include "TraceBrickTree.hpp"

#include <cmath>
#include <cstdint>

#include <helper_math.h>
#include <math_constants.h>

/*
    Device Functions for Traversing Brick Tree
*/

__device__ __forceinline__ bool AnyLess(int3 v, int3 threshold)
{
    return v.x < threshold.x || v.y < threshold.y || v.z < threshold.z;
}

__device__ __forceinline__ bool AnyGreaterEqual(int3 v, int3 threshold)
{
    return v.x >= threshold.x || v.y >= threshold.y || v.z >= threshold.z;
}

__device__ AabbHit AabbIntersect(const Ray& ray, const Aabb& aabb) 
{
    AabbHit hit {false, 0.0,  CUDART_INF_F};

    for (int axis = 0; axis < 3; ++axis) {
        float rayOriginAxis = axis == 0 ? ray.o.x : axis == 1 ? ray.o.y : ray.o.z;
        float rayDirectionAxis = axis == 0 ? ray.d.x : axis == 1 ? ray.d.y : ray.d.z;
        float aabbMinAxis = axis == 0 ? aabb.min.x : axis == 1 ? aabb.min.y : aabb.min.z;
        float aabbMaxAxis = axis == 0 ? aabb.max.x : axis == 1 ? aabb.max.y : aabb.max.z;

        // check if origin is in side line segment if direction is parrallel
        if (rayDirectionAxis == 0.0) {
            if (rayOriginAxis <= aabbMinAxis || rayOriginAxis >= aabbMaxAxis) {
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

__device__ constexpr uint MAX_RAY_STEPS = 2048;
__device__ constexpr int MAX_STACK_DEPTH = 64;
__device__ constexpr float EPS = 1e-20f;

__device__ VoxelHit TraceVoxel(Ray& ray, BrickTreeNode<uint8_t>* nodes, Brick<uint8_t>* bricks)
{
    constexpr uint brickDim = BRICK_DIM_X;
    static_assert(BRICK_DIM_X == BRICK_DIM_Y && BRICK_DIM_X == BRICK_DIM_Z);
    const BrickTreeNode<uint8_t>& rootNode = nodes[0];
    AabbHit rootHit = AabbIntersect(ray, rootNode.aabb);
    if (!rootHit.isHit) return VoxelHit{false, 0};

    uint stackPtr = 0;
    uint nodeIndexStack[MAX_STACK_DEPTH];
    float nodeTminStack[MAX_STACK_DEPTH];

    nodeIndexStack[stackPtr] = 0;
    nodeTminStack[stackPtr]  = rootHit.tMin;
    ++stackPtr;
    
    while(stackPtr > 0) {
        --stackPtr;
        uint nodeIndex = nodeIndexStack[stackPtr];
        const BrickTreeNode<uint8_t>& node = nodes[nodeIndex];

        // return single background value of leaf
        if (node.type == BrickTreeNode<uint8_t>::LEAF_NODE_BACKGROUND) {
            continue;
        }

        // return value from brick in leaf
        if (node.type == BrickTreeNode<uint8_t>::LEAF_NODE_BRICK) {
            const Brick<uint8_t>& brick = bricks[node.data.brickIndex];

            float  tMin = nodeTminStack[stackPtr];
            float3 rayPos = ray.o + tMin * ray.d;

            float3 voxelCellSize = (node.aabb.max - node.aabb.min) / float(brickDim);
            float3 voxelCellf = (rayPos - node.aabb.min) / voxelCellSize;

            // handle that voxelCellf can be negative
            int3 voxelCell = make_int3(
                ray.d.x < 0 ? int(ceilf(voxelCellf.x)) - 1 : int(floorf(voxelCellf.x)),
                ray.d.y < 0 ? int(ceilf(voxelCellf.y)) - 1 : int(floorf(voxelCellf.y)),
                ray.d.z < 0 ? int(ceilf(voxelCellf.z)) - 1 : int(floorf(voxelCellf.z))
            ); 
            voxelCell = clamp(voxelCell, make_int3(0), make_int3(brickDim - 1));

            float3 voxelCellMin = node.aabb.min + make_float3(voxelCell) * voxelCellSize;
            float3 voxelCellMax = voxelCellMin + voxelCellSize;

            float3 safeDir = ray.d;
            if (safeDir.x == 0.0f) safeDir.x = EPS;
            if (safeDir.y == 0.0f) safeDir.y = EPS;
            if (safeDir.z == 0.0f) safeDir.z = EPS;

            float3 t0 = (voxelCellMin - rayPos) /safeDir;
            float3 t1 = (voxelCellMax - rayPos) /safeDir;
            float3 t_next = tMin + fmaxf(t0, t1);
            if (ray.d.x == 0.0f) t_next.x = CUDART_INF_F;
            if (ray.d.y == 0.0f) t_next.y = CUDART_INF_F;
            if (ray.d.z == 0.0f) t_next.z = CUDART_INF_F;
            
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
                bool ocupied = voxelData != 0;
                if (ocupied) {
                    return VoxelHit{true, voxelData};
                };

                if (t_next.x < t_next.y) {
                    if (t_next.x < t_next.z) {
                        t_next.x += t_delta.x;
                        voxelCell.x += stepDir.x;
                    } else {
                        t_next.z += t_delta.z;
                        voxelCell.z += stepDir.z;
                    }
                } else {
                    if (t_next.y < t_next.z) {
                        t_next.y += t_delta.y;
                        voxelCell.y += stepDir.y;
                    } else {
                        t_next.z += t_delta.z;
                        voxelCell.z += stepDir.z;
                    }
                }

                bool outOfBounds = 
                    AnyLess(voxelCell, {0, 0, 0}) ||
                    AnyGreaterEqual(voxelCell, {brickDim, brickDim, brickDim});
                if (outOfBounds) {
                    break;
                }
            }
            
            continue;
        }

        // Add child nodes that intersect ray
        if (node.type == BrickTreeNode<uint8_t>::INTERNAL_NODE) {
            // intersect all children
            uint  hitCount = 0;
            float tMins[8];
            uint  childHitIndices[8];
            for (int i = 0; i < 8; ++i) {
                uint childIndex = node.children[i];
                const BrickTreeNode<uint8_t>& childNode = nodes[childIndex];
                AabbHit aabbHit = AabbIntersect(ray, childNode.aabb);
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

            // check for stack overflow
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

    return VoxelHit{false, 0};
}
