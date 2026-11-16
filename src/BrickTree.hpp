#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stack>
#include <vector>
#include <cstddef>

#include <cuda_runtime.h>
#include <helper_math.h>

#include "ogt_vox.h"

/*
    Sparse Voxel Octree using Brick Maps
*/

constexpr unsigned int BRICK_DIM_X = 8;
constexpr unsigned int BRICK_DIM_Y = 8;
constexpr unsigned int BRICK_DIM_Z = 8;
constexpr unsigned int BRICK_SIZE = BRICK_DIM_X * BRICK_DIM_Y * BRICK_DIM_Z;

struct Aabb
{
    float3 min;
    float3 max;
};

template<typename T>
struct Brick
{
    T data[BRICK_SIZE];
};

template<typename T>
struct BrickTreeNode
{
    enum NodeType {
        INTERNAL_NODE,
        LEAF_NODE_BRICK,
        LEAF_NODE_BACKGROUND
    } type;

    union {
        int brickIndex;
        T   backgroundValue;
    } data;

    unsigned int children[8];

    // Aabb
    Aabb aabb;
};

template<typename T>
struct BrickTree
{
    // Size of bricktree volume including padding
    unsigned int grid_size_x = 0;
    unsigned int grid_size_y = 0;
    unsigned int grid_size_z = 0;

    // To keep grid cells cubes
    unsigned int offset_x = 0;
    unsigned int offset_y = 0;
    unsigned int offset_z = 0;

    // Data
    std::vector<BrickTreeNode<T>> nodes;
    std::vector<Brick<T>>         bricks;
    uchar4 palette[256];
};

inline void SetPalleteBrickTree(BrickTree<uint8_t>& outTree, const ogt_vox_scene* scene)
{
    for (unsigned int i = 0; i < 256; ++i) {
        const ogt_vox_rgba& c = scene->palette.color[i];
        outTree.palette[i] = make_uchar4(c.r, c.g, c.b, c.a);
    }
}

inline void BuildBrickTree(BrickTree<uint8_t>& outTree, const ogt_vox_model* model)
{
    outTree.nodes.clear();
    outTree.bricks.clear();

    // find minimum dimensions of bounding cube 
    unsigned int gridSize = std::max({BRICK_DIM_X, BRICK_DIM_Y, BRICK_DIM_Z});
    while (gridSize < model->size_x || gridSize < model->size_y || gridSize < model->size_z)
        gridSize *= 2;

    outTree.grid_size_x = gridSize;
    outTree.grid_size_y = gridSize;
    outTree.grid_size_z = gridSize;
    outTree.offset_x = outTree.grid_size_x - model->size_x;
    outTree.offset_y = outTree.grid_size_y - model->size_y;
    outTree.offset_z = outTree.grid_size_z - model->size_z;

    // get voxel data (pallete index)
    auto fetchVoxelData = [&](unsigned int x, unsigned int y, unsigned int z) -> uint8_t {
        if (x < outTree.offset_x || y < outTree.offset_y || z < outTree.offset_z)
            return 0;
        if (x - outTree.offset_x >= model->size_x ||
            y - outTree.offset_y >= model->size_y ||
            z - outTree.offset_z >= model->size_z)
            return 0;

        size_t index = 
            (x - outTree.offset_x) +
            (y - outTree.offset_y) * size_t(model->size_x) +
            (z - outTree.offset_z) * size_t(model->size_x) * size_t(model->size_y);
        return model->voxel_data[index];
    };

    // check that every cell isn't occupied
    auto isBackground = [&](Aabb& aabb) -> bool {
        for (size_t z = aabb.min.z; z < aabb.max.z; ++z) {
            for (size_t y = aabb.min.y; y < aabb.max.y; ++y) {
                for (size_t x = aabb.min.x; x < aabb.max.x; ++x) {
                    bool occupied = fetchVoxelData(x, y, z) != 0;
                    if (occupied) return false;
        }}}    
        return true; 
    };

    // check that aabb has shrunk to a size we consider a brick
    auto isBrick = [](Aabb& aabb) -> bool {
        const float3 extent = aabb.max - aabb.min;
        return extent.x <= BRICK_DIM_X
            && extent.y <= BRICK_DIM_Y
            && extent.z <= BRICK_DIM_Z;
    };

    // Root node
    Aabb rootAabb = { 
        make_float3(0.0f),
        make_float3(static_cast<float>(gridSize))
    };
    BrickTreeNode<uint8_t> rootNode = {};
    rootNode.type = BrickTreeNode<uint8_t>::INTERNAL_NODE;
    rootNode.aabb = rootAabb;
    outTree.nodes.push_back(rootNode);

    // Push root onto stack and build tree
    std::stack<unsigned int> nodeIndices;
    nodeIndices.push(0);
    while (!nodeIndices.empty()) {
        // Get Top Node
        unsigned int nodeIndex = nodeIndices.top();
        nodeIndices.pop();
        BrickTreeNode<uint8_t>& node = outTree.nodes[nodeIndex];

        // Check if it is unoccupied
        if (isBackground(node.aabb)) {
            node.type = BrickTreeNode<uint8_t>::LEAF_NODE_BACKGROUND; 
            node.data.backgroundValue = 0;
            continue;
        }

        // Check if size constitutes a brick
        if (isBrick(node.aabb)) {
            node.type = BrickTreeNode<uint8_t>::LEAF_NODE_BRICK;
            node.data.brickIndex = outTree.bricks.size();

            // Create brick
            outTree.bricks.emplace_back();
            Brick<uint8_t>& brick = outTree.bricks.back();

            // Copy Data into brick
            unsigned int localIndex = 0;
            for (size_t z = node.aabb.min.z; z < node.aabb.max.z; ++z) {
                for (size_t y = node.aabb.min.y; y < node.aabb.max.y; ++y) {
                    for (size_t x = node.aabb.min.x; x < node.aabb.max.x; ++x) {
                        uint8_t palleteIndex = fetchVoxelData(x, y, z);
                        brick.data[localIndex++] = palleteIndex; 
            }}}  
            continue;
        }    

        // Initialize and add 8 children nodes to tree and stack
        const Aabb a = node.aabb;
        const unsigned int firstChild = static_cast<unsigned int>(outTree.nodes.size());
        const float3 midpoint = a.min + (a.max - a.min) * 0.5f;
        for (unsigned int i = 0; i < 8; ++i) {
            Aabb childAabb = {
                make_float3(
                    (i & 1) ? midpoint.x : a.min.x,
                    (i & 2) ? midpoint.y : a.min.y,
                    (i & 4) ? midpoint.z : a.min.z),
                make_float3(
                    (i & 1) ? a.max.x : midpoint.x,
                    (i & 2) ? a.max.y : midpoint.y,
                    (i & 4) ? a.max.z : midpoint.z)
            };

            // make child node
            BrickTreeNode<uint8_t> childNode = {};
            childNode.type = BrickTreeNode<uint8_t>::INTERNAL_NODE;
            childNode.aabb = childAabb;
            
            // set child pointer
            outTree.nodes[nodeIndex].children[i] = firstChild + i;

            // push to tree and stack
            outTree.nodes.push_back(childNode);    
            nodeIndices.push(firstChild + i);
        }
    }
};


