#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <stack>
#include <vector>
#include <cstddef>

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
    unsigned int minX; 
    unsigned int minY; 
    unsigned int minZ;
    unsigned int maxX; 
    unsigned int maxY; 
    unsigned int maxZ;
};

template<typename T>
struct Brick
{
    std::array<T, BRICK_SIZE> data;
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

    // 8 Children
    int child1 = -1;
    int child2 = -1;
    int child3 = -1;
    int child4 = -1;
    int child5 = -1;
    int child6 = -1;
    int child7 = -1;
    int child8 = -1;

    // Get children index in tree using indices like a 3d array
    int fetchChildIndex(unsigned int localX, unsigned int localY, unsigned int localZ) const
    {
        assert(localX < 2 && localY < 2 && localZ < 2);
        if (localX >= 2 || localY >= 2 || localZ >= 2) return -1;
        const int children[8] = {
            child1, child2, child3, child4, child5, child6, child7, child8
        };
        return children[localX + 2 * localY + 4 * localZ];
    }

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
};

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
        for (size_t z = aabb.minZ; z < aabb.maxZ; ++z) {
            for (size_t y = aabb.minY; y < aabb.maxY; ++y) {
                for (size_t x = aabb.minX; x < aabb.maxX; ++x) {
                    bool occupied = fetchVoxelData(x, y, z) != 0;
                    if (occupied) return false;
        }}}    
        return true; 
    };

    // check that aabb has shrunk to a size we consider a brick
    auto isBrick = [](Aabb& aabb) -> bool {
        return aabb.maxX - aabb.minX <= BRICK_DIM_X 
            && aabb.maxY - aabb.minY <= BRICK_DIM_Y 
            && aabb.maxZ - aabb.minZ <= BRICK_DIM_Z;
    };

    // Root node
    Aabb rootAabb = { 
        0, 0, 0, 
        outTree.grid_size_x, outTree.grid_size_y, outTree.grid_size_z
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
            for (size_t z = node.aabb.minZ; z < node.aabb.maxZ; ++z) {
                for (size_t y = node.aabb.minY; y < node.aabb.maxY; ++y) {
                    for (size_t x = node.aabb.minX; x < node.aabb.maxX; ++x) {
                        uint8_t palleteIndex = fetchVoxelData(x, y, z);
                        brick.data[localIndex++] = palleteIndex; 
            }}}  
            continue;
        }    

        // Initialize and add 8 children nodes to tree and stack
        const Aabb a = node.aabb;
        const unsigned int firstChild = static_cast<unsigned int>(outTree.nodes.size());
        const unsigned int midX = a.minX + (a.maxX - a.minX) / 2;
        const unsigned int midY = a.minY + (a.maxY - a.minY) / 2;
        const unsigned int midZ = a.minZ + (a.maxZ - a.minZ) / 2;
        for (unsigned int i = 0; i < 8; ++i) {
            Aabb childAabb = {
                (i & 1) ? midX : a.minX,
                (i & 2) ? midY : a.minY,
                (i & 4) ? midZ : a.minZ,
                (i & 1) ? a.maxX : midX,
                (i & 2) ? a.maxY : midY,
                (i & 4) ? a.maxZ : midZ
            };

            BrickTreeNode<uint8_t> childNode = {};
            childNode.type = BrickTreeNode<uint8_t>::INTERNAL_NODE;
            childNode.aabb = childAabb;
            
            outTree.nodes.push_back(childNode);    
            nodeIndices.push(firstChild + i);
        }

        // set child pointers for parrent node
        auto& parent = outTree.nodes[nodeIndex];
        parent.child1 = firstChild;
        parent.child2 = firstChild + 1;
        parent.child3 = firstChild + 2;
        parent.child4 = firstChild + 3;
        parent.child5 = firstChild + 4;
        parent.child6 = firstChild + 5;
        parent.child7 = firstChild + 6;
        parent.child8 = firstChild + 7;
    }
};


