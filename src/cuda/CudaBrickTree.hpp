#pragma once

#include "BrickTree.hpp"

/*
    Cuda device pointers for brick tree
*/

struct CudaBrickTree
{
    BrickTreeNode* nodes = nullptr;
    Brick*  bricks       = nullptr;
    Material* materials = nullptr;
    size_t nodeCount     = 0;
    size_t brickCount    = 0;
    unsigned int grid_size_x = 0, grid_size_y = 0, grid_size_z = 0;
    unsigned int offset_x = 0,    offset_y = 0,    offset_z = 0;
};

inline bool ReleaseBrickTree(CudaBrickTree& tree)
{
    bool success = cudaFree(tree.materials) == cudaSuccess;
    success &= cudaFree(tree.bricks)        == cudaSuccess;
    success &= cudaFree(tree.nodes)         == cudaSuccess;
    tree = {};
    return success;
}

inline bool UploadBrickTree(const BrickTree& tree, CudaBrickTree& deviceTree)
{
    if (deviceTree.nodes || deviceTree.bricks || deviceTree.materials)
        return false;

    deviceTree.nodeCount   = tree.nodes.size();
    deviceTree.brickCount  = tree.bricks.size();
    deviceTree.grid_size_x = tree.grid_size_x;
    deviceTree.grid_size_y = tree.grid_size_y;
    deviceTree.grid_size_z = tree.grid_size_z;
    deviceTree.offset_x    = tree.offset_x;
    deviceTree.offset_y    = tree.offset_y;
    deviceTree.offset_z    = tree.offset_z;

    auto upload = [](auto*& destination, const auto* source, size_t bytes) {
        if (bytes == 0) return true;
        cudaError_t error = cudaMalloc(&destination, bytes);
        if (error == cudaSuccess)
            error = cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice);
        return error == cudaSuccess;
    };

    if (!upload(deviceTree.nodes, tree.nodes.data(), tree.nodes.size() * sizeof(BrickTreeNode)) ||
        !upload(deviceTree.bricks, tree.bricks.data(), tree.bricks.size() * sizeof(Brick)) ||
        !upload(deviceTree.materials, tree.materials, sizeof(tree.materials))) 
    {
        ReleaseBrickTree(deviceTree);
        return false;
    }
    return true;
}

