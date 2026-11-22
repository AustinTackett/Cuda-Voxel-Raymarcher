
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "ogt_vox.h"
#include "BrickTree.hpp"

/*
    Load .vox model into brick tree on the cpu
*/
struct PlacedVoxel {
    int64_t x, y, z;
    uint8_t color;
};

inline bool FlattenVoxScene(const ogt_vox_scene& scene, ogt_vox_model& merged, std::vector<uint8_t>& voxels)
{
    std::vector<PlacedVoxel> placed;
    std::array<int64_t, 3>   minPos = {INT64_MAX, INT64_MAX, INT64_MAX};
    std::array<int64_t, 3>   maxPos = {INT64_MIN, INT64_MIN, INT64_MIN};

    auto layerHidden = [&](uint32_t i) {
        return i < scene.num_layers && scene.layers[i].hidden;
    };

    // Collect voxels from all visible instances.
    for (uint32_t i = 0; i < scene.num_instances; ++i) {
        // Check instance is visible
        const auto& instance = scene.instances[i];
        if (instance.hidden || layerHidden(instance.layer_index))
            continue;

        // Check group is visible
        bool hidden = false;
        for (uint32_t g = instance.group_index; g < scene.num_groups;
            g = scene.groups[g].parent_group_index) {
            const auto& group = scene.groups[g];
            if (group.hidden || layerHidden(group.layer_index)) {
                hidden = true;
                break;
            }
        }
        if (hidden || instance.model_index >= scene.num_models)
            continue;

        // Get model data
        const auto* model = scene.models[instance.model_index];
        if (!model || !model->voxel_data)
            continue;

        const auto t = ogt_vox_sample_instance_transform_global(
            &instance, 0, &scene);

        // Move voxel coordinates from model space to world space
        for (uint32_t z = 0; z < model->size_z; ++z) {
            for (uint32_t y = 0; y < model->size_y; ++y) {
                for (uint32_t x = 0; x < model->size_x; ++x) {
                    const size_t index = x + size_t(model->size_x) *
                        (y + size_t(model->size_y) * z);

                    const uint8_t color = model->voxel_data[index];
                    if (!color) continue;

                    const double px = double(x) + 0.5 - model->size_x / 2.0;
                    const double py = double(y) + 0.5 - model->size_y / 2.0;
                    const double pz = double(z) + 0.5 - model->size_z / 2.0;

                    PlacedVoxel v {
                        int64_t(std::floor(t.m00*px + t.m10*py + t.m20*pz + t.m30)),
                        int64_t(std::floor(t.m01*px + t.m11*py + t.m21*pz + t.m31)),
                        int64_t(std::floor(t.m02*px + t.m12*py + t.m22*pz + t.m32)),
                        color
                    };

                    placed.push_back(v);

                    const int64_t p[3] = {v.x, v.y, v.z};
                    for (int a = 0; a < 3; ++a) {
                        minPos[a] = std::min(minPos[a], p[a]);
                        maxPos[a] = std::max(maxPos[a], p[a]);
                    }
        }}}
    }

    if (placed.empty())
        return false;

    // Allocate the combined voxel grid.
    std::array<uint32_t, 3> size;
    size_t count = 1;

    for (int a = 0; a < 3; ++a) {
        const uint64_t extent = uint64_t(maxPos[a] - minPos[a]) + 1;
        if (extent > UINT32_MAX || extent > voxels.max_size() / count)
            return false;

        size[a] = uint32_t(extent);
        count *= size[a];
    }

    voxels.assign(count, 0);

    // Copy all placed voxels into the combined grid.
    for (const auto& v : placed) {
        const size_t x = size_t(v.x - minPos[0]);
        const size_t y = size_t(v.y - minPos[1]);
        const size_t z = size_t(v.z - minPos[2]);

        const size_t index = x + size_t(size[0]) *
            (y + size_t(size[1]) * z);

        if (!voxels[index] || scene.file_version < 200)
            voxels[index] = v.color;
    }

    merged = {};
    merged.size_x = size[0];
    merged.size_y = size[1];
    merged.size_z = size[2];
    merged.voxel_data = voxels.data();

    return true;
}

inline bool LoadVoxels(const std::string& path, BrickTree& tree, std::string& error, uint32_t flags = 0)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "Could not open voxel file.";
        return false;
    }

    std::vector<uint8_t> bytes(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    if (bytes.empty() || bytes.size() > UINT32_MAX) {
        error = "Invalid or oversized voxel file.";
        return false;
    }

    std::unique_ptr<const ogt_vox_scene, decltype(&ogt_vox_destroy_scene)> scene(
        ogt_vox_read_scene_with_flags(
            bytes.data(), uint32_t(bytes.size()),
            flags | k_read_scene_flags_groups),
        &ogt_vox_destroy_scene);

    if (!scene) {
        error = "Could not parse voxel scene.";
        return false;
    }

    ogt_vox_model merged{};
    std::vector<uint8_t> voxels;

    if (!FlattenVoxScene(*scene, merged, voxels)) {
        error = "Empty or oversized voxel scene.";
        return false;
    }

    BuildBrickTree(tree, &merged);
    SetMaterialBrickTree(tree, scene.get());

    error.clear();
    return true;
}
