#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "ogt_vox.h"

#include "BrickTree.hpp"

bool LoadVoxels(const std::string& filePath, BrickTree<uint8_t>& brickMap, std::string& error, uint32_t scene_read_flags = 0)
{
    FILE* fp = nullptr;
    if (fopen_s(&fp, filePath.c_str(), "rb") != 0 || !fp) {
        error = "Could not open voxel file at: " + filePath;
        return false;
    }

    fseek(fp, 0, SEEK_END);
    uint32_t buffer_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* buffer = new uint8_t[buffer_size];
    fread(buffer, buffer_size, 1, fp);
    fclose(fp);

    const ogt_vox_scene* scene = ogt_vox_read_scene_with_flags(buffer, buffer_size, scene_read_flags);
    if (!scene || scene->num_models <= 0 || !scene->models[0]) {
        delete[] buffer;
        if (scene) {
            ogt_vox_destroy_scene(scene);
        }
        error = "No models found in voxel file.";
        return false;
    }

    const ogt_vox_model* model = scene->models[0];
    if (!model->voxel_data || model->size_x <= 0 || model->size_y <= 0 || model->size_z <= 0) {
        delete[] buffer;
        ogt_vox_destroy_scene(scene);
        error = "Model in voxel file had no data or had a 0 dimension at: " + filePath;
        return false;
    }

    BuildBrickTree(brickMap, model);
    SetPalleteBrickTree(brickMap, scene);

    delete[] buffer;
    ogt_vox_destroy_scene(scene);
    error.clear();
    return true;
};
