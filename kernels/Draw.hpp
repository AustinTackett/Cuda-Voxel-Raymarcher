#pragma once

#include <CRRT.hpp>
#include "BrickTree.hpp"
#include "Camera.hpp"

void Draw(CRRT::FrameView frame, const Camera& camera, BrickTreeNode<uint8_t>* nodes, Brick<uint8_t>* bricks, uchar4* palette);
