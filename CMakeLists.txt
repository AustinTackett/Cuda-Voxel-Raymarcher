cmake_minimum_required(VERSION 3.28)
project(cuda-voxel-raymarcher LANGUAGES C CXX CUDA)
include(FetchContent)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_Declare(crrt
    GIT_REPOSITORY https://github.com/AustinTackett/crrt.git
    GIT_TAG v0.0.2
)
FetchContent_Declare(opengametools
    GIT_REPOSITORY https://github.com/jpaver/opengametools.git
    GIT_TAG 17b41f906392dba8636a70554f0b21d3739ca514
    SOURCE_SUBDIR src
)
FetchContent_Declare(cuda_math
    URL https://raw.githubusercontent.com/NVIDIA/cuda-samples/master/Common/helper_math.h
    DOWNLOAD_NO_EXTRACT TRUE
    DOWNLOAD_NAME helper_math.h
    TIMEOUT 30
)

FetchContent_MakeAvailable(crrt opengametools cuda_math)

add_executable(cuda-voxel-raymarcher
    main.cpp
    kernels/Draw.cu
    kernels/RayMarch.cu
    kernels/TraceBrickTree.cu
)
target_include_directories(cuda-voxel-raymarcher PRIVATE
    "${cuda_math_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    "${opengametools_SOURCE_DIR}/src"
)
target_link_libraries(cuda-voxel-raymarcher PRIVATE CRRT)
set_target_properties(cuda-voxel-raymarcher PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES all-major
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
