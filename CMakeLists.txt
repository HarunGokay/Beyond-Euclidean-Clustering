cmake_minimum_required(VERSION 3.14)
project(diffusion_clustering)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen and Spectra: use installed ones if there are any, otherwise download them
find_package(Eigen3 QUIET NO_MODULE)
find_package(Spectra QUIET)

include(FetchContent)
if (NOT TARGET Eigen3::Eigen)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()
if (NOT TARGET Spectra::Spectra)
    FetchContent_Declare(spectra
        GIT_REPOSITORY https://github.com/yixuan/spectra.git
        GIT_TAG v1.2.0)
    FetchContent_MakeAvailable(spectra)
endif()

add_executable(diffusion_clustering
    src/main.cpp src/utils.cpp src/kmeans.cpp src/hdbscan.cpp
    src/heatkernel.cpp src/preservetopo.cpp data/datagen.cpp)

target_include_directories(diffusion_clustering PRIVATE src data)

target_link_libraries(diffusion_clustering PRIVATE Eigen3::Eigen Spectra::Spectra)

# Definitions for MSVC
target_compile_definitions(diffusion_clustering PRIVATE _USE_MATH_DEFINES _ENFORCE_MATCHING_ALLOCATORS=0)
