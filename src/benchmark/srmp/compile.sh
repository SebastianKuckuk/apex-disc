#!/bin/bash -l

export BUILD_DIR=../../../build/benchmark/srmp
mkdir -p $BUILD_DIR

module purge
module load nvhpc openmpi

echo "Compiling CUDA Base"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-cuda-base stencil-2d-cuda-base.cu

echo "Compiling CUDA Patch Streams"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-cuda-patch-streams stencil-2d-cuda-patch-streams.cu

echo "Compiling CUDA Multi Streams"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-cuda-multi-streams stencil-2d-cuda-multi-streams.cu

echo "Compiling CUDA Fused Direct"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-cuda-fused-direct stencil-2d-cuda-fused-direct.cu

echo "Compiling CUDA Fused Direct Streams"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-cuda-fused-direct-streams stencil-2d-cuda-fused-direct-streams.cu

echo "Compiling CUDA Fused Direct Batched"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-cuda-fused-direct-batched stencil-2d-cuda-fused-direct-batched.cu

echo "Compiling CUDA Fused Direct Stream Sync"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-cuda-fused-direct-stream-sync stencil-2d-cuda-fused-direct-stream-sync.cu

echo "Compiling CUDA Fused Direct OpenMP"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -Xcompiler -fopenmp -o $BUILD_DIR/stencil-2d-cuda-fused-direct-omp stencil-2d-cuda-fused-direct-omp.cu
