#!/bin/bash -l

export BUILD_DIR=../../../build/benchmark/mrmp-class
mkdir -p $BUILD_DIR

module purge
module load nvhpc openmpi

echo "Compiling MPI CUDA Base"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-base stencil-2d-mpi-cuda-base.cu

echo "Compiling MPI CUDA Fused Direct"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct stencil-2d-mpi-cuda-fused-direct.cu

echo "Compiling MPI CUDA Fused Streams"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-streams stencil-2d-mpi-cuda-fused-direct-streams.cu

echo "Compiling MPI CUDA Fused Direct Batched"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-batched stencil-2d-mpi-cuda-fused-direct-batched.cu
