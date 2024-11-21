#!/bin/bash -l

export BUILD_DIR=../../../build/benchmark/mrmp
mkdir -p $BUILD_DIR

module purge
module load nvhpc openmpi

echo "Compiling MPI CUDA Base"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-base stencil-2d-mpi-cuda-base.cu
