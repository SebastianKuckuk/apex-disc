#!/bin/bash

export BUILD_DIR=../../../build/benchmark/mrsp
mkdir -p $BUILD_DIR

module purge
module load nvhpc openmpi

echo "Compiling MPI CUDA Base"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-base stencil-2d-mpi-cuda-base.cu

echo "Compiling MPI CUDA Streams"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-streams stencil-2d-mpi-cuda-streams.cu

echo "Compiling MPI CUDA Batched"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-batched stencil-2d-mpi-cuda-batched.cu

echo "Compiling MPI CUDA Fused"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused stencil-2d-mpi-cuda-fused.cu

module purge
module load nvhpc hpcx
export NVSHMEM_HOME=/apps/SPACK/0.19.1/opt/linux-almalinux8-zen/gcc-8.5.0/nvhpc-23.7-bzxcokzjvx4stynglo4u2ffpljajzlam/Linux_x86_64/23.7/comm_libs/nvshmem
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib

echo "Compiling NVSHMEM CUDA"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-nvshmem-cuda \
     -rdc=true -I $NVSHMEM_HOME/include -L $NVSHMEM_HOME/lib -lnvshmem -lcuda -lnvidia-ml stencil-2d-nvshmem-cuda.cu
