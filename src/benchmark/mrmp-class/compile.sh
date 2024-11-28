#!/bin/bash -l

export BUILD_DIR=../../../build/benchmark/mrmp-class
mkdir -p $BUILD_DIR

module purge
module load nvhpc openmpi

echo "Compiling MPI CUDA Base"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-base stencil-2d-mpi-cuda-base.cu

echo "Compiling MPI CUDA Fused Direct"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct stencil-2d-mpi-cuda-fused-direct.cu

echo "Compiling MPI CUDA Fused Direct Streams"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-streams stencil-2d-mpi-cuda-fused-direct-streams.cu

echo "Compiling MPI CUDA Fused Direct Batched"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-batched stencil-2d-mpi-cuda-fused-direct-batched.cu

echo "Compiling MPI CUDA Fused Direct Graph"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-graph stencil-2d-mpi-cuda-fused-direct-graph.cu

echo "Compiling MPI CUDA Fused Direct Graphs"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-graphs stencil-2d-mpi-cuda-fused-direct-graphs.cu

echo "Compiling MPI CUDA Fused Direct Graph Callback"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-graph-callback stencil-2d-mpi-cuda-fused-direct-graph-callback.cu

echo "Compiling NCCL CUDA Base"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-nccl-cuda-base stencil-2d-nccl-cuda-base.cu -lnccl

echo "Compiling NCCL CUDA Multistream"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-nccl-cuda-multistream stencil-2d-nccl-cuda-multistream.cu -lnccl

echo "Compiling NCCL CUDA Fused Direct Streams"
nvcc -ccbin=mpic++ -arch=sm_80 -O3 -std=c++17 -o $BUILD_DIR/stencil-2d-nccl-cuda-fused-direct-streams stencil-2d-nccl-cuda-fused-direct-streams.cu -lnccl
