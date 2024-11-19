#!/bin/bash -l

export BUILD_DIR=../../../build/benchmark/mrsp
export N_IT=10
export N_WARMUP=2

module purge
module load nvhpc openmpi

echo
echo "Executing MPI CUDA Base 1x1 8192**2"
mpirun -n 1 $BUILD_DIR/stencil-2d-mpi-cuda-base double 8196 8196 $N_WARMUP $N_IT 1 1

echo
echo "Executing MPI CUDA Stream 1x1 8192**2"
mpirun -n 1 $BUILD_DIR/stencil-2d-mpi-cuda-streams double 8196 8196 $N_WARMUP $N_IT 1 1

echo
echo "Executing MPI CUDA Batched 1x1 8192**2"
mpirun -n 1 $BUILD_DIR/stencil-2d-mpi-cuda-batched double 8196 8196 $N_WARMUP $N_IT 1 1

echo
echo "Executing MPI CUDA Fused 1x1 8192**2"
mpirun -n 1 $BUILD_DIR/stencil-2d-mpi-cuda-fused double 8196 8196 $N_WARMUP $N_IT 1 1

echo
echo "Executing NVSHMEM CUDA 1x1 8192**2"
mpirun -n 1 $BUILD_DIR/stencil-2d-nvshmem-cuda double 8196 8196 $N_WARMUP $N_IT 1 1

echo
echo "Executing MPI CUDA Base 2x2 4096**2"
mpirun -n 4 $BUILD_DIR/stencil-2d-mpi-cuda-base double 4098 4098 $N_WARMUP $N_IT 2 2

echo
echo "Executing MPI CUDA Stream 2x2 4096**2"
mpirun -n 4 $BUILD_DIR/stencil-2d-mpi-cuda-streams double 4098 4098 $N_WARMUP $N_IT 2 2

echo
echo "Executing MPI CUDA Batched 2x2 4096**2"
mpirun -n 4 $BUILD_DIR/stencil-2d-mpi-cuda-batched double 4098 4098 $N_WARMUP $N_IT 2 2

echo
echo "Executing MPI CUDA Fused 2x2 4096**2"
mpirun -n 4 $BUILD_DIR/stencil-2d-mpi-cuda-fused double 4098 4098 $N_WARMUP $N_IT 2 2

echo
echo "Executing NVSHMEM CUDA 2x2 4096**2"
mpirun -n 4 $BUILD_DIR/stencil-2d-nvshmem-cuda double 4098 4098 $N_WARMUP $N_IT 2 2
