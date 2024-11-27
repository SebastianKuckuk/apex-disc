#!/bin/bash -l

export BUILD_DIR=../../../build/benchmark/srmp
export N_IT=10
export N_WARMUP=2

module purge
module load nvhpc openmpi

for NX in 8192 2048 512 128; do
    NY=$NX
    PNX=$((8192 / $NX))
    PNY=$((8192 / $NY))

    echo
    echo "Executing CUDA Base $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-base double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing CUDA Patch Streams $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-patch-streams double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing CUDA Multi Streams $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-multi-streams double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing CUDA Fused Direct $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-fused-direct double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing CUDA Fused Direct Streams $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-fused-direct-streams double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing CUDA Fused Direct Batched $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-fused-direct-batched double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing CUDA Fused Direct Stream Sync $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-fused-direct-stream-sync double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing CUDA Fused Direct OpenMP $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-fused-direct-omp double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing CUDA Graph $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-cuda-graph double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing OpenMP Fused Direct $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-omp-fused-direct double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY

    echo
    echo "Executing OpenMP Task Fused Direct $PNX x $PNY : $NX x $NY"
    $BUILD_DIR/stencil-2d-omp-task-fused-direct double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $PNX $PNY
done
