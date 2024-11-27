#!/bin/bash -l

export BUILD_DIR=../../../build/benchmark/mrmp-class
export N_IT=10
export N_WARMUP=2

module purge
module load nvhpc openmpi

for NX in 8192 4096 2048 ; do
    NY=$NX
    for PNX in 4 2 1 ; do
        PNY=$PNX
        MPINX=$((8192 / $PNX / $NX))
        MPINY=$((8192 / $PNY / $NY))

        if [ $MPINX -eq 0 ] || [ $MPINY -eq 0 ]; then
            continue
        fi

        echo
        echo "Executing CUDA Base $MPINX x $MPINY : $PNX x $PNY : $NX x $NY"
        mpirun -n $((MPINX * MPINY)) $BUILD_DIR/stencil-2d-mpi-cuda-base double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $MPINX $MPINY $PNX $PNY

        echo
        echo "Executing CUDA Fused Direct $MPINX x $MPINY : $PNX x $PNY : $NX x $NY"
        mpirun -n $((MPINX * MPINY)) $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $MPINX $MPINY $PNX $PNY

        echo
        echo "Executing CUDA Fused Direct Streams $MPINX x $MPINY : $PNX x $PNY : $NX x $NY"
        mpirun -n $((MPINX * MPINY)) $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-streams double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $MPINX $MPINY $PNX $PNY

        echo
        echo "Executing CUDA Fused Direct Batched $MPINX x $MPINY : $PNX x $PNY : $NX x $NY"
        mpirun -n $((MPINX * MPINY)) $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-batched double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $MPINX $MPINY $PNX $PNY

        echo
        echo "Executing CUDA Fused Direct Graph $MPINX x $MPINY : $PNX x $PNY : $NX x $NY"
        mpirun -n $((MPINX * MPINY)) $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-graph double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $MPINX $MPINY $PNX $PNY

        echo
        echo "Executing CUDA Fused Direct Graphs $MPINX x $MPINY : $PNX x $PNY : $NX x $NY"
        mpirun -n $((MPINX * MPINY)) $BUILD_DIR/stencil-2d-mpi-cuda-fused-direct-graphs double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $MPINX $MPINY $PNX $PNY
    done
done
