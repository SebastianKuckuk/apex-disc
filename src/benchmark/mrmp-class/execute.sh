#!/bin/bash -l

export BUILD_DIR=../../../build/benchmark/mrmp-class
export N_IT=10
export N_WARMUP=2

module purge
module load nvhpc openmpi

for NX in 4096 2048; do
    NY=$NX
    for PNX in 1 2; do
        PNY=$PNX
        MPINX=$((8192 / $PNX / $NX))
        MPINY=$((8192 / $PNY / $NY))

        echo
        echo "Executing CUDA Base $MPINX x $MPINY : $PNX x $PNY : $NX x $NY"
        mpirun -n $((MPINX * MPINY)) $BUILD_DIR/stencil-2d-mpi-cuda-base double $(($NX+2)) $(($NY+2)) $N_WARMUP $N_IT $MPINX $MPINY $PNX $PNY
    done
done
