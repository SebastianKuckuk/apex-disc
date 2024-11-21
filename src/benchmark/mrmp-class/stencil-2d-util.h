#pragma once

#include "../../util.h"

#include <mpi.h>


enum NeighborType { REMOTE, LOCAL, BOUNDARY };

template <typename tpe>
struct Patch {
    tpe* u;
    tpe* uNew;

    tpe *d_u;
    tpe *d_uNew;

    NeighborType neighborType[4];
    int neighborPatchIdx[4];
    int neighborMpiRank[4];

    tpe *h_d_bufSend[4];
    tpe **d_d_bufSend;
    MPI_Request reqsSend[4];

    tpe *h_d_bufRecv[4];
    tpe **d_d_bufRecv;
    MPI_Request reqsRecv[4];

    void waitAll() {
        MPI_Waitall(4, reqsSend, MPI_STATUSES_IGNORE);
        MPI_Waitall(4, reqsRecv, MPI_STATUSES_IGNORE);
    }
};

template <typename tpe>
inline void initStencil2D(Patch<tpe>* patches, const size_t nx, const size_t ny, unsigned int numPatches) {
    for (auto p = 0; p < numPatches; ++p) {
        auto& patch = patches[p];

        for (size_t i1 = 0; i1 < ny; ++i1) {
            for (size_t i0 = 0; i0 < nx; ++i0) {
                if (0 == i0 || nx - 1 == i0 || 0 == i1 || ny - 1 == i1) {
                    patch.u[i0 + i1 * nx] = (tpe)0;
                    patch.uNew[i0 + i1 * nx] = (tpe)0;
                } else {
                    patch.u[i0 + i1 * nx] = (tpe)1;
                    patch.uNew[i0 + i1 * nx] = (tpe)1;
                }
            }
        }
    }
}

template <typename tpe>
inline tpe checkSolutionStencil2D(const Patch<tpe> *const __restrict__ patches, const size_t nx, const size_t ny, const size_t nIt, const unsigned int numPatches) {
    tpe res = 0;

    // skip halo and one additional layer to avoid necessity to communicate halos and setting boundary conditions
    // this will slightly pertubate the final residual when subdividing patches
    for (auto p = 0; p < numPatches; ++p) {
        auto& patch = patches[p];

        for (size_t i1 = 2; i1 < ny - 2; ++i1) {
            for (size_t i0 = 2; i0 < nx - 2; ++i0) {
                const tpe localRes = 4 * patch.u[i0 + i1 * nx] - patch.u[i0 + i1 * nx + 1] - patch.u[i0 + i1 * nx - 1] - patch.u[i0 + nx * (i1 + 1)] - patch.u[i0 + nx * (i1 - 1)];
                res += localRes * localRes;
            }
        }
    }

    return res;
}
