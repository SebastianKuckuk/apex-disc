#pragma once

#include "../../util.h"


template <typename tpe>
inline void initStencil2D(tpe *__restrict__ u, tpe *__restrict__ uNew, const size_t nx, const size_t ny) {
    for (size_t i1 = 0; i1 < ny; ++i1) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
            if (0 == i0 || nx - 1 == i0 || 0 == i1 || ny - 1 == i1) {
                u[i0 + i1 * nx] = (tpe)0;
                uNew[i0 + i1 * nx] = (tpe)0;
            } else {
                u[i0 + i1 * nx] = (tpe)1;
                uNew[i0 + i1 * nx] = (tpe)1;
            }
        }
    }
}

template <typename tpe>
inline tpe checkSolutionStencil2D(const tpe *const __restrict__ u, const tpe *const __restrict__ uNew, const size_t nx, const size_t ny, const size_t nIt) {
    tpe res = 0;

    // skip halo and one additional layer to avoid necessity to communicate halos and setting boundary conditions
    // this will slightly pertubate the final residual when subdividing patches
    for (size_t i1 = 2; i1 < ny - 2; ++i1) {
        for (size_t i0 = 2; i0 < nx - 2; ++i0) {
            const tpe localRes = 4 * u[i0 + i1 * nx] - u[i0 + i1 * nx + 1] - u[i0 + i1 * nx - 1] - u[i0 + nx * (i1 + 1)] - u[i0 + nx * (i1 - 1)];
            res += localRes * localRes;
        }
    }

    return res;
}
