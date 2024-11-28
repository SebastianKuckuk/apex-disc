#pragma once

#include "../../util.h"


template <typename tpe>
inline void initStencil2D(tpe *__restrict__ *__restrict__ u, tpe *__restrict__ *__restrict__ uNew, const size_t nx, const size_t ny,
                          const unsigned int patch_nx, const unsigned int patch_ny) {
    for (auto p1 = 0; p1 < patch_ny; ++p1) {
        for (auto p0 = 0; p0 < patch_nx; ++p0) {
            auto patch = p1 * patch_nx + p0;

            for (size_t i1 = 0; i1 < ny; ++i1) {
                for (size_t i0 = 0; i0 < nx; ++i0) {
                    if (0 == i0 || nx - 1 == i0 || 0 == i1 || ny - 1 == i1) {
                        u[patch][i0 + i1 * nx] = (tpe)0;
                        uNew[patch][i0 + i1 * nx] = (tpe)0;
                    } else {
                        u[patch][i0 + i1 * nx] = (tpe)1;
                        uNew[patch][i0 + i1 * nx] = (tpe)1;
                    }
                }
            }
        }
    }
}

template <typename tpe>
inline tpe checkSolutionStencil2D(const tpe *const __restrict__ *const __restrict__ u, const tpe *const __restrict__ *const __restrict__ uNew,
                                  const size_t nx, const size_t ny, const size_t nIt,
                                  const unsigned int patch_nx, const unsigned int patch_ny) {
    tpe res = 0;

    // skip halo and one additional layer to avoid necessity to communicate halos and setting boundary conditions
    // this will slightly pertubate the final residual when subdividing patches
    for (auto p1 = 0; p1 < patch_ny; ++p1) {
        for (auto p0 = 0; p0 < patch_nx; ++p0) {
            auto patch = p1 * patch_nx + p0;

            for (size_t i1 = 2; i1 < ny - 2; ++i1) {
                for (size_t i0 = 2; i0 < nx - 2; ++i0) {
                    const tpe localRes = 4 * u[patch][i0 + i1 * nx] - u[patch][i0 + i1 * nx + 1] - u[patch][i0 + i1 * nx - 1] - u[patch][i0 + nx * (i1 + 1)] - u[patch][i0 + nx * (i1 - 1)];
                    res += localRes * localRes;
                }
            }
        }
    }

    return res;
}
