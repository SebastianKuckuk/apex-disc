#include "stencil-2d-util.h"


template <typename tpe>
inline void performIteration(tpe **&u, tpe **&uNew, const size_t nx, const size_t ny, int patch_nx, int patch_ny) {
    for (auto py = 0; py < patch_ny; ++py) {
        for (auto px = 0; px < patch_nx; ++px) {
            auto patch = px + py * patch_nx;

            // pre-resolution to allow data mapping
            auto uPatch = u[patch];
            auto uNewPatch = uNew[patch];
            auto uWest = 0 == px ? nullptr : u[py * patch_nx + px - 1];
            auto uEast = patch_nx - 1 == px ? nullptr : u[py * patch_nx + px + 1];
            auto uSouth = 0 == py ? nullptr : u[(py - 1) * patch_nx + px];
            auto uNorth = patch_ny - 1 == py ? nullptr : u[(py + 1) * patch_nx + px];

            #pragma omp target teams distribute parallel for collapse(2)
            for (size_t i1 = 1; i1 < ny - 1; ++i1) {
                for (size_t i0 = 1; i0 < nx - 1; ++i0) {
                    tpe west, east, south, north;

                    if (1 == i0) {
                        if (0 == px)
                            west = 2 * (tpe)0 - uPatch[i0 + i1 * nx];
                        else
                            west = uWest[nx - 2 + i1 * nx];
                    } else {
                        west = uPatch[i0 + i1 * nx - 1];
                    }

                    if (nx - 2 == i0) {
                        if (patch_nx - 1 == px)
                            east = 2 * (tpe)0 - uPatch[i0 + i1 * nx];
                        else
                            east = uEast[1 + i1 * nx];
                    } else {
                        east = uPatch[i0 + i1 * nx + 1];
                    }

                    if (1 == i1) {
                        if (0 == py)
                            south = 2 * (tpe)0 - uPatch[i0 + i1 * nx];
                        else
                            south = uSouth[i0 + (ny - 2) * nx];
                    } else {
                        south = uPatch[i0 + nx * (i1 - 1)];
                    }

                    if (ny - 2 == i1) {
                        if (patch_ny - 1 == py)
                            north = 2 * (tpe)0 - uPatch[i0 + i1 * nx];
                        else
                            north = uNorth[i0 + nx];
                    } else {
                        north = uPatch[i0 + nx * (i1 + 1)];
                    }

                    auto res = 0.25 * (west + east + south + north);
                    uNewPatch[i0 + i1 * nx] = res;
                }
            }
        }
    }

    std::swap(u, uNew);
}

template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    unsigned int patch_nx, patch_ny;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt, patch_nx, patch_ny);

    auto numPatches = patch_nx * patch_ny;

    tpe **u = new tpe *[numPatches];
    tpe **uNew = new tpe *[numPatches];
    for (auto i = 0; i < numPatches; ++i) {
        u[i] = new tpe[nx * ny];
        uNew[i] = new tpe[nx * ny];
    }

    // init
    initStencil2D(u, uNew, nx, ny, patch_nx, patch_ny);

    for (auto i = 0; i < numPatches; ++i) {
        #pragma omp target enter data map(to : u[i][0 : nx * ny], uNew[i][0 : nx * ny])
    }

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i)
        performIteration(u, uNew, nx, ny, patch_nx, patch_ny);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i)
        performIteration(u, uNew, nx, ny, patch_nx, patch_ny);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, numPatches * (nx - 2) * (ny - 2), tpeName, sizeof(tpe) + sizeof(tpe), 7);

    for (auto i = 0; i < numPatches; ++i) {
        #pragma omp target exit data map(from : u[i][0 : nx * ny], uNew[i][0 : nx * ny])
    }

    // check solution
    tpe res = checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp, patch_nx, patch_ny);
    res = sqrt(res);
    std::cout << "  Final residual is " << res << std::endl;

    for (auto i = 0; i < numPatches; ++i) {
        delete[] u[i];
        delete[] uNew[i];
    }

    delete[] u;
    delete[] uNew;

    return 0;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Missing type specification" << std::endl;
        return -1;
    }

    std::string tpeName(argv[1]);

    if ("float" == tpeName)
        return realMain<float>(argc, argv);
    if ("double" == tpeName)
        return realMain<double>(argc, argv);

    std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;
    std::cout << "  int, long, float, double" << std::endl;
    return -1;
}
