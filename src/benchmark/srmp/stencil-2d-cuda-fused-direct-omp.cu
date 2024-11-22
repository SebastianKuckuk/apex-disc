#include "stencil-2d-cuda-util.cuh"
#include "stencil-2d-util.h"

#include "../../cuda-util.h"


template <typename tpe>
__global__ void stencil2d(const tpe *const __restrict__ *const __restrict__ u, tpe *__restrict__ *__restrict__ uNew, const size_t nx, const size_t ny,
                          const int px, const int py, const int patch_nx, const int patch_ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y + 1;

    auto patch = px + py * patch_nx;
    auto uPatch = u[patch];

    if (i0 < nx - 1 && i1 < ny - 1) {
        tpe west, east, south, north;

        if (1 == i0) {
            if (0 == px)
                west = 2 * (tpe)0 - uPatch[i0 + i1 * nx];
            else
                west = u[py * patch_nx + px - 1][nx - 2 + i1 * nx];
        } else {
            west = uPatch[i0 + i1 * nx - 1];
        }

        if (nx - 2 == i0) {
            if (patch_nx - 1 == px)
                east = 2 * (tpe)0 - uPatch[i0 + i1 * nx];
            else
                east = u[py * patch_nx + px + 1][1 + i1 * nx];
        } else {
            east = uPatch[i0 + i1 * nx + 1];
        }

        if (1 == i1) {
            if (0 == py)
                south = 2 * (tpe)0 - uPatch[i0 + i1 * nx];
            else
                south = u[(py - 1) * patch_nx + px][i0 + (ny - 2) * nx];
        } else {
            south = uPatch[i0 + nx * (i1 - 1)];
        }

        if (ny - 2 == i1) {
            if (patch_ny - 1 == py)
                north = 2 * (tpe)0 - uPatch[i0 + i1 * nx];
            else
                north = u[(py + 1) * patch_nx + px][i0 + nx];
        } else {
            north = uPatch[i0 + nx * (i1 + 1)];
        }

        auto res = 0.25 * (west + east + south + north);
        uNew[patch][i0 + i1 * nx] = res;
    }
}

template <typename tpe>
inline void performIteration(tpe **&h_d_u, tpe **&h_d_uNew, const size_t nx, const size_t ny,
                             int patch_nx, int patch_ny) {

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx - 2, blockSize.x), ceilingDivide(ny - 2, blockSize.y));

    #pragma omp parallel for collapse(2) schedule(static) num_threads(patch_nx * patch_ny)
    for (auto py = 0; py < patch_ny; ++py)
        for (auto px = 0; px < patch_nx; ++px)
            stencil2d<<<numBlocks, blockSize>>>(h_d_u, h_d_uNew, nx, ny, px, py, patch_nx, patch_ny);

    checkCudaError(cudaDeviceSynchronize(), true);

    std::swap(h_d_u, h_d_uNew);
}

template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    unsigned int patch_nx, patch_ny;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt, patch_nx, patch_ny);

    auto numPatches = patch_nx * patch_ny;

    tpe **u;
    checkCudaError(cudaMallocHost((void **)&u, sizeof(tpe *) * numPatches));
    tpe **uNew;
    checkCudaError(cudaMallocHost((void **)&uNew, sizeof(tpe *) * numPatches));
    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMallocHost((void **)&u[i], sizeof(tpe) * nx * ny));
        checkCudaError(cudaMallocHost((void **)&uNew[i], sizeof(tpe) * nx * ny));
    }

    tpe **h_d_u;
    tpe **h_d_uNew;
    checkCudaError(cudaMallocHost((void **)&h_d_u, sizeof(tpe *) * numPatches));
    checkCudaError(cudaMallocHost((void **)&h_d_uNew, sizeof(tpe *) * numPatches));
    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMalloc((void **)&h_d_u[i], sizeof(tpe) * nx * ny));
        checkCudaError(cudaMalloc((void **)&h_d_uNew[i], sizeof(tpe) * nx * ny));
    }

    tpe **d_d_u;
    tpe **d_d_uNew;
    checkCudaError(cudaMalloc((void **)&d_d_u, sizeof(tpe *) * numPatches));
    checkCudaError(cudaMalloc((void **)&d_d_uNew, sizeof(tpe *) * numPatches));
    checkCudaError(cudaMemcpy(d_d_u, h_d_u, sizeof(tpe *) * numPatches, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_d_uNew, h_d_uNew, sizeof(tpe *) * numPatches, cudaMemcpyHostToDevice));

    // init
    initStencil2D(u, uNew, nx, ny, patch_nx, patch_ny);

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMemcpy(h_d_u[i], u[i], sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(h_d_uNew[i], uNew[i], sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    }

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i)
        performIteration(h_d_u, h_d_uNew, nx, ny, patch_nx, patch_ny);
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i)
        performIteration(h_d_u, h_d_uNew, nx, ny, patch_nx, patch_ny);
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, numPatches * nx * ny, tpeName, sizeof(tpe) + sizeof(tpe), 7);

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMemcpy(u[i], h_d_u[i], sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(uNew[i], h_d_uNew[i], sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
    }

    // check solution
    tpe res = checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp, patch_nx, patch_ny);
    res = sqrt(res);
    std::cout << "  Final residual is " << res << std::endl;

    checkCudaError(cudaFree(d_d_u));
    checkCudaError(cudaFree(d_d_uNew));

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaFree(h_d_u[i]));
        checkCudaError(cudaFree(h_d_uNew[i]));
    }

    checkCudaError(cudaFreeHost(h_d_u));
    checkCudaError(cudaFreeHost(h_d_uNew));

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaFreeHost(u[i]));
        checkCudaError(cudaFreeHost(uNew[i]));
    }

    checkCudaError(cudaFreeHost(u));
    checkCudaError(cudaFreeHost(uNew));

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
