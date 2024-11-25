#include "stencil-2d-cuda-util.cuh"
#include "stencil-2d-util.h"

#include <mpi.h>

#include "../../cuda-util.h"


template <typename tpe>
__global__ void stencil2d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, const size_t nx, const size_t ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i0 < nx - 1 && i1 < ny - 1) {
        uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
    }
}

template <typename tpe>
inline void performIteration(tpe **&h_d_u, tpe **&h_d_uNew, tpe ***h_h_d_bufSend, tpe ***h_h_d_bufRecv, const size_t nx, const size_t ny,
                             int mpi_rank, int mpi_x, int mpi_y, int mpi_nx, int mpi_ny, MPI_Datatype MPI_TPE, MPI_Request *reqs,
                             unsigned int patch_nx, unsigned int patch_ny) {

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx - 2, blockSize.x), ceilingDivide(ny - 2, blockSize.y));
    int blockSize1D = 32;

    for (auto i = 0; i < 8 * patch_nx * patch_ny; ++i)
        reqs[i] = MPI_REQUEST_NULL;

    for (auto py = 0; py < patch_ny; ++py) {
        for (auto px = 0; px < patch_nx; ++px) {
            auto patch = py * patch_nx + px;

            if (0 == px && 0 == mpi_x)
                applyBCWest<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], nx, ny);
            else if (px > 0)
                exchangeWest<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], h_d_u[py * patch_nx + px - 1], nx, ny);
            else
                packBufferVertical<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], h_h_d_bufSend[patch][0], nx + 1, nx, ny);
            
            if (px == patch_nx - 1 && mpi_x == mpi_nx - 1)
                applyBCEast<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], nx, ny);
            else if (px < patch_nx - 1)
                exchangeEast<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], h_d_u[py * patch_nx + px + 1], nx, ny);
            else
                packBufferVertical<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], h_h_d_bufSend[patch][1], nx + nx - 2, nx, ny);

            if (0 == py && 0 == mpi_y)
                applyBCSouth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], nx, ny);
            else if (py > 0)
                exchangeSouth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], h_d_u[(py - 1) * patch_nx + px], nx, ny);
            else
                packBufferHorizontal<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], h_h_d_bufSend[patch][2], nx + 1, nx, ny);

            if (py == patch_ny - 1 && mpi_y == mpi_ny - 1)
                applyBCNorth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], nx, ny);
            else if (py < patch_ny - 1)
                exchangeNorth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], h_d_u[(py + 1) * patch_nx + px], nx, ny);
            else
                packBufferHorizontal<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(h_d_u[patch], h_h_d_bufSend[patch][3], (ny - 2) * nx + 1, nx, ny);
        }
    }

    checkCudaError(cudaDeviceSynchronize(), true);

    for (auto py = 0; py < patch_ny; ++py) {
        for (auto px = 0; px < patch_nx; ++px) {
            auto patch = py * patch_nx + px;

            if (0 == px && mpi_x > 0) {
                MPI_Isend(h_h_d_bufSend[patch][0], ny - 2, MPI_TPE, mpi_rank - 1, py, MPI_COMM_WORLD, &reqs[patch * 8 + 0]);
                MPI_Irecv(h_h_d_bufRecv[patch][0], ny - 2, MPI_TPE, mpi_rank - 1, py, MPI_COMM_WORLD, &reqs[patch * 8 + 4]);
            }
            if (px == patch_nx - 1 && mpi_x < mpi_nx - 1) {
                MPI_Isend(h_h_d_bufSend[patch][1], ny - 2, MPI_TPE, mpi_rank + 1, py, MPI_COMM_WORLD, &reqs[patch * 8 + 1]);
                MPI_Irecv(h_h_d_bufRecv[patch][1], ny - 2, MPI_TPE, mpi_rank + 1, py, MPI_COMM_WORLD, &reqs[patch * 8 + 5]);
            }
            if (0 == py && mpi_y > 0) {
                MPI_Isend(h_h_d_bufSend[patch][2], nx - 2, MPI_TPE, mpi_rank - mpi_nx, px, MPI_COMM_WORLD, &reqs[patch * 8 + 2]);
                MPI_Irecv(h_h_d_bufRecv[patch][2], nx - 2, MPI_TPE, mpi_rank - mpi_nx, px, MPI_COMM_WORLD, &reqs[patch * 8 + 6]);
            }
            if (py == patch_ny - 1 && mpi_y < mpi_ny - 1) {
                MPI_Isend(h_h_d_bufSend[patch][3], nx - 2, MPI_TPE, mpi_rank + mpi_nx, px, MPI_COMM_WORLD, &reqs[patch * 8 + 3]);
                MPI_Irecv(h_h_d_bufRecv[patch][3], nx - 2, MPI_TPE, mpi_rank + mpi_nx, px, MPI_COMM_WORLD, &reqs[patch * 8 + 7]);
            }
        }
    }

    for (auto py = 0; py < patch_ny; ++py) {
        for (auto px = 0; px < patch_nx; ++px) {
            auto patch = py * patch_nx + px;

            if (0 == px && mpi_x > 0) {
                MPI_Wait(&reqs[patch * 8 + 4], MPI_STATUS_IGNORE);
                unpackBufferVertical<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(h_h_d_bufRecv[patch][0], h_d_u[patch], nx, nx, ny);
            }
            if (px == patch_nx - 1 && mpi_x < mpi_nx - 1) {
                MPI_Wait(&reqs[patch * 8 + 5], MPI_STATUS_IGNORE);
                unpackBufferVertical<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(h_h_d_bufRecv[patch][1], h_d_u[patch], nx + nx - 1, nx, ny);
            }
            if (0 == py && mpi_y > 0) {
                MPI_Wait(&reqs[patch * 8 + 6], MPI_STATUS_IGNORE);
                unpackBufferHorizontal<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(h_h_d_bufRecv[patch][2], h_d_u[patch], 1, nx, ny);
            }
            if (py == patch_ny - 1 && mpi_y < mpi_ny - 1) {
                MPI_Wait(&reqs[patch * 8 + 7], MPI_STATUS_IGNORE);
                unpackBufferHorizontal<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(h_h_d_bufRecv[patch][3], h_d_u[patch], (ny - 1) * nx + 1, nx, ny);
            }
        }
    }

    for (auto py = 0; py < patch_ny; ++py)
        for (auto px = 0; px < patch_nx; ++px)
            stencil2d<<<numBlocks, blockSize>>>(h_d_u[py * patch_nx + px], h_d_uNew[py * patch_nx + px], nx, ny);

    // TODO: its not necessary to wait for all requests (inner patches, actual boundary patches, receives already done, ...)
    MPI_Waitall(8 * patch_nx * patch_ny, reqs, MPI_STATUSES_IGNORE);

    checkCudaError(cudaDeviceSynchronize(), true);

    std::swap(h_d_u, h_d_uNew);
}

template <typename tpe>
inline int realMain(int argc, char *argv[], MPI_Datatype MPI_TPE) {
    MPI_Init(&argc, &argv);

    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    unsigned int mpi_nx, mpi_ny, patch_nx, patch_ny;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt, mpi_nx, mpi_ny, patch_nx, patch_ny);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_size != mpi_nx * mpi_ny) {
        std::cerr << "Number of MPI processes must be " << mpi_nx * mpi_ny << std::endl;
        MPI_Finalize();
        return -1;
    }

    int mpi_x = mpi_rank % mpi_nx;
    int mpi_y = mpi_rank / mpi_nx;

    int numDevices;
    checkCudaError(cudaGetDeviceCount(&numDevices));
    int device = mpi_rank % numDevices;
    checkCudaError(cudaSetDevice(device));

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

    // TODO: the number of buffers could be reduced since only the boundary patches need to communicate
    tpe ***h_h_d_bufSend;
    tpe ***h_h_d_bufRecv;
    checkCudaError(cudaMallocHost((void **)&h_h_d_bufSend, sizeof(tpe *) * numPatches));
    checkCudaError(cudaMallocHost((void **)&h_h_d_bufRecv, sizeof(tpe *) * numPatches));
    for (auto p = 0; p < numPatches; ++p) {
        checkCudaError(cudaMallocHost((void **)&h_h_d_bufSend[p], sizeof(tpe *) * 4));
        checkCudaError(cudaMallocHost((void **)&h_h_d_bufRecv[p], sizeof(tpe *) * 4));
        for (auto i = 0; i < 4; ++i) {
            checkCudaError(cudaMalloc((void **)&h_h_d_bufSend[p][i], sizeof(tpe) * (i < 2 ? ny - 2 : nx - 2)));
            checkCudaError(cudaMalloc((void **)&h_h_d_bufRecv[p][i], sizeof(tpe) * (i < 2 ? ny - 2 : nx - 2)));
        }
    }

    // TODO: the number of requests could be reduced to 2 * patch_nx + 2 * patch_ny - 4 since only the boundary patches need to communicate
    MPI_Request *reqs = new MPI_Request[8 * numPatches];

    // init
    initStencil2D(u, uNew, nx, ny, patch_nx, patch_ny);

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMemcpy(h_d_u[i], u[i], sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(h_d_uNew[i], uNew[i], sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    }

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i)
        performIteration(h_d_u, h_d_uNew, h_h_d_bufSend, h_h_d_bufRecv, nx, ny, mpi_rank, mpi_x, mpi_y, mpi_nx, mpi_ny, MPI_TPE, reqs, patch_nx, patch_ny);
    checkCudaError(cudaDeviceSynchronize(), true);

    MPI_Barrier(MPI_COMM_WORLD);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i)
        performIteration(h_d_u, h_d_uNew, h_h_d_bufSend, h_h_d_bufRecv, nx, ny, mpi_rank, mpi_x, mpi_y, mpi_nx, mpi_ny, MPI_TPE, reqs, patch_nx, patch_ny);
    checkCudaError(cudaDeviceSynchronize(), true);

    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::steady_clock::now();

    if (0 == mpi_rank)
        printStats<tpe>(end - start, nIt, numPatches * mpi_size * (nx - 2) * (ny - 2), tpeName, sizeof(tpe) + sizeof(tpe), 7);

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMemcpy(u[i], h_d_u[i], sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(uNew[i], h_d_uNew[i], sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
    }

    // check solution
    tpe res = checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp, patch_nx, patch_ny);
    MPI_Reduce(0 == mpi_rank ? MPI_IN_PLACE : &res, &res, 1, MPI_TPE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (0 == mpi_rank) {
        res = sqrt(res);
        std::cout << "  Final residual is " << res << std::endl;
    }

    delete[] reqs;

    for (auto p = 0; p < numPatches; ++p) {
        for (auto i = 0; i < 4; ++i) {
            checkCudaError(cudaFree(h_h_d_bufSend[p][i]));
            checkCudaError(cudaFree(h_h_d_bufRecv[p][i]));
        }
        checkCudaError(cudaFreeHost(h_h_d_bufSend[p]));
        checkCudaError(cudaFreeHost(h_h_d_bufRecv[p]));
    }
    checkCudaError(cudaFreeHost(h_h_d_bufSend));
    checkCudaError(cudaFreeHost(h_h_d_bufRecv));

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

    MPI_Finalize();

    return 0;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Missing type specification" << std::endl;
        return -1;
    }

    std::string tpeName(argv[1]);

    if ("float" == tpeName)
        return realMain<float>(argc, argv, MPI_FLOAT);
    if ("double" == tpeName)
        return realMain<double>(argc, argv, MPI_DOUBLE);

    std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;
    std::cout << "  int, long, float, double" << std::endl;
    return -1;
}
