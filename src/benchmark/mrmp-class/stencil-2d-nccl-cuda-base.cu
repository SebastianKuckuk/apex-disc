#include "stencil-2d-cuda-util.cuh"
#include "stencil-2d-util.h"

#include <mpi.h>

#include "../../cuda-util.h"
#include "../../nccl-util.h"


template <typename tpe>
__global__ void stencil2d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, const size_t nx, const size_t ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i0 < nx - 1 && i1 < ny - 1) {
        uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
    }
}

template <typename tpe>
inline void performIteration(Patch<tpe> *patches, unsigned int numPatches, const size_t nx, const size_t ny,
                             int mpi_rank, int mpi_x, int mpi_y, int mpi_nx, int mpi_ny, ncclDataType_t nccl_tpe,
                             ncclComm_t *nccl_comms) {

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx - 2, blockSize.x), ceilingDivide(ny - 2, blockSize.y));
    int blockSize1D = 32;

    for (auto p = 0; p < numPatches; ++p) {
        auto& patch = patches[p];

        if (BOUNDARY == patch.neighborType[0])
            applyBCWest<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(patch.d_u, nx, ny);
        else if (LOCAL == patch.neighborType[0])
            exchangeWest<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(patch.d_u, patches[patch.neighborPatchIdx[0]].d_u, nx, ny);
        else
            packBufferVertical<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(patch.d_u, patch.h_d_bufSend[0], nx + 1, nx, ny);

        if (BOUNDARY == patch.neighborType[1])            
            applyBCEast<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(patch.d_u, nx, ny);
        else if (LOCAL == patch.neighborType[1])
            exchangeEast<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(patch.d_u, patches[patch.neighborPatchIdx[1]].d_u, nx, ny);
        else
            packBufferVertical<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(patch.d_u, patch.h_d_bufSend[1], nx + nx - 2, nx, ny);

        if (BOUNDARY == patch.neighborType[2])
            applyBCSouth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(patch.d_u, nx, ny);
        else if (LOCAL == patch.neighborType[2])
            exchangeSouth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(patch.d_u, patches[patch.neighborPatchIdx[2]].d_u, nx, ny);
        else
            packBufferHorizontal<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(patch.d_u, patch.h_d_bufSend[2], nx + 1, nx, ny);

        if (BOUNDARY == patch.neighborType[3])
            applyBCNorth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(patch.d_u, nx, ny);
        else if (LOCAL == patch.neighborType[3])
            exchangeNorth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(patch.d_u, patches[patch.neighborPatchIdx[3]].d_u, nx, ny);
        else
            packBufferHorizontal<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(patch.d_u, patch.h_d_bufSend[3], (ny - 2) * nx + 1, nx, ny);
    }

    // no tags in nccl - use different communicators to emulate tags
    checkNcclError(ncclGroupStart());
    {
        for (auto p = 0; p < numPatches; ++p) {
            auto& patch = patches[p];

            for (auto i = 0; i < 4; ++i) {
                if (REMOTE == patch.neighborType[i]) {
                    checkNcclError(ncclSend(patch.h_d_bufSend[i], i < 2 ? ny - 2 : nx - 2, nccl_tpe,
                                   patch.neighborMpiRank[i], nccl_comms[p], 0));
                    checkNcclError(ncclRecv(patch.h_d_bufRecv[i], i < 2 ? ny - 2 : nx - 2, nccl_tpe,
                                   patch.neighborMpiRank[i], nccl_comms[patch.neighborPatchIdx[i]], 0));
                }
            }
        }
    }    
    checkNcclError(ncclGroupEnd());

    for (auto p = 0; p < numPatches; ++p) {
        auto& patch = patches[p];

        if (REMOTE == patch.neighborType[0])
            unpackBufferVertical<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(patch.h_d_bufRecv[0], patch.d_u, nx, nx, ny);
        if (REMOTE == patch.neighborType[1])
            unpackBufferVertical<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D>>>(patch.h_d_bufRecv[1], patch.d_u, nx + nx - 1, nx, ny);
        if (REMOTE == patch.neighborType[2])
            unpackBufferHorizontal<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(patch.h_d_bufRecv[2], patch.d_u, 1, nx, ny);
        if (REMOTE == patch.neighborType[3])
            unpackBufferHorizontal<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D>>>(patch.h_d_bufRecv[3], patch.d_u, (ny - 1) * nx + 1, nx, ny);
    }

    for (auto p = 0; p < numPatches; ++p) {
        auto& patch = patches[p];
        stencil2d<<<numBlocks, blockSize>>>(patch.d_u, patch.d_uNew, nx, ny);
    }

    for (auto p = 0; p < numPatches; ++p) {
        auto& patch = patches[p];
        std::swap(patch.d_u, patch.d_uNew);
    }
}

template <typename tpe>
inline int realMain(int argc, char *argv[], MPI_Datatype MPI_TPE, ncclDataType_t nccl_tpe) {
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
    Patch<tpe> *patches;
    checkCudaError(cudaMallocHost((void **)&patches, sizeof(Patch<tpe>) * numPatches));

    for (auto py = 0; py < patch_ny; ++py) {
        for (auto px = 0; px < patch_nx; ++px) {
            auto p = py * patch_nx + px;
            auto& patch = patches[p];

            checkCudaError(cudaMallocHost((void **)&patch.u, sizeof(tpe) * nx * ny));
            checkCudaError(cudaMallocHost((void **)&patch.uNew, sizeof(tpe) * nx * ny));

            checkCudaError(cudaMalloc((void **)&patch.d_u, sizeof(tpe) * nx * ny));
            checkCudaError(cudaMalloc((void **)&patch.d_uNew, sizeof(tpe) * nx * ny));

            if (0 == px && 0 == mpi_x) {
                patch.neighborType[0] = BOUNDARY;
                patch.neighborPatchIdx[0] = -1;
                patch.neighborMpiRank[0] = -1;
            } else if (0 == px) {
                patch.neighborType[0] = REMOTE;
                patch.neighborPatchIdx[0] = py * patch_nx + (patch_nx - 1);
                patch.neighborMpiRank[0] = mpi_rank - 1;
            } else {
                patch.neighborType[0] = LOCAL;
                patch.neighborPatchIdx[0] = p - 1;
                patch.neighborMpiRank[0] = mpi_rank;
            }

            if (patch_nx - 1 == px && mpi_nx - 1 == mpi_x) {
                patch.neighborType[1] = BOUNDARY;
                patch.neighborPatchIdx[1] = -1;
                patch.neighborMpiRank[1] = -1;
            } else if (patch_nx - 1 == px) {
                patch.neighborType[1] = REMOTE;
                patch.neighborPatchIdx[1] = py * patch_nx;
                patch.neighborMpiRank[1] = mpi_rank + 1;
            } else {
                patch.neighborType[1] = LOCAL;
                patch.neighborPatchIdx[1] = p + 1;
                patch.neighborMpiRank[1] = mpi_rank;
            }

            if (0 == py && 0 == mpi_y) {
                patch.neighborType[2] = BOUNDARY;
                patch.neighborPatchIdx[2] = -1;
                patch.neighborMpiRank[2] = -1;
            } else if (0 == py) {
                patch.neighborType[2] = REMOTE;
                patch.neighborPatchIdx[2] = (patch_ny - 1) * patch_nx + px;
                patch.neighborMpiRank[2] = mpi_rank - mpi_nx;
            } else {
                patch.neighborType[2] = LOCAL;
                patch.neighborPatchIdx[2] = p - patch_nx;
                patch.neighborMpiRank[2] = mpi_rank;
            }

            if (patch_ny - 1 == py && mpi_ny - 1 == mpi_y) {
                patch.neighborType[3] = BOUNDARY;
                patch.neighborPatchIdx[3] = -1;
                patch.neighborMpiRank[3] = -1;
            } else if (patch_ny - 1 == py) {
                patch.neighborType[3] = REMOTE;
                patch.neighborPatchIdx[3] = px;
                patch.neighborMpiRank[3] = mpi_rank + mpi_nx;
            } else {
                patch.neighborType[3] = LOCAL;
                patch.neighborPatchIdx[3] = p + patch_nx;
                patch.neighborMpiRank[3] = mpi_rank;
            }

            for (auto i = 0; i < 4; ++i) {
                if (REMOTE == patch.neighborType[i]) {
                    checkCudaError(cudaMalloc((void **)&patch.h_d_bufSend[i], sizeof(tpe) * (i < 2 ? ny - 2 : nx - 2)));
                    checkCudaError(cudaMalloc((void **)&patch.h_d_bufRecv[i], sizeof(tpe) * (i < 2 ? ny - 2 : nx - 2)));
                } else {
                    patch.h_d_bufSend[i] = nullptr;
                    patch.h_d_bufRecv[i] = nullptr;
                    patch.reqsSend[i] = MPI_REQUEST_NULL;
                    patch.reqsRecv[i] = MPI_REQUEST_NULL;
                }
            }

            checkCudaError(cudaMalloc((void **)&patch.d_d_bufSend, sizeof(tpe *) * 4));
            checkCudaError(cudaMalloc((void **)&patch.d_d_bufRecv, sizeof(tpe *) * 4));

            checkCudaError(cudaMemcpy(patch.d_d_bufSend, patch.h_d_bufSend, sizeof(tpe *) * 4, cudaMemcpyHostToDevice));
            checkCudaError(cudaMemcpy(patch.d_d_bufRecv, patch.h_d_bufRecv, sizeof(tpe *) * 4, cudaMemcpyHostToDevice));
        }
    }

    Patch<tpe> *d_patches;
    checkCudaError(cudaMalloc((void **)&d_patches, sizeof(Patch<tpe>) * numPatches));
    checkCudaError(cudaMemcpy(d_patches, patches, sizeof(Patch<tpe>) * numPatches, cudaMemcpyHostToDevice));

    ncclComm_t *nccl_comms;
    checkCudaError(cudaMallocHost((void **)&nccl_comms, sizeof(ncclComm_t) * numPatches));
    for (auto p = 0; p < numPatches; ++p) {
        ncclUniqueId nccl_uid;
        if (0 == mpi_rank)
            checkNcclError(ncclGetUniqueId(&nccl_uid));
        MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        checkNcclError(ncclCommInitRank(&nccl_comms[p], mpi_size, nccl_uid, mpi_rank));
    }

    if (0 == mpi_rank) {
        int nccl_version = 0;
        checkNcclError(ncclGetVersion(&nccl_version));
    }

    // init
    initStencil2D(patches, nx, ny, numPatches);

    for (auto p = 0; p < numPatches; ++p) {
        auto& patch = patches[p];
        checkCudaError(cudaMemcpy(patch.d_u, patch.u, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(patch.d_uNew, patch.uNew, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    }

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i)
        performIteration(patches, numPatches, nx, ny, mpi_rank, mpi_x, mpi_y, mpi_nx, mpi_ny, nccl_tpe, nccl_comms);
    checkCudaError(cudaDeviceSynchronize(), true);

    MPI_Barrier(MPI_COMM_WORLD);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i)
        performIteration(patches, numPatches, nx, ny, mpi_rank, mpi_x, mpi_y, mpi_nx, mpi_ny, nccl_tpe, nccl_comms);
    checkCudaError(cudaDeviceSynchronize(), true);

    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::steady_clock::now();

    if (0 == mpi_rank)
        printStats<tpe>(end - start, nIt, numPatches * mpi_size * (nx - 2) * (ny - 2), tpeName, sizeof(tpe) + sizeof(tpe), 7);

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMemcpy(patches[i].u, patches[i].d_u, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(patches[i].uNew, patches[i].d_uNew, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
    }

    // check solution
    tpe res = checkSolutionStencil2D(patches, nx, ny, nIt + nItWarmUp, numPatches);
    MPI_Reduce(0 == mpi_rank ? MPI_IN_PLACE : &res, &res, 1, MPI_TPE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (0 == mpi_rank) {
        res = sqrt(res);
        std::cout << "  Final residual is " << res << std::endl;
    }

    for (auto p = 0; p < numPatches; ++p) {
        auto& patch = patches[p];

        for (auto i = 0; i < 4; ++i) {
            if (REMOTE == patch.neighborType[i]) {
                checkCudaError(cudaFree(patch.h_d_bufSend[i]));
                checkCudaError(cudaFree(patch.h_d_bufRecv[i]));
            }
        }

        checkCudaError(cudaFree(patch.d_d_bufSend));
        checkCudaError(cudaFree(patch.d_d_bufRecv));

        checkCudaError(cudaFree(patch.d_u));
        checkCudaError(cudaFree(patch.d_uNew));

        checkCudaError(cudaFreeHost(patch.u));
        checkCudaError(cudaFreeHost(patch.uNew));
    }

    checkCudaError(cudaFree(d_patches));
    checkCudaError(cudaFreeHost(patches));

    for (auto p = 0; p < numPatches; ++p)
        checkNcclError(ncclCommDestroy(nccl_comms[p]));
    checkCudaError(cudaFreeHost(nccl_comms));

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
        return realMain<float>(argc, argv, MPI_FLOAT, ncclFloat);
    if ("double" == tpeName)
        return realMain<double>(argc, argv, MPI_DOUBLE, ncclDouble);

    std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;
    std::cout << "  int, long, float, double" << std::endl;
    return -1;
}
