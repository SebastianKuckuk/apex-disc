#include "stencil-2d-cuda-util.cuh"
#include "stencil-2d-util.h"

#include <mpi.h>

#include "../../cuda-util.h"


template <typename tpe>
__global__ void stencil2d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, const size_t nx, const size_t ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 >= 1 && i0 < nx - 1 && i1 >= 1 && i1 < ny - 1) {
        uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
    }
}

template <typename tpe>
__global__ void packAndApplyBC(tpe *__restrict__ u, tpe *__restrict__ sendBuf[4], const size_t nx, const size_t ny,
                               const int mpi_x, const int mpi_y, const int mpi_nx, const int mpi_ny) {
    size_t dir = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idxStart = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idxStride = blockDim.x * gridDim.x;

    auto buf = sendBuf[dir];

    switch (dir) {
    case 0:
        if (mpi_x > 0) {
            for (size_t y = idxStart; y < ny - 2; y += idxStride)
                buf[y] = u[(nx + 1) + y * nx];
        } else {
            for (size_t y = idxStart; y < ny - 2; y += idxStride)
                u[(y + 1) * nx] = 2 * (tpe)0 - u[(y + 1) * nx + 1];
        }
        break;

    case 1:
        if (mpi_x < mpi_nx - 1) {
            for (size_t y = idxStart; y < ny - 2; y += idxStride)
                buf[y] = u[(nx + nx - 2) + y * nx];
        } else {
            for (size_t y = idxStart; y < ny - 2; y += idxStride)
                u[(y + 1) * nx + nx - 1] = 2 * (tpe)0 - u[(y + 1) * nx + nx - 2];
        }
        break;

    case 2:
        if (mpi_y > 0) {
            for (size_t x = idxStart; x < nx - 2; x += idxStride)
                buf[x] = u[(nx + 1) + x];
        } else {
            for (size_t x = idxStart; x < nx - 2; x += idxStride)
                u[x + 1] = 2 * (tpe)0 - u[nx + x + 1];
        }
        break;

    case 3:
        if (mpi_y < mpi_ny - 1) {
            for (size_t x = idxStart; x < nx - 2; x += idxStride)
                buf[x] = u[(ny - 2) * nx + 1 + x];
        } else {
            for (size_t x = idxStart; x < nx - 2; x += idxStride)
                u[(ny - 1) * nx + x + 1] = 2 * (tpe)0 - u[(ny - 2) * nx + x + 1];
        }
        break;
    }
}

template <typename tpe>
__global__ void unpack(tpe *__restrict__ u, tpe *__restrict__ recvBuf[4], const size_t nx, const size_t ny,
                       const int mpi_x, const int mpi_y, const int mpi_nx, const int mpi_ny) {
    size_t dir = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idxStart = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idxStride = blockDim.x * gridDim.x;

    auto buf = recvBuf[dir];

    switch (dir) {
    case 0:
        if (mpi_x > 0)
            for (size_t y = idxStart; y < ny - 2; y += idxStride)
                u[nx + y * nx] = buf[y];
        break;

    case 1:
        if (mpi_x < mpi_nx - 1)
            for (size_t y = idxStart; y < ny - 2; y += idxStride)
                u[nx + nx - 1 + y * nx] = buf[y];
        break;

    case 2:
        if (mpi_y > 0)
            for (size_t x = idxStart; x < nx - 2; x += idxStride)
                u[1 + x] = buf[x];
        break;

    case 3:
        if (mpi_y < mpi_ny - 1)
            for (size_t x = idxStart; x < nx - 2; x += idxStride)
                u[(ny - 1) * nx + 1 + x] = buf[x];
        break;
    }
}

template <typename tpe>
inline void performIteration(tpe *&d_u, tpe *&d_uNew, tpe *d_d_bufSend[4], tpe *d_d_bufRecv[4], tpe *h_d_bufSend[4], tpe *h_d_bufRecv[4],
                             const size_t nx, const size_t ny,
                             int mpi_rank, int mpi_x, int mpi_y, int mpi_nx, int mpi_ny, MPI_Datatype MPI_TPE) {

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x), ceilingDivide(ny, blockSize.y));
    int blockSize1D = 32;

    MPI_Request reqs[] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
                          MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    packAndApplyBC<<<dim3(ceilingDivide(max(nx, ny) - 2, blockSize1D), 4), dim3(blockSize1D, 1)>>>(d_u, d_d_bufSend, nx, ny, mpi_x, mpi_y, mpi_nx, mpi_ny);

    checkCudaError(cudaDeviceSynchronize(), true);

    if (mpi_x > 0) {
        MPI_Isend(h_d_bufSend[0], ny - 2, MPI_TPE, mpi_rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(h_d_bufRecv[0], ny - 2, MPI_TPE, mpi_rank - 1, 0, MPI_COMM_WORLD, &reqs[4]);
    }
    if (mpi_x < mpi_nx - 1) {
        MPI_Isend(h_d_bufSend[1], ny - 2, MPI_TPE, mpi_rank + 1, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(h_d_bufRecv[1], ny - 2, MPI_TPE, mpi_rank + 1, 0, MPI_COMM_WORLD, &reqs[5]);
    }
    if (mpi_y > 0) {
        MPI_Isend(h_d_bufSend[2], nx - 2, MPI_TPE, mpi_rank - mpi_nx, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(h_d_bufRecv[2], nx - 2, MPI_TPE, mpi_rank - mpi_nx, 0, MPI_COMM_WORLD, &reqs[6]);
    }
    if (mpi_y < mpi_ny - 1) {
        MPI_Isend(h_d_bufSend[3], nx - 2, MPI_TPE, mpi_rank + mpi_nx, 0, MPI_COMM_WORLD, &reqs[3]);
        MPI_Irecv(h_d_bufRecv[3], nx - 2, MPI_TPE, mpi_rank + mpi_nx, 0, MPI_COMM_WORLD, &reqs[7]);
    }

    MPI_Waitall(4, &reqs[4], MPI_STATUSES_IGNORE);
    unpack<<<dim3(ceilingDivide(max(nx, ny) - 2, blockSize1D), 4), dim3(blockSize1D, 1)>>>(d_u, d_d_bufRecv, nx, ny, mpi_x, mpi_y, mpi_nx, mpi_ny);

    stencil2d<<<numBlocks, blockSize>>>(d_u, d_uNew, nx, ny);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    std::swap(d_u, d_uNew);
}

template <typename tpe>
inline int realMain(int argc, char *argv[], MPI_Datatype MPI_TPE) {
    MPI_Init(&argc, &argv);

    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    unsigned int mpi_nx, mpi_ny;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt, mpi_nx, mpi_ny);

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

    tpe *u;
    checkCudaError(cudaMallocHost((void **)&u, sizeof(tpe) * nx * ny));
    tpe *uNew;
    checkCudaError(cudaMallocHost((void **)&uNew, sizeof(tpe) * nx * ny));

    tpe *d_u;
    checkCudaError(cudaMalloc((void **)&d_u, sizeof(tpe) * nx * ny));
    tpe *d_uNew;
    checkCudaError(cudaMalloc((void **)&d_uNew, sizeof(tpe) * nx * ny));

    tpe *h_d_bufSend[4];
    tpe *h_d_bufRecv[4];
    for (auto i = 0; i < 4; ++i) {
        checkCudaError(cudaMalloc((void **)&h_d_bufSend[i], sizeof(tpe) * (i < 2 ? ny - 2 : nx - 2)));
        checkCudaError(cudaMalloc((void **)&h_d_bufRecv[i], sizeof(tpe) * (i < 2 ? ny - 2 : nx - 2)));
    }

    tpe **d_d_bufSend;
    tpe **d_d_bufRecv;
    checkCudaError(cudaMalloc((void **)&d_d_bufSend, sizeof(tpe *) * 4));
    checkCudaError(cudaMalloc((void **)&d_d_bufRecv, sizeof(tpe *) * 4));
    checkCudaError(cudaMemcpy(d_d_bufSend, h_d_bufSend, sizeof(tpe *) * 4, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_d_bufRecv, h_d_bufRecv, sizeof(tpe *) * 4, cudaMemcpyHostToDevice));

    // init
    initStencil2D(u, uNew, nx, ny);

    checkCudaError(cudaMemcpy(d_u, u, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_uNew, uNew, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i)
        performIteration(d_u, d_uNew, d_d_bufSend, d_d_bufRecv, h_d_bufSend, h_d_bufRecv, nx, ny, mpi_rank, mpi_x, mpi_y, mpi_nx, mpi_ny, MPI_TPE);
    checkCudaError(cudaDeviceSynchronize(), true);

    MPI_Barrier(MPI_COMM_WORLD);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt + 1; ++i)
        performIteration(d_u, d_uNew, d_d_bufSend, d_d_bufRecv, h_d_bufSend, h_d_bufRecv, nx, ny, mpi_rank, mpi_x, mpi_y, mpi_nx, mpi_ny, MPI_TPE);
    checkCudaError(cudaDeviceSynchronize(), true);

    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::steady_clock::now();

    if (0 == mpi_rank)
        printStats<tpe>(end - start, nIt, nx * ny, tpeName, mpi_size * (sizeof(tpe) + sizeof(tpe)), mpi_size * 7);

    checkCudaError(cudaMemcpy(u, d_u, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(uNew, d_uNew, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));

    // check solution
    tpe res = checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp);
    MPI_Reduce(0 == mpi_rank ? MPI_IN_PLACE : &res, &res, 1, MPI_TPE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (0 == mpi_rank) {
        res = sqrt(res);
        std::cout << "  Final residual is " << res << std::endl;
    }

    for (auto i = 0; i < 4; ++i) {
        checkCudaError(cudaFree(h_d_bufSend[i]));
        checkCudaError(cudaFree(h_d_bufRecv[i]));
    }
    checkCudaError(cudaFree(d_d_bufSend));
    checkCudaError(cudaFree(d_d_bufRecv));

    checkCudaError(cudaFree(d_u));
    checkCudaError(cudaFree(d_uNew));

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
