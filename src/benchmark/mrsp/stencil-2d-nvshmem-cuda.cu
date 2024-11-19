#include "stencil-2d-util.h"

#include <mpi.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "../../cuda-util.h"


template <typename tpe>
__global__ void stencil2d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, const size_t nx, const size_t ny,
                          const int mpi_x, const int mpi_y, const int mpi_nx, const int mpi_ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 >= 1 && i0 < nx - 1 && i1 >= 1 && i1 < ny - 1) {
        tpe west, east, south, north;

        if (1 == i0) {
            if (0 == mpi_x)
                west = 2 * (tpe)0 - u[i0 + i1 * nx];
            else
                west = nvshmem_double_g(&u[i1 * nx + nx - 2], mpi_y * mpi_nx + mpi_x - 1);
        } else {
            west = u[i0 + i1 * nx - 1];
        }

        if (nx - 2 == i0) {
            if (mpi_nx - 1 == mpi_x)
                east = 2 * (tpe)0 - u[i0 + i1 * nx];
            else
                east = nvshmem_double_g(&u[i1 * nx + 1], mpi_y * mpi_nx + mpi_x + 1);
        } else {
            east = u[i0 + i1 * nx + 1];
        }

        if (1 == i1) {
            if (0 == mpi_y)
                south = 2 * (tpe)0 - u[i0 + i1 * nx];
            else
                south = nvshmem_double_g(&u[(ny - 2) * nx + i0], (mpi_y - 1) * mpi_nx + mpi_x);
        } else {
            south = u[i0 + nx * (i1 - 1)];
        }

        if (ny - 2 == i1) {
            if (mpi_ny - 1 == mpi_y)
                north = 2 * (tpe)0 - u[i0 + i1 * nx];
            else
                north = nvshmem_double_g(&u[nx + i0], (mpi_y + 1) * mpi_nx + mpi_x);
        } else {
            north = u[i0 + nx * (i1 + 1)];
        }

        auto res = 0.25 * (west + east + south + north);
        uNew[i0 + i1 * nx] = res;
    }
}

template <typename tpe>
inline void performIteration(tpe *&d_u, tpe *&d_uNew, const size_t nx, const size_t ny,
                             int mpi_rank, int mpi_x, int mpi_y, int mpi_nx, int mpi_ny) {

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x), ceilingDivide(ny, blockSize.y));

    stencil2d<<<numBlocks, blockSize>>>(d_u, d_uNew, nx, ny, mpi_x, mpi_y, mpi_nx, mpi_ny);
    nvshmem_barrier_all();
    std::swap(d_u, d_uNew);
}

template <typename tpe>
inline int realMain(int argc, char *argv[], MPI_Datatype MPI_TPE) {
    MPI_Init(&argc, &argv);
    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    unsigned int mpi_nx, mpi_ny;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt, mpi_nx, mpi_ny);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_size != mpi_nx * mpi_ny) {
        std::cerr << "Number of MPI processes must be " << mpi_nx * mpi_ny << std::endl;
        nvshmem_finalize();
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
    d_u = (tpe *)nvshmem_malloc(sizeof(tpe) * nx * ny);
    tpe *d_uNew;
    d_uNew = (tpe *)nvshmem_malloc(sizeof(tpe) * nx * ny);

    // init
    initStencil2D(u, uNew, nx, ny);

    checkCudaError(cudaMemcpy(d_u, u, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_uNew, uNew, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x), ceilingDivide(ny, blockSize.y));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i)
        performIteration(d_u, d_uNew, nx, ny, mpi_rank, mpi_x, mpi_y, mpi_nx, mpi_ny);
    checkCudaError(cudaDeviceSynchronize(), true);

    MPI_Barrier(MPI_COMM_WORLD);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i)
        performIteration(d_u, d_uNew, nx, ny, mpi_rank, mpi_x, mpi_y, mpi_nx, mpi_ny);
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

    nvshmem_free(d_u);
    nvshmem_free(d_uNew);

    checkCudaError(cudaFreeHost(u));
    checkCudaError(cudaFreeHost(uNew));

    nvshmem_finalize();
    MPI_Finalize();

    return 0;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Missing type specification" << std::endl;
        return -1;
    }

    std::string tpeName(argv[1]);

    // if ("float" == tpeName)
    //     return realMain<float>(argc, argv, MPI_FLOAT);
    if ("double" == tpeName)
        return realMain<double>(argc, argv, MPI_DOUBLE);

    std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;
    std::cout << "  int, long, float, double" << std::endl;
    return -1;
}
