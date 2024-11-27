#include "stencil-2d-cuda-util.cuh"
#include "stencil-2d-util.h"

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
inline void performIteration(tpe **&h_d_u, tpe **&h_d_uNew, const size_t nx, const size_t ny,
                             int patch_nx, int patch_ny,
                             cudaStream_t *streams, cudaGraph_t &graph, cudaGraphExec_t &graphExec) {

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx - 2, blockSize.x), ceilingDivide(ny - 2, blockSize.y));
    int blockSize1D = 32;

    static bool firstTime = true;
    if (firstTime) {
        firstTime = false;

        cudaEvent_t branchEvent;
        checkCudaError(cudaEventCreate(&branchEvent, cudaEventDisableTiming));

        cudaEvent_t *events;
        checkCudaError(cudaMallocHost((void **)&events, sizeof(cudaEvent_t) * 5 * patch_nx * patch_ny));
        for (auto i = 0; i < 5 * patch_nx * patch_ny; ++i)
            checkCudaError(cudaEventCreate(&events[i], cudaEventDisableTiming));

        checkCudaError(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));

        checkCudaError(cudaEventRecord(branchEvent, streams[0]));

        for (auto py = 0; py < patch_ny; ++py) {
            for (auto px = 0; px < patch_nx; ++px) {
                auto patch = py * patch_nx + px;

                for (auto i = 0; i < 4; ++i)
                    checkCudaError(cudaStreamWaitEvent(streams[5 * patch + i], branchEvent, 0));

                if (px > 0)
                    exchangeWest<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D, 0, streams[5 * patch + 0]>>>(h_d_u[patch], h_d_u[py * patch_nx + px - 1], nx, ny);
                else
                    applyBCWest<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D, 0, streams[5 * patch + 0]>>>(h_d_u[patch], nx, ny);

                if (px < patch_nx - 1)
                    exchangeEast<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D, 0, streams[5 * patch + 1]>>>(h_d_u[patch], h_d_u[py * patch_nx + px + 1], nx, ny);
                else
                    applyBCEast<<<ceilingDivide(ny - 2, blockSize1D), blockSize1D, 0, streams[5 * patch + 1]>>>(h_d_u[patch], nx, ny);

                if (py > 0)
                    exchangeSouth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D, 0, streams[5 * patch + 2]>>>(h_d_u[patch], h_d_u[(py - 1) * patch_nx + px], nx, ny);
                else
                    applyBCSouth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D, 0, streams[5 * patch + 2]>>>(h_d_u[patch], nx, ny);

                if (py < patch_ny - 1)
                    exchangeNorth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D, 0, streams[5 * patch + 3]>>>(h_d_u[patch], h_d_u[(py + 1) * patch_nx + px], nx, ny);
                else
                    applyBCNorth<<<ceilingDivide(nx - 2, blockSize1D), blockSize1D, 0, streams[5 * patch + 3]>>>(h_d_u[patch], nx, ny);

                for (auto i = 0; i < 4; ++i)
                    checkCudaError(cudaEventRecord(events[5 * patch + i], streams[5 * patch + i]));
            }
        }

        for (auto py = 0; py < patch_ny; ++py) {
            for (auto px = 0; px < patch_nx; ++px) {
                auto patch = px + py * patch_nx;

                // wait for operations on the same patch
                for (auto i = 0; i < 4; ++i)
                    checkCudaError(cudaStreamWaitEvent(streams[5 * patch + 4], events[5 * patch + i], 0));

                // wait for operations from neighbor patches
                if (px > 0)
                    checkCudaError(cudaStreamWaitEvent(streams[5 * patch + 4], events[5 * (py * patch_nx + px - 1) + 0], 0));
                if (px < patch_nx - 1)
                    checkCudaError(cudaStreamWaitEvent(streams[5 * patch + 4], events[5 * (py * patch_nx + px + 1) + 1], 0));
                if (py > 0)
                    checkCudaError(cudaStreamWaitEvent(streams[5 * patch + 4], events[5 * ((py - 1) * patch_nx + px) + 2], 0));
                if (py < patch_ny - 1)
                    checkCudaError(cudaStreamWaitEvent(streams[5 * patch + 4], events[5 * ((py + 1) * patch_nx + px) + 3], 0));
                
                // finally perform the computation
                stencil2d<<<numBlocks, blockSize, 0, streams[5 * patch + 4]>>>(h_d_u[patch], h_d_uNew[patch], nx, ny);

                checkCudaError(cudaEventRecord(events[5 * patch + 4], streams[5 * patch + 4]));
            }
        }

        // merge back into capture stream
        for (auto p = 0; p < patch_nx * patch_ny; ++p)
            checkCudaError(cudaStreamWaitEvent(streams[0], events[5 * p + 4], 0));
        checkCudaError(cudaStreamWaitEvent(streams[0], events[5 * 0 + 4], 0));

        checkCudaError(cudaStreamEndCapture(streams[0], &graph));
        checkCudaError(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

        for (auto i = 0; i < 5 * patch_nx * patch_ny; ++i)
            checkCudaError(cudaEventDestroy(events[i]));
        checkCudaError(cudaFreeHost(events));

        checkCudaError(cudaEventDestroy(branchEvent));
    }

    checkCudaError(cudaGraphLaunch(graphExec, streams[0]));
    checkCudaError(cudaStreamSynchronize(streams[0]));

    // TODO: swap won't work like this
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

    cudaStream_t *streams;
    checkCudaError(cudaMallocHost((void **)&streams, sizeof(cudaStream_t) * 5 * numPatches));
    for (auto i = 0; i < 5 * numPatches; ++i)
        checkCudaError(cudaStreamCreate(&streams[i]));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // init
    initStencil2D(u, uNew, nx, ny, patch_nx, patch_ny);

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMemcpy(h_d_u[i], u[i], sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(h_d_uNew[i], uNew[i], sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    }

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i)
        performIteration(h_d_u, h_d_uNew, nx, ny, patch_nx, patch_ny, streams, graph, graphExec);
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i)
        performIteration(h_d_u, h_d_uNew, nx, ny, patch_nx, patch_ny, streams, graph, graphExec);
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, numPatches * (nx - 2) * (ny - 2), tpeName, sizeof(tpe) + sizeof(tpe), 7);

    for (auto i = 0; i < numPatches; ++i) {
        checkCudaError(cudaMemcpy(u[i], h_d_u[i], sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(uNew[i], h_d_uNew[i], sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
    }

    // check solution
    tpe res = checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp, patch_nx, patch_ny);
    res = sqrt(res);
    std::cout << "  Final residual is " << res << std::endl;

    checkCudaError(cudaGraphExecDestroy(graphExec));
    checkCudaError(cudaGraphDestroy(graph));

    for (auto i = 0; i < 5 * numPatches; ++i)
        checkCudaError(cudaStreamDestroy(streams[i]));
    checkCudaError(cudaFreeHost(streams));

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
