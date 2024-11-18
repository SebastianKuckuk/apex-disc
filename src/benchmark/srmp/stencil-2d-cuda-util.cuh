// exchange kernels

template <typename tpe>
__global__ void exchangeWest(tpe *__restrict__ u, tpe *__restrict__ uNeigh, const size_t nx, const size_t ny) {
    size_t yStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t y = yStart; y < ny - 2; y += blockDim.x * gridDim.x)
        uNeigh[(nx + nx - 1) + y * nx] = u[(nx + 1) + y * nx];
}

template <typename tpe>
__global__ void exchangeEast(tpe *__restrict__ u, tpe *__restrict__ uNeigh, const size_t nx, const size_t ny) {
    size_t yStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t y = yStart; y < ny - 2; y += blockDim.x * gridDim.x)
        uNeigh[nx + y * nx] = u[(nx + nx - 2) + y * nx];
}

template <typename tpe>
__global__ void exchangeSouth(tpe *__restrict__ u, tpe *__restrict__ uNeigh, const size_t nx, const size_t ny) {
    size_t xStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t x = xStart; x < nx - 2; x += blockDim.x * gridDim.x)
        uNeigh[(ny - 1) * nx + 1 + x] = u[(nx + 1) + x];
}

template <typename tpe>
__global__ void exchangeNorth(tpe *__restrict__ u, tpe *__restrict__ uNeigh, const size_t nx, const size_t ny) {
    size_t xStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t x = xStart; x < nx - 2; x += blockDim.x * gridDim.x)
        uNeigh[x + 1] = u[(ny - 2) * nx + 1 + x];
}

// boundary condition kernels

template <typename tpe>
__global__ void applyBCWest(tpe *__restrict__ u, const size_t nx, const size_t ny) {
    size_t yStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t y = yStart; y < ny - 2; y += blockDim.x * gridDim.x)
        u[(y + 1) * nx] = 2 * (tpe)0 - u[(y + 1) * nx + 1];
}

template <typename tpe>
__global__ void applyBCEast(tpe *__restrict__ u, const size_t nx, const size_t ny) {
    size_t yStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t y = yStart; y < ny - 2; y += blockDim.x * gridDim.x)
        u[(y + 1) * nx + nx - 1] = 2 * (tpe)0 - u[(y + 1) * nx + nx - 2];
}

template <typename tpe>
__global__ void applyBCSouth(tpe *__restrict__ u, const size_t nx, const size_t ny) {
    size_t xStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t x = xStart; x < nx - 2; x += blockDim.x * gridDim.x)
        u[x + 1] = 2 * (tpe)0 - u[nx + x + 1];
}

template <typename tpe>
__global__ void applyBCNorth(tpe *__restrict__ u, const size_t nx, const size_t ny) {
    size_t xStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t x = xStart; x < nx - 2; x += blockDim.x * gridDim.x)
        u[(ny - 1) * nx + x + 1] = 2 * (tpe)0 - u[(ny - 2) * nx + x + 1];
}
