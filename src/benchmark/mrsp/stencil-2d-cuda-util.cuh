#pragma once

// packing kernels

template <typename tpe>
__global__ void packBufferHorizontal(const tpe *const __restrict__ u, tpe *__restrict__ buf, size_t baseIndex, const size_t nx, const size_t ny) {
    size_t xStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t x = xStart; x < nx - 2; x += blockDim.x * gridDim.x)
        buf[x] = u[baseIndex + x];
}

template <typename tpe>
__global__ void packBufferVertical(const tpe *const __restrict__ u, tpe *__restrict__ buf, size_t baseIndex, const size_t nx, const size_t ny) {
    size_t yStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t y = yStart; y < ny - 2; y += blockDim.x * gridDim.x)
        buf[y] = u[baseIndex + y * nx];
}

template <typename tpe>
__global__ void unpackBufferHorizontal(const tpe *const __restrict__ buf, tpe *__restrict__ u, size_t baseIndex, const size_t nx, const size_t ny) {
    size_t xStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t x = xStart; x < nx - 2; x += blockDim.x * gridDim.x)
        u[baseIndex + x] = buf[x];
}

template <typename tpe>
__global__ void unpackBufferVertical(const tpe *const __restrict__ buf, tpe *__restrict__ u, size_t baseIndex, const size_t nx, const size_t ny) {
    size_t yStart = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t y = yStart; y < ny - 2; y += blockDim.x * gridDim.x)
        u[baseIndex + y * nx] = buf[y];
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
