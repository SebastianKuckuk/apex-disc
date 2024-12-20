#pragma once

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>


#ifdef __NVCC__
#   define FCT_DECORATOR __host__ __device__
#else
#   define FCT_DECORATOR
#endif


void parseCLA_1d(int argc, char* const* argv, char*& tpeName, size_t& nx, size_t& nItWarmUp, size_t& nIt,
                 unsigned int& mpi_nx) {
    // default values
    nx = 1024 * 1024;
    nItWarmUp = 2;
    nIt = 10;
    mpi_nx = 1;

    // override with command line arguments
    int i = 1;
    if (argc > i) tpeName = argv[i];
    ++i;
    if (argc > i) nx = atoi(argv[i]);
    ++i;
    if (argc > i) nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i) nIt = atoi(argv[i]);
    ++i;
    if (argc > i) mpi_nx = atoi(argv[i]);
    ++i;
}

void parseCLA_2d(int argc, char* const* argv, char*& tpeName, size_t& nx, size_t& ny, size_t& nItWarmUp, size_t& nIt,
                 unsigned int& mpi_nx, unsigned int& mpi_ny) {
    // default values
    nx = 1024;
    ny = nx;
    nItWarmUp = 2;
    nIt = 10;
    mpi_nx = 1;
    mpi_ny = 1;

    // override with command line arguments
    int i = 1;
    if (argc > i) tpeName = argv[i];
    ++i;
    if (argc > i) nx = atoi(argv[i]);
    ++i;
    if (argc > i) ny = atoi(argv[i]);
    ++i;
    if (argc > i) nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i) nIt = atoi(argv[i]);
    ++i;
    if (argc > i) mpi_nx = atoi(argv[i]);
    ++i;
    if (argc > i) mpi_ny = atoi(argv[i]);
    ++i;
}

void parseCLA_2d(int argc, char* const* argv, char*& tpeName, size_t& nx, size_t& ny, size_t& nItWarmUp, size_t& nIt,
                 unsigned int& mpi_nx, unsigned int& mpi_ny, unsigned int& patch_nx, unsigned int& patch_ny) {
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt, mpi_nx, mpi_ny);
    
    // default values
    patch_nx = 1;
    patch_ny = 1;

    // override with command line arguments
    int i = 8;
    if (argc > i) patch_nx = atoi(argv[i]);
    ++i;
    if (argc > i) patch_ny = atoi(argv[i]);
    ++i;
}

void parseCLA_3d(int argc, char* const* argv, char*& tpeName, size_t& nx, size_t& ny, size_t& nz, size_t& nItWarmUp, size_t& nIt,
                 unsigned int& mpi_nx, unsigned int& mpi_ny, unsigned int& mpi_nz) {
    // default values
    nx = 128;
    ny = nx;
    nz = ny;
    nItWarmUp = 2;
    nIt = 10;
    mpi_nx = 1;
    mpi_ny = 1;
    mpi_nz = 1;

    // override with command line arguments
    int i = 1;
    if (argc > i) tpeName = argv[i];
    ++i;
    if (argc > i) nx = atoi(argv[i]);
    ++i;
    if (argc > i) ny = atoi(argv[i]);
    ++i;
    if (argc > i) nz = atoi(argv[i]);
    ++i;
    if (argc > i) nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i) nIt = atoi(argv[i]);
    ++i;
    if (argc > i) mpi_nx = atoi(argv[i]);
    ++i;
    if (argc > i) mpi_ny = atoi(argv[i]);
    ++i;
    if (argc > i) mpi_nz = atoi(argv[i]);
    ++i;
}

template<typename tpe>
void printStats(const std::chrono::duration<double> elapsedSeconds, size_t nIt, size_t nCells, char* tpeName, size_t numBytesPerCell, size_t numFlopsPerCell) {
    std::cout << "  #cells / #it:  " << nCells << " / " << nIt << "\n";
    std::cout << "  type:          " << tpeName << "\n";
    std::cout << "  elapsed time:  " << 1e3 * elapsedSeconds.count() << " ms\n";
    std::cout << "  per iteration: " << 1e3 * elapsedSeconds.count() / nIt << " ms\n";
    std::cout << "  MLUP/s:        " << 1e-6 * nCells * nIt / elapsedSeconds.count() << "\n";
    std::cout << "  bandwidth:     " << 1e-9 * numBytesPerCell * nCells * nIt / elapsedSeconds.count() << " GB/s\n";
    std::cout << "  compute:       " << 1e-9 * numFlopsPerCell * nCells * nIt / elapsedSeconds.count() << " GFLOP/s\n";
}

FCT_DECORATOR size_t ceilingDivide(size_t a, size_t b) {
    return (a + b - 1) / b;
}

FCT_DECORATOR size_t ceilToMultipleOf(size_t a, size_t b) {
    return ceilingDivide(a, b) * b;
}
