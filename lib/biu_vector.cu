/*================================================================
*  Copyright (C)2019 All rights reserved.
*  FileName : biu_vector.cu
*  Author   : ElonKou
*  Email    : elonkou@ktime.cc
*  Date     : 2019年08月17日 星期六 17时43分21秒
================================================================*/

#include "biu.hh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PREFIX __global__ void
#define VECTOR_ADD(TYPE)                                                       \
    __global__ void vector_add_##TYPE(TYPE *A, TYPE *B, TYPE *C, int N) {      \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
        if (idx < N) {                                                         \
            C[idx] = A[idx] + B[idx];                                          \
        }                                                                      \
    }                                                                          \
    extern "C" void add_##TYPE(TYPE *A, TYPE *B, TYPE *C, int N) {             \
        int nBytes = N * sizeof(TYPE);                                         \
        int *cuda_A, *cuda_B, *cuda_C;                                         \
        cudaMalloc((void **)&cuda_A, nBytes);                                  \
        cudaMalloc((void **)&cuda_B, nBytes);                                  \
        cudaMalloc((void **)&cuda_C, nBytes);                                  \
        cudaMemcpy((void *)cuda_A, (void *)A, nBytes, cudaMemcpyHostToDevice); \
        cudaMemcpy((void *)cuda_B, (void *)B, nBytes, cudaMemcpyHostToDevice); \
        dim3 blockSize(CUDA_W, CUDA_H);                                        \
        dim3 gridSize((N + CUDA_W - 1) / CUDA_W, (N + CUDA_H - 1) / CUDA_H);   \
        vector_add_int<<<gridSize, blockSize>>>(cuda_A, cuda_B, cuda_C, N);    \
        cudaMemcpy((void *)C, (void *)cuda_C, nBytes, cudaMemcpyDeviceToHost); \
        cudaFree(cuda_A);                                                      \
        cudaFree(cuda_B);                                                      \
        cudaFree(cuda_C);                                                      \
    }

#define VECTOR_ADD_CONST(TYPE_A, TYPE_B)                                       \
    __global__ void vector_add_##TYPE_A##_##TYPE_B(TYPE_A *A, TYPE_B C,        \
                                                   int N) {                    \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
        if (idx < N) {                                                         \
            A[idx] += C;                                                       \
        }                                                                      \
    }                                                                          \
    extern "C" void add_const_##TYPE_A##_##TYPE_B(TYPE_A *A, TYPE_B C,         \
                                                  int N) {                     \
        int nBytes = N * sizeof(TYPE_A);                                       \
        TYPE_A *cuda_A;                                                        \
        cudaMalloc((void **)&cuda_A, nBytes);                                  \
        cudaMemcpy((void *)cuda_A, (void *)A, nBytes, cudaMemcpyHostToDevice); \
        dim3 blockSize(CUDA_W, CUDA_H);                                        \
        dim3 gridSize((N + CUDA_W - 1) / CUDA_W, (N + CUDA_H - 1) / CUDA_H);   \
        vector_add_##TYPE_A##_##TYPE_B<<<gridSize, blockSize>>>(cuda_A, C, N); \
        cudaMemcpy((void *)A, (void *)cuda_A, nBytes, cudaMemcpyDeviceToHost); \
        cudaFree(cuda_A);                                                      \
    }

VECTOR_ADD(int)
VECTOR_ADD(float)
VECTOR_ADD(double)
VECTOR_ADD_CONST(int, int)
VECTOR_ADD_CONST(float, float)
VECTOR_ADD_CONST(double, double)