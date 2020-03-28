/*================================================================
*  Copyright (C)2020 All rights reserved.
*  FileName : 00.add.cu
*  Author   : maxsense
*  Email    : elonkou@ktime.cc
*  Date     : 2020年03月18日 星期三 10时48分48秒
================================================================*/

#include "biulib.hh"
#include "opencv2/core/devmem2d.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

template <typename T>
__global__ void cuda_add(T* a, T* b, T* c) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index]     = a[index] + b[index];
}

template <typename T>
__global__ void cuda_conv_2d(T* ori, T* kernal, T* res, int ori_w, int ori_h, int ker_w, int ker_h, int stride, bool padding, T padding_num) {
    int idx           = threadIdx.x + blockDim.x * blockIdx.x;
    int idy           = threadIdx.y + blockDim.y * blockIdx.y;
    int width_repeat  = 0;
    int height_repeat = 0;
    if (padding) {
        width_repeat  = (ori_w + blockDim.x - 1) / (blockDim.x * stride);
        height_repeat = (ori_h + blockDim.y - 1) / (blockDim.y * stride);
    }
    // printf("%d,%d ", width_repeat, height_repeat);
    for (int j = 0; j < height_repeat; ++j) {
        for (int i = 0; i < width_repeat; ++i) {
            int x = (i * 32 + idx);
            int y = (j * 32 + idy);
            if (x < ori_w && y < ori_h) {
                int r_id  = x + y * ori_w;
                res[r_id] = (T)0;
                int k_x   = 0;
                int k_y   = 0;
                int o_id  = 0;

                k_x  = x;
                k_y  = y - 1;
                o_id = k_x + k_y * ori_w;
                if (k_x >= 0 && k_x < ori_w && k_y >= 0 && k_y < ori_h) {
                    res[r_id] += ori[o_id] * kernal[1];
                }
                k_x  = x - 1;
                k_y  = y;
                o_id = k_x + k_y * ori_w;
                if (k_x >= 0 && k_x < ori_w && k_y >= 0 && k_y < ori_h) {
                    res[r_id] += ori[o_id] * kernal[3];
                }
                k_x  = x + 1;
                k_y  = y;
                o_id = k_x + k_y * ori_w;
                if (k_x >= 0 && k_x < ori_w && k_y >= 0 && k_y < ori_h) {
                    res[r_id] += ori[o_id] * kernal[5];
                }
                k_x  = x;
                k_y  = y + 1;
                o_id = k_x + k_y * ori_w;
                if (k_x >= 0 && k_x < ori_w && k_y >= 0 && k_y < ori_h) {
                    res[r_id] += ori[o_id] * kernal[7];
                }
            }
        }
    }
}

// template <typename T>
// __global__ void cuda_conv_2d(T* ori, T* kernal, T* res, int ori_w, int ori_h, int ker_w, int ker_h, int stride, bool padding, T padding_num) {
//     int idx           = threadIdx.x + blockDim.x * blockIdx.x;
//     int idy           = threadIdx.y + blockDim.y * blockIdx.y;
//     int width_repeat  = 0;
//     int height_repeat = 0;
//     if (padding) {
//         width_repeat  = (ori_w + blockDim.x - 1) / (blockDim.x * stride);
//         height_repeat = (ori_h + blockDim.y - 1) / (blockDim.y * stride);
//     }
//     // printf("%d,%d ", width_repeat, height_repeat);
//     for (int j = 0; j < height_repeat; ++j) {
//         for (int i = 0; i < width_repeat; ++i) {
//             int x = (i * blockDim.x + idx) * stride;
//             int y = (j * blockDim.y + idy) * stride;
//             if (x < ori_w && y < ori_h) {
//                 // printf("%d,%d ", x, y);
//                 int r_id  = x + y * ori_w;
//                 res[r_id] = (T)0;
//                 for (int kj = 0; kj < ker_h; ++kj) {
//                     for (int ki = 0; ki < ker_w; ++ki) {
//                         int k_x = x + (ki - ker_w / 2) * stride;
//                         int k_y = y + (kj - ker_h / 2) * stride;
//                         // printf("%d,%d ", k_x, k_y);
//                         int o_id = k_x + k_y * ori_w;
//                         int k_id = ki + kj * ker_w;
//                         if (k_x < 0 || k_x >= ori_w || k_y < 0 || k_y >= ori_h) {
//                             res[r_id] += padding_num * kernal[k_id];
//                         } else {
//                             res[r_id] += ori[o_id] * kernal[k_id];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

Mat<float> conv_2d(Mat<float> a, Mat<float> b, int stride, bool padding, float padding_num) {
    float *    d_a, *d_b, *d_c;
    int        NBytesA = a.dim.x * a.dim.y * sizeof(float);
    int        NBytesB = b.dim.x * b.dim.y * sizeof(float);
    Mat<float> res(a.dim);
    cudaMalloc((void**)&d_a, NBytesA);
    cudaMalloc((void**)&d_b, NBytesB);
    cudaMalloc((void**)&d_c, NBytesA);
    cudaMemcpy((void*)d_a, (void*)a.data, NBytesA, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_b, (void*)b.data, NBytesB, cudaMemcpyHostToDevice);
    dim3 blocksPerGrid((a.dim.x + CUDA_BLOCK_WIDTH - 1) / CUDA_BLOCK_WIDTH, (a.dim.y + CUDA_BLOCK_HEIGHT - 1) / CUDA_BLOCK_HEIGHT);
    dim3 threadsPerBlock(CUDA_BLOCK_WIDTH, CUDA_BLOCK_HEIGHT);
    cuda_conv_2d<float><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, a.dim.x, a.dim.y, b.dim.x, b.dim.y, stride, padding, padding_num);
    cudaMemcpy((void*)res.data, (void*)d_c, NBytesA, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return res;
}

Mat<unsigned char> conv_2d(Mat<unsigned char> a, Mat<unsigned char> b, int stride, bool padding, unsigned char padding_num) {
    unsigned char *    d_a, *d_b, *d_c;
    int                NBytesA = a.dim.x * a.dim.y * sizeof(unsigned char);
    int                NBytesB = b.dim.x * b.dim.y * sizeof(unsigned char);
    auto               st      = std::chrono::high_resolution_clock().now();
    Mat<unsigned char> res(a.dim);
    cudaMalloc((void**)&d_a, NBytesA);
    cudaMalloc((void**)&d_b, NBytesB);
    cudaMalloc((void**)&d_c, NBytesA);
    cudaMemcpy((void*)d_a, (void*)a.data, NBytesA, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_b, (void*)b.data, NBytesB, cudaMemcpyHostToDevice);
    dim3 blocksPerGrid((a.dim.x + CUDA_BLOCK_WIDTH - 1) / CUDA_BLOCK_WIDTH, (a.dim.y + CUDA_BLOCK_HEIGHT - 1) / CUDA_BLOCK_HEIGHT);
    dim3 threadsPerBlock(CUDA_BLOCK_WIDTH, CUDA_BLOCK_HEIGHT);
    cuda_conv_2d<unsigned char><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, a.dim.x, a.dim.y, b.dim.x, b.dim.y, stride, padding, padding_num);
    cudaMemcpy((void*)res.data, (void*)d_c, NBytesA, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return res;
}
template <typename T>
vector<T> add(vector<T> a, vector<T> b) {
    T *d_a, *d_b, *d_c;
    assert(a.size() == b.size());
    size_t    NBytes = a.size() * sizeof(T);
    vector<T> res;
    res.resize(a.size());
    cudaMalloc((void**)&d_a, NBytes);
    cudaMalloc((void**)&d_b, NBytes);
    cudaMalloc((void**)&d_c, NBytes);
    cudaMemcpy((void*)d_a, (void*)a.data(), NBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_b, (void*)b.data(), NBytes, cudaMemcpyHostToDevice);
    dim3 blocksPerGrid((a.dim.x + CUDA_BLOCK_WIDTH - 1) / CUDA_BLOCK_WIDTH, (a.dim.y + CUDA_BLOCK_HEIGHT - 1) / CUDA_BLOCK_HEIGHT);
    dim3 threadsPerBlock(CUDA_BLOCK_WIDTH, CUDA_BLOCK_HEIGHT);
    cuda_add<float><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    cudaMemcpy((void*)res.data(), (void*)d_c, NBytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return res;
}
