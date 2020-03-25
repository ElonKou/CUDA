/*================================================================
*  Copyright (C)2020 All rights reserved.
*  FileName : 00.add.cu
*  Author   : maxsense
*  Email    : elonkou@ktime.cc
*  Date     : 2020年03月18日 星期三 10时48分48秒
================================================================*/

#include <cuda_runtime.h>
#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

using namespace std;

template<typename T>
__global__ void cuda_add(T *a, T *b, T *c){
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

template<typename T>
vector<T> add(vector<T> a, vector<T> b){
    T*d_a, *d_b, *d_c;
    assert(a.size() == b.size());
    size_t NBytes = a.size() * sizeof(T);
    vector<T> res;
    res.resize(a.size());
    cudaMalloc((void **)&d_a, NBytes);
    cudaMalloc((void **)&d_b, NBytes);
    cudaMalloc((void **)&d_c, NBytes);
    cudaMemcpy((void *)d_a, (void *)a.data(), NBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_b, (void *)b.data(), NBytes, cudaMemcpyHostToDevice);
    dim3 blocksPerGrid(a.size() / 32);
    dim3 threadsPerBlock(32);
    cuda_add<float><<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    cudaMemcpy((void *)res.data(), (void *)d_c, NBytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return res;
}

int main(int argc, char *argv[]){
    size_t N  = 64;
    vector<float> a, b, c;
    for(size_t i = 0;i < N; i++){
        a.push_back(i * 1.0);
        b.push_back(i * 2.1 - 5.3);
    }
    c = add<float>(a, b);
    for(size_t i = 0;i < N; i++){
        cout << c[i] << " ";
    }
    return 0;
}
