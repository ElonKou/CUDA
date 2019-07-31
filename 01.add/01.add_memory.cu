#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
// #include "curand.h"
// #include "cublas_v2.h"
using namespace std;

__global__ void add(float *x, float *y, float *z, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n){
        z[index] = x[index] + y [index];
    }
}

__global__ void mul(float *x, float *y, float *z, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n){
        z[index] = x[index] * y[index];
    }
}

int main() {
    int N = 16;
    int nBytes = N * sizeof(float);
    float *x, *y, *z, *m;

    cudaMallocManaged((void **)&x, nBytes);
    cudaMallocManaged((void **)&y, nBytes);
    cudaMallocManaged((void **)&z, nBytes);
    cudaMallocManaged((void **)&m, nBytes);

    for (int i = 0; i < N; ++i) {
        x[i] = 10.0 * i;
        y[i] = 20.0 * i;
    }

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    add<<<gridSize, blockSize >>>(x, y, z, N);
    mul<<<gridSize, blockSize >>>(x, y, m, N);
    cudaDeviceSynchronize();

    cout << blockSize.x << " " << blockSize.y << endl;
    cout << gridSize.x << " " << gridSize.y << endl;
    for(int i =0;i < N;i ++){
        cout <<x[i] << "+" << y[i] << "=" <<  z[i] << endl;
        cout <<x[i] << "*" << y[i] << "=" <<  m[i] << endl;
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    return 0;
}