#include <stdio.h>
#include <stdlib.h>


__global__ void add(float* x, float * y, float* z, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // int stride = blockDim.x * gridDim.x;
    // printf("index:%d, stride:%d\n", index, stride);
    z[index] = x[index] + y[index];
}

int main()
{
    int N = 16;
    int nBytes = N * sizeof(float);
    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0 * i;
        y[i] = 20.0 * i;
    }

    // 申请device内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    // 将host数据拷贝到device
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
    // 定义kernel的执行配置
    dim3 blockSize(N);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    add <<< gridSize, blockSize >>> (d_x, d_y, d_z, N);

    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N;i ++){
        printf("%f ", z[i]);
    }
    printf("\n");


    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    // 释放host内存
    free(x);
    free(y);
    free(z);

    return 0;
}
