#include <stdio.h>
#define N 10

__global__ void add(int *a, int *b, int *c){
    if(threadIdx.x < N){
        c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
    }
}

int main(){
    int A[N], B[N], C[N];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));

    for(int i =0;i < N; i++){
        A[i] = -i;
        B[i] = i * i;
        C[i] = 0;
    }

    cudaMemcpy(dev_a, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, B, N * sizeof(int), cudaMemcpyHostToDevice);
    add<<<1, 10>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(C, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0;i < N; i++){
        printf("%d + %d = %d\n", A[i], B[i], C[i]);
    }
    return 0;
}