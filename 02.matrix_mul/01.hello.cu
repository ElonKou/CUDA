#include <stdio.h>
#include <cuda_runtime.h>

__global__ void SayHello(){
    printf("Hello from GPU:%d\n", threadIdx.x);
}

int main(){
    SayHello<<<1,10>>>();
    // cudaDeviceSynchronize();
    cudaDeviceReset(); 
    return 0;
}

