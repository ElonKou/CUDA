/*================================================================
*  Copyright (C)2020 All rights reserved.
*  FileName : 01.check_cuda.cu
*  Author   : maxsense
*  Email    : elonkou@ktime.cc
*  Date     : 2020年03月18日 星期三 10时21分59秒
================================================================*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char *argv[]){
    int dev;
    cudaError_t cuda_error = cudaGetDeviceCount(&dev);
    cudaDeviceProp devProp;
    for(size_t i = 0 ;i < dev; i++){
        cuda_error = cudaGetDeviceProperties(&devProp, i);
        std::cout << "GPU device cnt: " << dev << std::endl; 
        std::cout << "Device Name: " << devProp.name << std::endl;
        std::cout << "SM count :" << devProp.multiProcessorCount << std::endl;
        std::cout << "Size of thread block shared memory: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Max thread count of thread block: ：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max thread count of EM: " << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max thread warp count of EM: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    }
    return 0;
}
