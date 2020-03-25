/*================================================================
*  Copyright (C)2020 All rights reserved.
*  FileName : 00.sayhello.cu
*  Author   : maxsense
*  Email    : elonkou@ktime.cc
*  Date     : 2020年03月18日 星期三 09时37分20秒
================================================================*/

#include <stdio.h>

__global__ void SayHello(){
    printf("hello from %d\n", threadIdx.x);
}

int main(int argc, char *argv[]){
    SayHello<<<1, 10>>>();
    // cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}
