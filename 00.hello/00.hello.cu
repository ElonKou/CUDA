#include <stdio.h>

#define N 8

__global__ void MatAdd(float A[N][N], float B[N][N], float C[][N]){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("i=%d, j= %d, c=%f\n", i, j, C[0][0]);
    // printf("i=%d, j= %d\n", i, j);
    printf("A=%f", A[0][0]);
    // printf("blockDim.x=%d, blockDim.y= %d\n", blockDim.x, blockDim.y);
    // printf("threadIdx.x=%d, threadIdx.y= %d\n", threadIdx.x, threadIdx.y);
    if (i < N && j < N){
        C[i][j] = A[i][j] + B[i][j];
    }
    // printf("c = %f \n", C[i][j]);
}

__global__ void modified(int *x){
    *x = 5;
    printf("Hello %d\n", *x);
}

void printArr(float arr[N][N]){
    for (int i = 0;i < N;i ++){
        for(int j = 0;j < N;j ++){
            printf("%0.1f\t",arr[i][j]);
        }
        printf("\n");
    }
}

void other(int *x){
    *x = 5;
    printf("other %d\n", *x);
}

int main(){
    float A[N][N];
    float B[N][N];
    float C[N][N];
    for (int i = 0;i < N;i ++){
        for(int j = 0;j < N;j ++){
            A[i][j] = i * N + j + 0.0;
            B[i][j] = j * N + i + 0.0;
            C[i][j] = 0.0;
        }
    }
    printArr(A);
    printArr(B);
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<< numBlocks, threadsPerBlock >>>(A, B, C);
    printArr(C);
    // int x = 11;
    // printf("x = %d\n", x);
    // modified<<<1, 3>>>(&x);
    cudaDeviceReset();
    // // other(&x);
    // printf("x = %d\n", x);
    return 0;
}
