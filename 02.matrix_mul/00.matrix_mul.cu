#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

struct Matrix{
    int width;
    int height;
    float *elements;
};

void showMatrix(Matrix *m){
    for(int i = 0;i < m->height;i++){
        for(int j =0;j< m->width;j++){
            cout << m->elements[j + i * m->width] << " ";
        }
        cout << endl;
    }
}

__global__ void matrixMul(Matrix *A, Matrix *B, Matrix *C){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    // printf("%d %d\t", row, col);
    float value = 0.0;
    float x = 0.0;
    float y = 0.0;
    if(col < C->width && row < C->height){
        for(int i = 0; i < A->width; i++){
            x = A->elements[row * A->width + i];
            y = B->elements[i * B->height + col];
            value += x * y;
        }
        C->elements[C->width * row + col] = value;
    }
}

int main(){
    int width = 1 << 2;
    int height = 1 << 2;
    int nBytes = width * height *sizeof(float);
    Matrix *A, *B, *C;

    cudaMallocManaged((void **)&A, sizeof(Matrix));
    cudaMallocManaged((void **)&B, sizeof(Matrix));
    cudaMallocManaged((void **)&C, sizeof(Matrix));
    cudaMallocManaged((void **)&A->elements, nBytes);
    cudaMallocManaged((void **)&B->elements, nBytes);
    cudaMallocManaged((void **)&C->elements, nBytes);

    A->width = width;
    A->height = height;
    B->width = height;
    B->height = width;
    C->width = height;
    C->height = height;
    
    for(int i =0 ;i < width * height;i ++){
        // A->elements[i] = i * 1.0;
        // B->elements[i] = i * 2.0;
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
        C->elements[i] = 0.0;
    }

    dim3 blockSize(4, 4);
    dim3 gridSize((height + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    matrixMul <<< gridSize, blockSize>>> (A, B, C);

    cudaDeviceSynchronize();

    // showMatrix(A);
    // showMatrix(B);
    showMatrix(C);

    return 0;
}