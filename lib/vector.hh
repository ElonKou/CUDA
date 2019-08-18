#ifndef VECTOR_ADD_H_GPU
#define VECTOR_ADD_H_GPU

#define CUDA_W 16
#define CUDA_H 16

#pragma once

extern "C" {
void add_int(int *A, int *B, int *C, int N);
void add_float(float *A, float *B, float *C, int N);
}

#endif