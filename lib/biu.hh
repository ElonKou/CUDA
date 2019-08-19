/*================================================================
*  Copyright (C)2019 All rights reserved.
*  FileName : biu.hh
*  Author   : ElonKou
*  Email    : elonkou@ktime.cc
*  Date     : 2019年08月19日 星期一 01时20分57秒
================================================================*/

#ifndef CUDA_BIU_H
#define CUDA_BIU_H

#define CUDA_W 16
#define CUDA_H 16

#pragma once

extern "C" {
void add_int(int *A, int *B, int *C, int N);
void add_float(float *A, float *B, float *C, int N);
void add_double(double *A, double *B, double *C, int N);
void add_const_int_int(int *A, int C, int N);
void add_const_float_float(float *A, float C, int N);
void add_const_double_double(double *A, double C, int N);
}

#endif