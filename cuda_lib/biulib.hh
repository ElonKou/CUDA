#pragma once
#ifndef BIULIB_H_
#define BIULIB_H_

#define CUDA_BLOCK_WIDTH 32
#define CUDA_BLOCK_HEIGHT 32

#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <vector>

using namespace std;

#define CHECK(call)                                      \
    {                                                    \
        const cudaError_t error = call;                  \
        if (error != cudaSuccess) {                      \
            printf("ERROR: %s:%d,", __FILE__, __LINE__); \
            exit(1);                                     \
        }                                                \
    }

union Dim {
    struct {
        int x;
        int y;
        int z;
    };
    int data[];
    Dim() {}
    Dim(int x_)
        : x(x_) {}
    Dim(int x_, int y_)
        : x(x_)
        , y(y_) {
    }
    Dim(int x_, int y_, int z_)
        : x(x_)
        , y(y_)
        , z(z_) {
    }
    int operator[](int index) { return data[index]; }
    ~Dim() {}
};

template <typename T>
struct Mat {
    Dim dim;
    T*  data;
    Mat(Dim dim_)
        : dim(dim_) {
        data = (T*)malloc(dim.x * dim.y * sizeof(T));
        memset(data, 0, dim.x * dim.y * sizeof(T));
    }
    Mat() {
        free(data);
    }
    void Random() {
        for (size_t j = 0; j < dim.y; j++) {
            for (size_t i = 0; i < dim.x; i++) {
                data[j * dim.y + i] = (T)(rand() % 2);
            }
        }
    }
    friend ostream& operator<<(ostream& os, Mat& mat) {
        for (size_t j = 0; j < mat.dim.y; j++) {
            for (size_t i = 0; i < mat.dim.x; i++) {
                // os << mat.data[j * mat.dim.y + i] << " ";
                os << int(mat.data[j * mat.dim.y + i]) << " ";
            }
            if (j != (mat.dim.y - 1)) {
                os << endl;
            }
        }
        return os;
    }
};

// template <typename T>
// vector<T> add(vector<T> a, vector<T> b);

Mat<unsigned char> cv2Mat(cv::Mat grey);

cv::Mat Mat2cv(Mat<unsigned char> mat);

Mat<float> conv_2d(Mat<float> a, Mat<float> b, int stride, bool padding, float padding_num = 0.0);

Mat<unsigned char> conv_2d(Mat<unsigned char> a, Mat<unsigned char> b, int stride, bool padding, unsigned char padding_num = 0);

#endif