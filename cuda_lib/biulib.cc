/*================================================================
*  Copyright (C)2020 All rights reserved.
*  FileName : biulib.cc
*  Author   : maxsense
*  Email    : koudongliang@maxsense.ai
*  Date     : Tue 24 Mar 2020 03:56:12 PM CST
================================================================*/

#include "biulib.hh"
#include <iostream>

using namespace std;

Mat<unsigned char> cv2Mat(cv::Mat grey) {
    Mat<unsigned char> res(Dim(grey.size[1], grey.size[0]));
    mempcpy((void*)res.data, (void*)grey.data, grey.size[0] * grey.size[1]);
    return res;
}

cv::Mat Mat2cv(Mat<unsigned char> mat) {
    cv::Mat res(mat.dim.y, mat.dim.x, CV_8UC1);
    mempcpy((void*)res.data, (void*)mat.data, mat.dim.x * mat.dim.y);
    return res;
}