/*================================================================
*  Copyright (C)2020 All rights reserved.
*  FileName : main.cc
*  Author   : maxsense
*  Email    : koudongliang@maxsense.ai
*  Date     : Tue 24 Mar 2020 03:41:16 PM CST
================================================================*/

#include "biulib.hh"
#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
    cv::Mat xx = cv::imread("../image/test_2.png");
    cv::Mat grey;
    cv::cvtColor(xx, grey, CV_BGR2GRAY);
    Mat<unsigned char> a = cv2Mat(grey);
    Mat<unsigned char> b(Dim(3, 3));
    Mat<unsigned char> c(a.dim);
    b.Random();
    c  = conv_2d(a, b, 1, true, 0);
    xx = Mat2cv(c);
    std::cout << b << std::endl;
    cv::imshow("image", xx);
    cv::waitKey(0);
    return 0;
}
