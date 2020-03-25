/*================================================================
*  Copyright (C)2019 All rights reserved.
*  FileName : biu_test.cc
*  Author   : ElonKou
*  Email    : koudongliang@maxsense.ai
*  Date     : 2019年08月17日 星期六 17时43分25秒
================================================================*/

#include <iostream>
#include "biu.hh"

int main() {
    int N = 16;
    int A[16], B[16], C[16];
    for (int i = 0; i < N; ++i) {
        A[i] = 1 * i;
        B[i] = 2 * i;
        C[i] = 0;
    }
    add_int(A, B, C, N);
    for (int i = 0; i < 16; i++) {
        std::cout << C[i] << " ";
    }

    return 0;
}