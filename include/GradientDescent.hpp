#ifndef GRADIENTDESCENT_HPP
#define GRADIENTDESCENT_HPP

#include "Type.hpp"

namespace DeepLearning
{

    inline void numerical_Gradient(double (*f)(int N, double* x), int N, int M, double* x, double* grad)
    {
        double val, dxf, fdx;
        for(int j = 0; j < M; ++j) {
            for(int i = 0; i < N; ++i) {
                val = x[i + j * N];
                x[i + j * N] = val - 1e-4;
                dxf = f(N, x + j * N);
                x[i + j * N] = val + 1e-4;
                fdx = f(N, x + j * N);
                grad[i + j * N] = (fdx - dxf) / 2e-4;
                x[i + j * N] = val;
            }
        }
    }

    inline void gradientDescent(double (*f)(int n, double* x), double alpha, double numStep, int N, int M, double* y)
    {
        rVec grad(N*M);
        for(int i = 0; i < numStep; ++i) {
            numerical_Gradient(f, N, M, y, grad.data());
            for(int k = 0; k < M; ++k) {
                for(int j = 0; j < N; ++j) {
                    y[k * N + j] -= alpha * grad[k * N + j];
                }
            }
        }
    }

}

#endif // GRADIENTDESCENT_HPP
