#ifndef ACTIVATIONFUNCTION_HPP
#define ACTIVATIONFUNCTION_HPP

#include "Type.hpp"

namespace DeepLearning
{
    class Relu {
    public:
        static double getActivation(double x) {
            if(x > 0) return x;
            else      return 0;
        }
        static double getActivationGradient(double x) {
            if(x > 0) return 1;
            else      return 0;
        }
    };

    class Sigmoid {
    public:
        static double getActivation(double x) {
            return 1.0 / (1.0 + exp(-x));
        }
        static double getActivationGradient(double x) {
            return (1.0 - getActivation(x)) * getActivation(x);
        }
    };

    class Naive {
    public:
        static double getActivation(double x) {
            return x;
        }
        static double getActivationGradient(double x) {
            return 1;
        }
    };

    class SoftMax {
    private:
        static double getMax(int N, double* x)
        {
            /// assume N is not zero
//            if(!N) return 0.0;
            double max = x[0];
            for(int i = 1; i < N; ++i) max = std::max(max, x[i]);
            return max;
        }

    public:
        static void getSoftMax(int N, double* x, double* y) {
            if(!N) return;
            double maxVal = getMax(N, x);
            double sum = 0.0;
            for(int i = 0; i < N; ++i) {
                /// prevent overflow
                sum += y[i] = exp(x[i] - maxVal);
            }
            for(int i = 0; i < N; ++i) {
                y[i] /= sum;
            }
        }
    };

    class Identity{
    private:
    public:
        static void getIdentity(int N, double* x, double* y) {
            for(int i = 0; i < N; ++i) y[i] = x[i];
        }
    };

}

#endif // ACTIVATIONFUNCTION_HPP
