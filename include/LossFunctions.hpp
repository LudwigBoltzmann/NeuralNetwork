#ifndef LOSSFUNCTION_HPP
#define LOSSFUNCTION_HPP

#include "Type.hpp"

namespace DeepLearning
{


    /// mean square error
    inline double MSE(int N, double* output, double* target)
    {
        CILK_C_REDUCER_OPADD(sum, double, 0.0);
        CILK_C_REGISTER_REDUCER(sum);
        cilk_for(int i = 0; i < N; ++i) {

            REDUCER_VIEW(sum) += (output[i] - target[i]) * (output[i] - target[i]);
        }
        double ret = 0.5 * sum.value;
        CILK_C_UNREGISTER_REDUCER(sum);
        return ret;
    }

    /// cross entropy error
    inline double CEE(int N, double* output, double* target)
    {
        CILK_C_REDUCER_OPADD(sum, double, 0.0);
        CILK_C_REGISTER_REDUCER(sum);
        cilk_for(int i = 0; i < N; ++i)
        {

            REDUCER_VIEW(sum) += target[i] * log(output[i] + 1e-7);
        }
        double ret = -sum.value;
        CILK_C_UNREGISTER_REDUCER(sum);
        return ret;
    }

}

#endif // LOSSFUNCTION_HPP
