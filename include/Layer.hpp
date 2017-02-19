#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "Type.hpp"
#include "ActivationFunction.hpp"
#include "LossFunctions.hpp"
#include "GradientDescent.hpp"

namespace DeepLearning
{
    class PrimitiveLayer
    {
    public:
        enum E_functionType { T_MSE, T_CEE };
        enum O_filterType { T_SOFTMAX, T_IDENTITY };
    protected:
        PrimitiveLayer*     m_preLayer;
        PrimitiveLayer*     m_postLayer;
        int     m_numInput;
        int     m_numOutput;
        rVec    m_input;
        rVec    m_output;
        double* m_outputPtr;    //// not implemented yet
        rMat    m_weight;
        rVec    m_bias;
        rMat    m_wGrad;
        rVec    m_bGrad;
        E_functionType  m_errorFunctionType;
        O_filterType    m_outputFilterType;

    public:
        PrimitiveLayer()
            : m_preLayer(NULL), m_postLayer(NULL), m_numInput(0), m_numOutput(0),
              m_input(), m_output(), m_outputPtr(NULL), m_weight(), m_bias(), m_wGrad(), m_bGrad(),
              m_errorFunctionType(T_CEE), m_outputFilterType(T_SOFTMAX)
        {}
        PrimitiveLayer(int nInput, int nOutput)
            : m_preLayer(NULL), m_postLayer(NULL), m_numInput(nInput), m_numOutput(nOutput),
              m_input(), m_output(), m_outputPtr(NULL), m_weight(), m_bias(), m_wGrad(), m_bGrad(),
              m_errorFunctionType(T_CEE), m_outputFilterType(T_SOFTMAX)
        {
            init(nInput, nOutput);
        }

        rMat& getWeight(void) { return m_weight; }
        rVec& getInput(void) { return m_input; }
        rVec& getOutput(void) { return m_output; }
        rVec& getBias(void) { return m_bias; }

        void setErrorFunction(E_functionType type)  { m_errorFunctionType = type; }
        void setOutputFilter(O_filterType type)     { m_outputFilterType = type; }
        void init(int nInput, int nOutput);
        virtual void feedForward(double* input, int N) = 0;
        virtual double loss(int N, double* x, double* target) = 0;
        virtual void backPropagation(double* dout, int N) = 0;
    };

    void PrimitiveLayer::init(int nInput, int nOutput)
    {
        m_numInput = nInput;
        m_numOutput = nOutput;
        m_input.resize(nInput);
        m_output.resize(nOutput);
        m_weight.resize(nInput,nOutput);
        m_wGrad.resize(nInput,nOutput);
        m_bias.resize(nOutput);
        m_bGrad.resize(nInput,nOutput);

        randomGenerator rg;
        //// temporary weight initialization
        int n_weight = m_weight.data().size();
        for(int i = 0; i < n_weight; ++i) {
            m_weight.data()[i] = rg.boxMuller();
        }
        for(int i = 0; i < m_numOutput; ++i)
            m_bias[i] = rg.boxMuller();
    }


    template <typename AF>
    class Layer : public PrimitiveLayer
    {
    public:
        Layer() : PrimitiveLayer() {}
        Layer(int nInput, int nOutput) : PrimitiveLayer(nInput, nOutput) {}

        void feedForward(double* input, int N);
        double loss(int N, double *x, double *target);
        void backPropagation(double* dout, int N);
    };

    template <typename AF>
    void Layer<AF>::feedForward(double* input, int N) {
        m_input.assign(input, input + N);

        for(int i = 0; i < m_numOutput; ++i) {
            double output = 0.0;
            for(int j = 0; j < m_numInput; ++j) {
                output += m_input[j] * m_weight[j][i];
            }
            output = output + m_bias[i];
            m_output[i] = AF::getActivation(output);
        }
    }

    template <typename AF>
    double Layer<AF>::loss(int N, double *x, double *target)
    {
        if(!N) return 0.0;

        feedForward(m_input.data(),m_numInput);

        rVec y(N);

        switch (m_outputFilterType) {
        case T_IDENTITY:
            Identity::getIdentity(N,x,y.data());
            break;
        case T_SOFTMAX:
            SoftMax::getSoftMax(N,x,y.data());
            break;
        }

        switch (m_errorFunctionType) {
        case T_CEE:
            return CEE(N,y.data(),target);
        case T_MSE:
            return MSE(N,y.data(),target);
        }

        return 0.0;
    }

    template <typename AF>
    void Layer<AF>::backPropagation(double* dout, int N)
    {
        for(int i = 0; i < m_numOutput; ++i) {
            double output = 0.0;
            for(int j = 0; j < m_numInput; ++j) {
                output += m_input[j] * m_weight[j][i];
            }
            output = output + m_bias[i];
            m_output[i] = AF::getActivation(output);
        }

    }



}

#endif // PERCEPTRON_HPP
