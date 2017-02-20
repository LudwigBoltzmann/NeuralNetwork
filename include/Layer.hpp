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
        int                 m_numInput;
        int                 m_numOutput;
        rVec                m_input;
        rVec                m_output;
        double*             m_outputPtr;    //// not implemented yet
        rMat                m_weights;
        rVec                m_bias;
        rMat                m_wGrad;
        rVec                m_bGrad;
        rVec                m_delta;
        double              m_learningRate;
        E_functionType      m_errorFunctionType;
        O_filterType        m_outputFilterType;

    public:
        PrimitiveLayer()
            : m_preLayer(NULL), m_postLayer(NULL), m_numInput(0), m_numOutput(0),
              m_input(), m_output(), m_outputPtr(NULL), m_weights(), m_bias(),
              m_wGrad(), m_bGrad(), m_delta(), m_learningRate(0.001),
              m_errorFunctionType(T_MSE), m_outputFilterType(T_IDENTITY)
        {}
        PrimitiveLayer(int nInput, int nOutput)
            : m_preLayer(NULL), m_postLayer(NULL), m_numInput(nInput), m_numOutput(nOutput),
              m_input(), m_output(), m_outputPtr(NULL), m_weights(), m_bias(),
              m_wGrad(), m_bGrad(), m_delta(), m_learningRate(0.001),
              m_errorFunctionType(T_MSE), m_outputFilterType(T_IDENTITY)
        {
            init(nInput, nOutput);
        }

        PrimitiveLayer*&    getPreLayer(void)  {   return  m_preLayer; }
        PrimitiveLayer*&    getPostLayer(void)  {   return  m_postLayer; }
        rMat&               getWeights(void) { return m_weights; }
        rVec&               getInput(void) { return m_input; }
        rVec&               getOutput(void) { return m_output; }
        rVec&               getBias(void) { return m_bias; }
        rVec&               getDelta(void) { return m_delta; }
        double&             getLearningRate(void) { return m_learningRate; }

        void connectPostLayer(PrimitiveLayer* postLayer) {
            m_postLayer = postLayer;
            m_postLayer->getPreLayer() = this;
        }

        void setErrorFunction(E_functionType type)  { m_errorFunctionType = type; }
        void setOutputFilter(O_filterType type)     { m_outputFilterType = type; }
        void init(int nInput, int nOutput);
        virtual void feedForward(double* input, int N) = 0;
        virtual void feedForward() = 0;
        virtual void backPropagation(double* target, int N) = 0;
        virtual void backPropagation() = 0;
        virtual void updateWeights() = 0;
        virtual double loss(int N, double* x, double* target) = 0;
    };

    void PrimitiveLayer::init(int nInput, int nOutput)
    {
        m_numInput = nInput;
        m_numOutput = nOutput;
        m_input.resize(nInput);
        m_output.resize(nOutput);
        m_weights.resize(nInput,nOutput);
        m_wGrad.resize(nInput,nOutput);
        m_bias.resize(nOutput);
        m_bGrad.resize(nInput,nOutput);
        m_delta.resize(nOutput);

        randomGenerator rg;
        //// temporary weight initialization
        int n_weight = m_weights.data().size();
        for(int i = 0; i < n_weight; ++i) {
            m_weights.data()[i] = rg.boxMuller();
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
        void feedForward();
        void backPropagation(double* target, int N);
        void backPropagation();
        void updateWeights();
        double loss(int N, double *x, double *target);
    };

    template <typename AF>
    void Layer<AF>::feedForward(double* input, int N) {
        m_input.assign(input, input + N);

        for(int i = 0; i < m_numOutput; ++i) {
            double output = 0.0;
            for(int j = 0; j < m_numInput; ++j) {
                output += m_input[j] * m_weights[j][i];
            }
            output = output + m_bias[i];
            m_output[i] = AF::getActivation(output);
        }
    }

    template <typename AF>
    void Layer<AF>::feedForward() {
        if(m_preLayer == NULL) {
            std::cerr<<"Wrong use of the feed forward"<<std::endl;
            return;
        }

        m_input.assign(m_preLayer->getOutput().begin(), m_preLayer->getOutput().end());

        for(int i = 0; i < m_numOutput; ++i) {
            double output = 0.0;
            for(int j = 0; j < m_numInput; ++j) {
                output += m_input[j] * m_weights[j][i];
            }
            output = output + m_bias[i];
            m_output[i] = AF::getActivation(output);
        }
    }


    template <typename AF>
    void Layer<AF>::backPropagation(double* target, int N)
    {
        if(m_postLayer == NULL) {
            for(int i = 0; i < N; ++i)
                m_delta[i] = target[i] - m_output[i];
        } else {
            backPropagation();
        }
    }

    template <typename AF>
    void Layer<AF>::backPropagation()
    {
        if(m_postLayer == NULL) {
            std::cerr<<"Wrong use of the back propagation"<<std::endl;
            return;
        }
        rVec& postLayerDelta = m_postLayer->getDelta();
        rMat& postLayerWeights = m_postLayer->getWeights();
        rVec& currentDelta = m_delta;

        int numOutput = postLayerWeights.ncol();
        int numInput  = postLayerWeights.nrow();

        for(int j = 0; j < numInput; ++j) {
            double delta = 0;
            for(int i = 0; i < numOutput; ++i) {
                delta += postLayerWeights[j][i] * postLayerDelta[i];
            }
            currentDelta[j] = delta;
        }
    }

    template <typename AF>
    void Layer<AF>::updateWeights()
    {
        for(int j = 0; j < m_numOutput; ++j) {
            double delta = m_delta[j];
            for(int i = 0; i < m_numInput; ++i) {
                m_weights[i][j] += m_learningRate * delta * AF::getActivationGradient(m_input[i]) * m_input[i];
            }
            m_bias[j] += m_learningRate * delta * AF::getActivationGradient(1.0);
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


}

#endif // PERCEPTRON_HPP
