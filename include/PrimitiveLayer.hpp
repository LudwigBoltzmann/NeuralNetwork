#ifndef PRIMITIVELAYER_HPP
#define PRIMITIVELAYER_HPP

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
    enum G_descentType { T_SGD, T_MOMENTUM, T_ADA, T_ADAM };
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
    double              m_momentum;
    E_functionType      m_errorFunctionType;
    O_filterType        m_outputFilterType;
    G_descentType       m_gradientDescentType;

public:
    PrimitiveLayer()
        : m_preLayer(NULL), m_postLayer(NULL), m_numInput(0), m_numOutput(0),
          m_input(), m_output(), m_outputPtr(NULL), m_weights(), m_bias(),
          m_wGrad(), m_bGrad(), m_delta(),
          m_learningRate(0.001), m_momentum(0.1),
          m_errorFunctionType(T_MSE), m_outputFilterType(T_IDENTITY),
          m_gradientDescentType(T_SGD)
    {}
    PrimitiveLayer(int nInput, int nOutput)
        : m_preLayer(NULL), m_postLayer(NULL), m_numInput(nInput), m_numOutput(nOutput),
          m_input(), m_output(), m_outputPtr(NULL), m_weights(), m_bias(),
          m_wGrad(), m_bGrad(), m_delta(),
          m_learningRate(0.001), m_momentum(0.1),
          m_errorFunctionType(T_MSE), m_outputFilterType(T_IDENTITY),
          m_gradientDescentType(T_SGD)
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
    double&             getMomentum(void) { return m_momentum; }

    void connectPostLayer(PrimitiveLayer* postLayer) {
        m_postLayer = postLayer;
        m_postLayer->getPreLayer() = this;
    }

    void setDescentType(G_descentType type)     { m_gradientDescentType = type; if(type != T_MOMENTUM) m_momentum = 0; }
    void setErrorFunction(E_functionType type)  { m_errorFunctionType = type; }
    void setOutputFilter(O_filterType type)     { m_outputFilterType = type; }
    void init(int nInput, int nOutput);
    double loss(int N, double* x, double* target);

    virtual void feedForward(double* input, int N) = 0;
    virtual void feedForward() = 0;
    virtual void backPropagation(double* target, int N) = 0;
    virtual void backPropagation() = 0;
    virtual void updateWeights() = 0;
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
    double scale = 1.0;
    for(int i = 0; i < n_weight; ++i) {
        m_weights.data()[i] = rg.boxMuller() / scale;
    }
    for(int i = 0; i < m_numOutput; ++i)
        m_bias[i] = rg.boxMuller() / scale;
}

double PrimitiveLayer::loss(int N, double *x, double *target)
{
    if(!N) return 0.0;

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

#endif // PRIMITIVELAYER_HPP
