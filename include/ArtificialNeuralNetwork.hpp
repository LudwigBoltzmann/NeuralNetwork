#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "PrimitiveLayer.hpp"

namespace DeepLearning
{
template <typename AF>
class ANN : public PrimitiveLayer
{
public:
    ANN() : PrimitiveLayer() {}
    ANN(int nInput, int nOutput) : PrimitiveLayer(nInput, nOutput) {}

    void feedForward(double* input, int N);
    void feedForward();
    void backPropagation(double* target, int N);
    void backPropagation();
    void updateWeights();
};

template <typename AF>
void ANN<AF>::feedForward(double* input, int N) {
    m_input.assign(input, input + N);

    cilk_for(int i = 0; i < m_numOutput; ++i) {
        double output = 0.0;
        for(int j = 0; j < m_numInput; ++j) {
            output += m_input[j] * m_weights[j][i];
        }
        output = output + m_bias[i];
        m_output[i] = AF::getActivation(output);
    }
}

template <typename AF>
void ANN<AF>::feedForward() {
    if(m_preLayer == NULL) {
        std::cerr<<"Wrong use of the feed forward"<<std::endl;
        return;
    }

    m_input.assign(m_preLayer->getOutput().begin(), m_preLayer->getOutput().end());

    cilk_for(int i = 0; i < m_numOutput; ++i) {
        double output = 0.0;
        for(int j = 0; j < m_numInput; ++j) {
            output += m_input[j] * m_weights[j][i];
        }
        output = output + m_bias[i];
        m_output[i] = AF::getActivation(output);
    }
}


template <typename AF>
void ANN<AF>::backPropagation(double* target, int N)
{
    if(m_postLayer == NULL) {
        for(int i = 0; i < N; ++i)
            m_delta[i] = target[i] - m_output[i];
    } else {
        backPropagation();
    }
}

template <typename AF>
void ANN<AF>::backPropagation()
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

    cilk_for(int j = 0; j < numInput; ++j) {
        double delta = 0;
        for(int i = 0; i < numOutput; ++i) {
            delta += postLayerWeights[j][i] * postLayerDelta[i];
        }
        currentDelta[j] = delta;
    }
}

template <typename AF>
void ANN<AF>::updateWeights()
{
    switch (m_gradientDescentType) {
    case T_SGD:
        cilk_for(int j = 0; j < m_numOutput; ++j) {
            double delta = m_delta[j];
            for(int i = 0; i < m_numInput; ++i) {
                m_weights[i][j] += m_learningRate * delta * AF::getActivationGradient(m_input[i]) * m_input[i];
            }
            m_bias[j] += m_learningRate * delta * AF::getActivationGradient(1.0);
        }
        break;
    case T_MOMENTUM:
        cilk_for(int j = 0; j < m_numOutput; ++j) {
            double delta = m_delta[j];
            for(int i = 0; i < m_numInput; ++i) {
                m_wGrad[i][j] = m_momentum * m_wGrad[i][j] + m_learningRate * delta * AF::getActivationGradient(m_input[i]) * m_input[i];
                m_weights[i][j] += m_wGrad[i][j];
            }
            m_bGrad[j] = m_momentum * m_bGrad[j] + m_learningRate * delta * AF::getActivationGradient(1.0);
            m_bias[j] += m_bGrad[j];
        }
        break;
    case T_ADA:
    {
        double h = 0;
        for(int j = 0; j < m_numOutput; ++j) {
            double delta = m_delta[j];
            for(int i = 0; i < m_numInput; ++i) {
                m_wGrad[i][j] = delta * AF::getActivationGradient(m_input[i]) * m_input[i];
                h += m_wGrad[i][j] * m_wGrad[i][j];
            }
            m_bGrad[j] = delta * AF::getActivationGradient(1.0);
            h += m_bGrad[j] * m_bGrad[j];
        }
        m_momentum += h;
        cilk_for(int j = 0; j < m_numOutput; ++j) {
            for(int i = 0; i < m_numInput; ++i) {
                m_weights[i][j] += m_learningRate * m_wGrad[i][j] / sqrt(m_momentum);
            }
            m_bias[j] += m_learningRate * m_bGrad[j] / sqrt(m_momentum);
        }
        break;
    }
    }

}


}

#endif // PERCEPTRON_HPP
