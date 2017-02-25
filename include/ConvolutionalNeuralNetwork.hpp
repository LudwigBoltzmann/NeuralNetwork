#ifndef CONVOLUTIONALNEURALNETWORK_HPP
#define CONVOLUTIONALNEURALNETWORK_HPP

#include "PrimitiveLayer.hpp"

namespace DeepLearning
{

template <typename AF>
class CNN : public PrimitiveLayer
{
public:
    enum r_layerRoll { R_CONVOLUTION, R_POOLING, R_ACTIVATION };
private:
    int     m_whd[3];   /// Width x Height x Depth
    rMat    m_filter;


public:
    CNN() : PrimitiveLayer() {}
    CNN(int nInput, int nOutput, int* whd) : PrimitiveLayer(nInput, nOutput)
    {
        m_whd[0] = whd[0];  // Width
        m_whd[1] = whd[1];  // Height
        m_whd[2] = whd[2];  // Depth
    }

    void    initCNN();
    void    convolution(double *input, int* whd);
    void    activation(double *input, int* whd);
    void    pooling(double* input, int* whd);
    void    feedForward();
    void    feedForward(double *input, int N);
    void    backPropagation();
    void    backPropagation(double* target, int N);
    void    updateWeights();
};

template <typename AF>
void CNN<AF>::convolution(double* input, int* whd)
{

}

template <typename AF>
void CNN<AF>::activation(double* input, int* whd)
{

}

template <typename AF>
void CNN<AF>::pooling(double* input, int* whd)
{

}

template <typename AF>
void CNN<AF>::feedForward(double* input, int* whd)
{

}

template <typename AF>
void CNN<AF>::feedForward()
{

}

template <typename AF>
void CNN<AF>::backPropagation(double* target, int* whd)
{

}

template <typename AF>
void CNN<AF>::backPropagation()
{

}

template <typename AF>
void CNN<AF>::updateWeights()
{

}



}

#endif // CONVOLUTIONALNEURALNETWORK_HPP
