#ifndef CONVOLUTIONALNEURALNETWORK_HPP
#define CONVOLUTIONALNEURALNETWORK_HPP

#include "PrimitiveLayer.hpp"

namespace DeepLearning
{

class CNNInfo
{
public:
    int iWHD[3];
    int fWHD[3];
    int oWHD[3];
    int dim;
    int numChannel;
    int numFilter;
    int stride;
    int padding;
};

template <typename AF>
class CNN : public PrimitiveLayer
{
public:
    typedef std::vector<rTen>   rTVec;
    enum r_layerRoll { R_NONE, R_CONVOLUTION, R_POOLING, R_ACTIVATION };
    enum p_poolingType { P_MAX, P_AVERAGE, P_MIN };
private:
    int     m_iWHD[3];      /// Width x Height x Depth
    int     m_fWHD[3];      /// Width x Height x Depth
    int     m_pWHD[3];      /// Width x Height x Depth
    int     m_oWHD[3];      /// Width x Height x Depth
    int     m_dim;          /// dimension
    int     m_numChannel;   /// num channel
    int     m_numFilter;    /// num filter
    int     m_stride;       /// stride size
    int     m_padding;      /// padding size
    rTVec   m_tensor;       /// tensor for filter
    r_layerRoll m_roll;     /// roll of this layer
    p_poolingType m_poolType;   /// pooling type

public:
    CNN()
        : PrimitiveLayer(),
          m_numFilter(1), m_numChannel(1),
          m_stride(1), m_padding(0), m_dim(2),
          m_roll(R_NONE), m_poolType(P_MAX)
    {}

    // nInput = nwidth * nheight * ndepth * nchannel of input size
    CNN(int nInput, int nOutput, int* iwhd)
        : PrimitiveLayer(nInput, nOutput),
          m_numFilter(1), m_numChannel(1),
          m_stride(1), m_padding(0), m_dim(2),
          m_roll(R_NONE), m_poolType(P_MAX)
    {
        m_iWHD[0] = iwhd[0];  // Width
        m_iWHD[1] = iwhd[1];  // Height
        m_iWHD[2] = iwhd[2];  // Depth
    }

    r_layerRoll&    getRoll(void)           { return m_roll; }
    p_poolingType&  getPoolingType(void)    { return m_poolType; }
    int*    getiwhd(void) { return m_iWHD; }
    int*    getfwhd(void) { return m_fWHD; }
    int*    getowhd(void) { return m_oWHD; }
    void    setIDim(int w, int h, int d);
    void    setFDim(int w, int h, int d);
    void    setODim(int w, int h, int d);
    void    setRoll(r_layerRoll roll) { m_roll = roll; }
    void    setPoolingType(p_poolingType type) { m_poolType = type; }
    void    getLayerInfo(CNNInfo& info);
    void    initCNN(int dim, int numChannel, int stride, int* filterSize, int numFilter, int padding);
    double  conv(double* input, int f, int c);
    void    convolution(double *INPUT);
    void    activation(double *input);
    double  maximumPool(double* input);
    double  minimumPool(double* input);
    double  averagePool(double* input);
    void    pooling(double *input);
    void    feedForward();
    void    feedForward(double *input, int*);
    void    backPropagation();
    void    backPropagation(double* target, int* whd);
    void    updateWeights();
};

template <typename AF>
void CNN<AF>::setIDim(int w, int h, int d)
{
    m_iWHD[0] = w;
    m_iWHD[1] = h;
    m_iWHD[2] = d;
}

template <typename AF>
void CNN<AF>::setFDim(int w, int h, int d)
{
    m_fWHD[0] = w;
    m_fWHD[1] = h;
    m_fWHD[2] = d;
}

template <typename AF>
void CNN<AF>::setODim(int w, int h, int d)
{
    m_oWHD[0] = w;
    m_oWHD[1] = h;
    m_oWHD[2] = d;
}

template <typename AF>
void CNN<AF>::getLayerInfo(CNNInfo &info)
{
    info.iWHD[0] = m_iWHD[0];
    info.iWHD[1] = m_iWHD[1];
    info.iWHD[2] = m_iWHD[2];
    info.fWHD[0] = m_fWHD[0];
    info.fWHD[1] = m_fWHD[1];
    info.fWHD[2] = m_fWHD[2];
    info.oWHD[0] = m_oWHD[0];
    info.oWHD[1] = m_oWHD[1];
    info.oWHD[2] = m_oWHD[2];
    info.dim     = m_dim;
    info.numChannel = m_numChannel;
    info.numFilter  = m_numFilter;
    info.stride     = m_stride;
    info.padding    = m_padding;
}

template <typename AF>
void CNN<AF>::initCNN(int dim, int numChannel, int stride, int *filterSize, int numFilter, int padding)
{
    if(m_roll == R_NONE) {
        std::cout<<"init CNN must be called after set the roll of this layer"<<std::endl;
        exit(-1);
    }
    m_dim = dim;
    m_numChannel = numChannel;
    m_stride = stride;
    m_numFilter = numFilter;
    m_padding = padding;

    if(m_roll == R_POOLING) {

        m_pWHD[0] = filterSize[0];
        m_pWHD[1] = filterSize[1];
        if(dim == 3)
            m_pWHD[2] = filterSize[2];
        else
            m_pWHD[2] = 1;

        m_oWHD[0] = (m_iWHD[0] - filterSize[0]) / stride + 1;
        m_oWHD[1] = (m_iWHD[1] - filterSize[1]) / stride + 1;
        if(dim == 3)
            m_oWHD[2] = (m_iWHD[2] - filterSize[2]) / stride + 1;
        else
            m_oWHD[2] = 1;

    } else {
        m_fWHD[0] = filterSize[0];
        m_fWHD[1] = filterSize[1];
        if(dim == 3)
            m_fWHD[2] = filterSize[2];
        else
            m_fWHD[2] = 1;

        m_oWHD[0] = (m_iWHD[0] + 2 * m_padding - filterSize[0]) / stride + 1;
        m_oWHD[1] = (m_iWHD[1] + 2 * m_padding - filterSize[1]) / stride + 1;
        if(dim == 3)
            m_oWHD[2] = (m_iWHD[2] + 2 * m_padding - filterSize[2]) / stride + 1;
        else
            m_oWHD[2] = 1;

        m_tensor.resize(numFilter, rTen( filterSize[0],
                                         filterSize[1],
                                         filterSize[2],
                                         numChannel) );
        randomGenerator rg;
        for(int f = 0; f < numFilter; ++f) {
            rTen& tensor = m_tensor[f];
            for(int j = 0; j < tensor.data().size(); ++j) {
                tensor.data()[j] = rg.getUniform();
                tensor.data()[j] = 1.0;
            }
        }

    }
    int inputSize;
    int outputSize;
    if(dim == 3) {
        inputSize =   (m_iWHD[0] + 2 * m_padding)
                    * (m_iWHD[1] + 2 * m_padding)
                    * (m_iWHD[2] + 2 * m_padding) * numChannel;
    } else {
        inputSize =   (m_iWHD[0] + 2 * m_padding)
                    * (m_iWHD[1] + 2 * m_padding) * numChannel;
    }

    if(dim == 3) {
        outputSize =  (m_oWHD[0] + 2 * m_padding)
                    * (m_oWHD[1] + 2 * m_padding)
                    * (m_oWHD[2] + 2 * m_padding) * numChannel;
    } else {
        outputSize =  (m_oWHD[0] + 2 * m_padding)
                    * (m_oWHD[1] + 2 * m_padding) * numChannel;
    }
    m_input.resize(inputSize);
    m_output.resize(outputSize);
}

template <typename AF>
inline double CNN<AF>::conv(double *input, int f, int c)
{
    int i, j, k;
    rTen& tensor = m_tensor[f];
    double sum = 0.0, *data, *data1, *data2;
    for(k = 0; k < m_fWHD[2]; ++k) {
        data = input + k * m_iWHD[1] * m_iWHD[0];
        for(j = 0; j < m_fWHD[1]; ++j) {
            data1 = data + j * m_iWHD[0];
            for(i = 0; i < m_fWHD[0]; ++i) {
                data2 = data1 + i;
                sum += data2[i] * tensor(i,j,k,c);
            }
        }
    }
    return sum;
}

template <typename AF>
void CNN<AF>::convolution(double* INPUT)
{
    int D;
    int P = m_padding;
    int f, c, k, j, i, ks, js, is;
    double *data, *output, *input, *out, *input1, *out1, *input2;
    if(m_dim == 3) {
        D = P;
    } else {
        D = 0;
    }
    int Width  = m_iWHD[0] + 2 * P;
    int Height = m_iWHD[1] + 2 * P;
    for(int k = 0; k < m_iWHD[2]; ++k) {
        double* inputData2 = INPUT + k * m_iWHD[0] * m_iWHD[1];
        double* thisdata2  = m_input.data() + (k + D) * Width * Height;
        for(int j = 0; j < m_iWHD[1]; ++j) {
            double* inputData3 = inputData2 + j * m_iWHD[0];
            double* thisdata3  = thisdata2  + (j + P) * Width;
            for(int i = 0; i < m_iWHD[0]; ++i) {
                thisdata3[i + P] = inputData3[i];
            }
        }
    }

    m_output.assign(m_output.size(),0.0);
    for(f = 0; f < m_numFilter; ++f) {
        output  = m_output.data() + f * m_oWHD[0] * m_oWHD[1] * m_oWHD[2];
        for(c = 0; c < m_numChannel; ++c) {
            data = m_input.data() + c * m_iWHD[0] * m_iWHD[1] * m_iWHD[2];
            for(k = 0, ks = 0; k < m_oWHD[2]; ++k, ks += m_stride) {
                input = data   + ks * m_iWHD[0] * m_iWHD[1];
                out  = output  +  k * m_oWHD[0] * m_oWHD[1];
                for(j = 0, js = 0; j < m_oWHD[1]; ++j, js += m_stride) {
                    input1 = input   + js * m_iWHD[0];
                    out1  = out      +  j * m_oWHD[0];
                    for(i = 0, is = 0; i < m_oWHD[0]; ++i, is += m_stride) {
                        input2 = input1 + is;
                        out1[i] += conv(input2,f,c);
                    }
                }
            }
        }
    }
}

template <typename AF>
void CNN<AF>::activation(double* input)
{
    m_input.assign(input, input + m_numInput);
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
double CNN<AF>::maximumPool(double *input)
{
    double ret = -1e100;
    double *input0, *input1;
    for(int k = 0; k < m_pWHD[2]; ++k) {
        input0 = input + k * m_iWHD[0] * m_iWHD[1];
        for(int j = 0; j < m_pWHD[1]; ++j) {
            input1 = input0 + j * m_iWHD[0];
            for(int i = 0; i < m_pWHD[0]; ++i) {
                ret = std::max(ret,input1[i]);
            }
        }
    }
    return ret;
}

template <typename AF>
double CNN<AF>::minimumPool(double *input)
{
    double ret = 1e100;
    double *input0, *input1;
    for(int k = 0; k < m_pWHD[2]; ++k) {
        input0 = input + k * m_iWHD[0] * m_iWHD[1];
        for(int j = 0; j < m_pWHD[1]; ++j) {
            input1 = input0 + j * m_iWHD[0];
            for(int i = 0; i < m_pWHD[0]; ++i) {
                ret = std::min(ret,input1[i]);
            }
        }
    }
    return ret;
}

template <typename AF>
double CNN<AF>::averagePool(double *input)
{
    double ret = 0.0;
    double *input0, *input1;
    for(int k = 0; k < m_pWHD[2]; ++k) {
        input0 = input + k * m_iWHD[0] * m_iWHD[1];
        for(int j = 0; j < m_pWHD[1]; ++j) {
            input1 = input0 + j * m_iWHD[0];
            for(int i = 0; i < m_pWHD[0]; ++i) {
                ret += input1[i];
            }
        }
    }
    return ret / (m_pWHD[0] * m_pWHD[1] * m_pWHD[2]);
}

template <typename AF>
void CNN<AF>::pooling(double* input)
{
    int i, j, k, oi, oj, ok;
    double ret, *input0, *input1;
    double *output0, *output1;

    switch (m_poolType) {
    case P_MAX:
        ret = -1.0e100;
        break;
    case P_AVERAGE:
        ret =  0.0;
        break;
    case P_MIN:
        ret =  1.0e100;
        break;
    }

    switch (m_poolType) {
    case P_MAX:
    {
        for(k = 0, ok = 0; k < m_iWHD[2]; k+=m_pWHD[2], ++ok) {
            input0 = input + k * m_iWHD[0] * m_iWHD[1];
            output0 = m_output.data() + ok * m_oWHD[0] * m_oWHD[1];
            for(j = 0, oj = 0; j < m_iWHD[1]; j+=m_pWHD[1], ++oj) {
                input1 = input0 + j * m_iWHD[0];
                output1 = output0 + oj * m_oWHD[0];
                for(i = 0, oi = 0; i < m_iWHD[0]; i+=m_pWHD[0], ++oi) {
                    output1[oi] = maximumPool(input1 + i);
                }
            }
        }
    }
        break;
    case P_AVERAGE:
    {
        for(k = 0, ok = 0; k < m_iWHD[2]; k+=m_pWHD[2], ++ok) {
            input0 = input + k * m_iWHD[0] * m_iWHD[1];
            output0 = m_output.data() + ok * m_oWHD[0] * m_oWHD[1];
            for(j = 0, oj = 0; j < m_iWHD[1]; j+=m_pWHD[1], ++oj) {
                input1 = input0 + j * m_iWHD[0];
                output1 = output0 + oj * m_oWHD[0];
                for(i = 0, oi = 0; i < m_iWHD[0]; i+=m_pWHD[0], ++oi) {
                    output1[oi] = averagePool(input1 + i);
                }
            }
        }
    }
        break;
    case P_MIN:
    {
        for(k = 0, ok = 0; k < m_iWHD[2]; k+=m_pWHD[2], ++ok) {
            input0 = input + k * m_iWHD[0] * m_iWHD[1];
            output0 = m_output.data() + ok * m_oWHD[0] * m_oWHD[1];
            for(j = 0, oj = 0; j < m_iWHD[1]; j+=m_pWHD[1], ++oj) {
                input1 = input0 + j * m_iWHD[0];
                output1 = output0 + oj * m_oWHD[0];
                for(i = 0, oi = 0; i < m_iWHD[0]; i+=m_pWHD[0], ++oi) {
                    output1[oi] = minimumPool(input1 + i);
                }
            }
        }
    }
        break;
    }
}

template <typename AF>
void CNN<AF>::feedForward(double* input, int*)
{
    switch (m_roll) {
        case R_NONE:
        {
            std::cout<<"The roll of this layer is not set."<<std::endl;
            exit(-1);
            break;
        }
        case R_CONVOLUTION:
        {
            convolution(input);
            break;
        }
        case R_ACTIVATION:
        {
            activation(input);
            break;
        }
        case R_POOLING:
        {
            pooling(input);
            break;
        }
    }
}

template <typename AF>
void CNN<AF>::feedForward()
{
    switch (m_roll) {
        case R_NONE:
        {
            std::cout<<"The roll of this layer is not set."<<std::endl;
            exit(-1);
            break;
        }
        case R_CONVOLUTION:
        {
            convolution(m_preLayer->getOutput().data());
            break;
        }
        case R_ACTIVATION:
        {
            activation(m_preLayer->getOutput().data());
            break;
        }
        case R_POOLING:
        {
            pooling(m_preLayer->getOutput().data());
            break;
        }
    }
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
