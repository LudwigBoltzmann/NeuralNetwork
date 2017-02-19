#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Type.hpp"
#include "Layer.hpp"

namespace DeepLearning
{

typedef std::vector<PrimitiveLayer*> layerVec;

class NeuralNetwork
{
private:
    layerVec    m_layers;

};


}


#endif // NEURALNETWORK_H
