#include <iostream>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "include/Layer.hpp"
#include "include/ActivationFunction.hpp"
#include "include/MNIST/mnist_reader_less.hpp"
#include "include/LossFunctions.hpp"
#include "include/GradientDescent.hpp"

#include "mkl.h"

using namespace std;
using namespace DeepLearning;

double function2(int N, double* x) {
    return x[0] * x[0] + x[1] * x[1];
}

void multiLayerTrain()
{
    double input[2] = { 0.6, 0.9 };
    double target[3] = { 1, 0.3, 0.6 };
    std::cout<<std::endl<<"Test multilayer Train"<<std::endl<<std::endl;
    DeepLearning::Layer<DeepLearning::Sigmoid> layer0;
    DeepLearning::Layer<DeepLearning::Sigmoid> layer1;
    DeepLearning::Layer<DeepLearning::Naive> layer2;

    layer0.init(2,5);
    layer1.init(5,7);
    layer2.init(7,3);

    layer0.connectPostLayer(&layer1);
    layer1.connectPostLayer(&layer2);

    layer0.getLearningRate() = 0.001;
    layer1.getLearningRate() = 0.01;
    layer2.getLearningRate() = 0.1;

    std::cout<<"Epoch : "<<-1<<"\ttarget = "<<target[0]<<" "<<target[1]<<" "<<target[2]<<std::endl;

    for(int i = 0; i < 100; ++i) {
        layer0.feedForward(input,2);
        layer1.feedForward();
        layer2.feedForward();

        layer2.backPropagation(target,3);
        layer1.backPropagation();
        layer0.backPropagation();

        layer0.updateWeights();
        layer1.updateWeights();
        layer2.updateWeights();

        rVec& output = layer2.getOutput();
        if(i%10 == 0)
            std::cout<<"Epoch : "<<i<<"\toutput = "<<output[0]<<" "<<output[1]<<" "<<output[2]<<" "<<layer2.loss(3,layer2.getOutput().data(),target)<<std::endl;
    }
    std::cout<<std::endl<<"Test multilayer Train Done"<<std::endl<<std::endl;

}

void matrixSolveRegressionTest()
{
    int num_training_Data = 100000;
    double matrix[9];

    matrix[0] = 3.0;    matrix[1] = 0.0;   matrix[2] = 0.0;
    matrix[3] = 0.0;   matrix[4] = 3.0;    matrix[5] = 0.0;
    matrix[6] = 0.0;    matrix[7] = 0.0;   matrix[8] = 3.0;

    double input[3 * num_training_Data];    /// b
    double target[3 * num_training_Data];   /// result

    int ld = 3;
    int ipiv[3];
    int info;
    int one = 1;

    DGETRF(&ld,&ld,matrix,&ld,ipiv,&info);
    std::cout<<"info = "<<info<<std::endl;

    for(int i = 0; i < num_training_Data; ++i) {
        double* rhs = input + i * 3;
        double* x   = target + i * 3;
        x[0] = rhs[0] = (double)rand() / RAND_MAX * 100;
        x[1] = rhs[1] = (double)rand() / RAND_MAX * 100;
        x[2] = rhs[2] = (double)rand() / RAND_MAX * 100;

        DGETRS("N", &ld, &one, matrix, &ld, ipiv, x, &ld, &info);
//            std::cout<<x[0]<<" "<<x[1]<<" "<<x[2]<<std::endl;
    }

    std::cout<<std::endl<<"Test matrix solve"<<std::endl<<std::endl;
    DeepLearning::Layer<DeepLearning::Naive> layer0;
    DeepLearning::Layer<DeepLearning::Naive> layer1;
    DeepLearning::Layer<DeepLearning::Naive> layer2;
    DeepLearning::Layer<DeepLearning::Naive> layer3;

    layer0.init(3,3);
    layer1.init(3,3);
    layer2.init(3,3);
    layer3.init(3,3);

    layer0.connectPostLayer(&layer1);
    layer1.connectPostLayer(&layer2);
    layer2.connectPostLayer(&layer3);

    layer0.getLearningRate() = 0.000000015;
    layer1.getLearningRate() = 0.00000015;
    layer2.getLearningRate() = 0.0000015;
    layer3.getLearningRate() = 0.000015;

    for(int i = 0; i < num_training_Data; ++i) {
        layer0.feedForward(input + i * 3, 3);
        layer1.feedForward();
        layer2.feedForward();
        layer3.feedForward();

        layer3.backPropagation(target + i * 3, 3);
        layer2.backPropagation();
        layer1.backPropagation();
        layer0.backPropagation();

        layer0.updateWeights();
        layer1.updateWeights();
        layer2.updateWeights();
        layer3.updateWeights();

        rVec& output = layer3.getOutput();
        if(i%1000 == 0)
            std::cout<<"Epoch : "<<i<<"\t loss = "<<layer3.loss(3, layer3.getOutput().data(), target + i * 3)<<std::endl;
    }
    std::cout<<" Train Done"<<std::endl;
    std::cout<<" validation"<<std::endl;

    double validation_rhs[3] = {10, 15, 25};
    double answer[3] = {10, 15, 25};

    DGETRS("N", &ld, &one, matrix, &ld, ipiv, answer, &ld, &info);

    std::cout<<"Answer = "<<answer[0]<<" "<<answer[1]<<" "<<answer[2]<<std::endl;

    layer0.feedForward(validation_rhs, 3);
    layer1.feedForward();
    layer2.feedForward();
    layer3.feedForward();

    rVec& output = layer3.getOutput();
    std::cout<<"output = "<<output[0]<<" "<<output[1]<<" "<<output[2]<<std::endl;
    std::cout<<std::endl<<"Test matrix solve Done"<<std::endl<<std::endl;
}

void trainTest()
{
    double input[2] = { 0.6, 0.9 };
    double target[3] = { 1, 0.3, 0.6 };
    std::cout<<std::endl<<"Test train"<<std::endl<<std::endl;
    DeepLearning::Layer<DeepLearning::Naive> layer;
    layer.init(2,3);
    layer.getLearningRate() = 0.1;

    std::cout<<"Epoch : "<<0<<" target = "<<target[0]<<" "<<target[1]<<" "<<target[2]<<std::endl;

    for(int i = 0; i < 100; ++i) {
        layer.feedForward(input,2);
        layer.backPropagation(target,3);
        layer.updateWeights();

        rVec& output = layer.getOutput();
        if(i%10 == 0)
        std::cout<<"Epoch : "<<i<<" output = "<<output[0]<<" "<<output[1]<<" "<<output[2]<<std::endl;
    }
    std::cout<<std::endl<<"Test train Done"<<std::endl<<std::endl;
}

void mnistTest()
{
    std::cout<<"MNIST test start"<<std::endl;
    std::vector<std::vector<uint8_t>> trainImage =
            mnist::read_mnist_image_file("./MNISTSet/train-images-idx3-ubyte");
    std::vector<std::vector<uint8_t>> testImage =
            mnist::read_mnist_image_file("./MNISTSet/t10k-images-idx3-ubyte");
    std::vector<uint8_t>    trainLabel =
            mnist::read_mnist_label_file("./MNISTSet/train-labels-idx1-ubyte");
    std::vector<uint8_t>    testLabel =
            mnist::read_mnist_label_file("./MNISTSet/t10k-labels-idx1-ubyte");

    std::cout<<"the number of train set = "<<trainImage.size()<<std::endl;
    std::cout<<"the number of test set  = "<<testImage.size()<<std::endl;

    Layer<Sigmoid> layer0;
    Layer<Sigmoid> layer1;
    Layer<Sigmoid> layer2;
    Layer<Sigmoid> layer3;
    layer0.init(784,784);
    layer1.init(784,512);
    layer2.init(512,1024);
    layer3.init(1024,10);

    layer0.connectPostLayer(&layer1);
    layer1.connectPostLayer(&layer2);
    layer2.connectPostLayer(&layer3);

    layer0.getLearningRate() = 0.0001;
    layer1.getLearningRate() = 0.001;
    layer2.getLearningRate() = 0.01;
    layer3.getLearningRate() = 0.1;

    layer0.getMomentum() = 0.1;
    layer1.getMomentum() = 0.1;
    layer2.getMomentum() = 0.1;
    layer3.getMomentum() = 0.1;

    Layer<Sigmoid>::G_descentType type = Layer<Sigmoid>::T_SGD;
    layer0.setDescentType(type);
    layer1.setDescentType(type);
    layer2.setDescentType(type);
    layer3.setDescentType(type);
    layer0.setErrorFunction(Layer<Sigmoid>::T_MSE);
    layer1.setErrorFunction(Layer<Sigmoid>::T_MSE);
    layer2.setErrorFunction(Layer<Sigmoid>::T_MSE);
    layer3.setErrorFunction(Layer<Sigmoid>::T_MSE);
    layer0.setOutputFilter(Layer<Sigmoid>::T_SOFTMAX);
    layer1.setOutputFilter(Layer<Sigmoid>::T_SOFTMAX);
    layer2.setOutputFilter(Layer<Sigmoid>::T_SOFTMAX);
    layer3.setOutputFilter(Layer<Sigmoid>::T_SOFTMAX);


    double input[784];
    double target[10];
    double average = 0.0;

    /// pre-train
    for(int iter = 0; iter < 10; ++iter)
    for(int i = 0; i < 100; ++i) {
        for(int j = 0; j < 784; ++j)
            input[j] = trainImage[i][j];
        for(int j = 0; j < 10; ++j)
            target[j] = 0;
        target[trainLabel[i]] = 1;
        layer0.feedForward(input,784);
        layer1.feedForward();
        layer2.feedForward();
        layer3.feedForward();

        layer3.backPropagation(target,10);
        layer2.backPropagation();
        layer1.backPropagation();
        layer0.backPropagation();

        layer0.updateWeights();
        layer1.updateWeights();
        layer2.updateWeights();
        layer3.updateWeights();

        average += layer3.loss(10,layer3.getOutput().data(),target);

        if((i+1)%100 == 0) {
            std::cout<<"epoch = "<<i+1<<" "<<average / 100<<std::endl;
            average = 0.0;
        }
    }

    /// train
    for(int i = 0; i < trainImage.size(); ++i) {
        for(int j = 0; j < 784; ++j)
            input[j] = trainImage[i][j];
        for(int j = 0; j < 10; ++j)
            target[j] = 0;
        target[trainLabel[i]] = 1;
        layer0.feedForward(input,784);
        layer1.feedForward();
        layer2.feedForward();
        layer3.feedForward();

        layer3.backPropagation(target,10);
        layer2.backPropagation();
        layer1.backPropagation();
        layer0.backPropagation();

        layer0.updateWeights();
        layer1.updateWeights();
        layer2.updateWeights();
        layer3.updateWeights();

        average += layer3.loss(10,layer3.getOutput().data(),target);

        if((i+1)%100 == 0) {
            std::cout<<"epoch = "<<i+1<<" "<<average / 100<<std::endl;
            average = 0.0;
        }

    }

    int count = 0;
    /// test
    for(int i = 0; i < testImage.size(); ++i) {
        for(int j = 0; j < 784; ++j)
            input[j] = testImage[i][j];

        layer0.feedForward(input,784);
        layer1.feedForward();
        layer2.feedForward();
        layer3.feedForward();

        int max = -1;
        double maxval = 0.0;
        for(int j = 0; j < 10; ++j) {
            if(maxval < layer3.getOutput()[j]) {
                maxval = layer3.getOutput()[j];
                max = j;
            }
        }

        std::cout<<max<<" "<<(int)testLabel[i];
        if(max == (int)testLabel[i]) {
            std::cout<<"  Correct"<<std::endl;
            count++;
        }
        else                 std::cout<<"  FAIL"<<std::endl;
    }

    std::cout<<"Accuracy = "<<(double)count / testImage.size() * 100.0<<"%"<<std::endl;



    std::cout<<"MNIST test done"<<std::endl;
}

int main(int argc, char *argv[])
{

    __cilkrts_set_param("nworkers","4");
    std::cout<<"Hello Neural Network!"<<std::endl;

    /// test Train
    {
        trainTest();
    }

    /// test multi layer train
    {
        multiLayerTrain();
    }

    /// test MNIST
    {
        mnistTest();
    }

    return 0;
}
