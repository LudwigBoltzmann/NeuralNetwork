#include <iostream>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>


/// load ANN
#include "include/ArtificialNeuralNetwork.hpp"
/// load CNN
#include "include/ConvolutionalNeuralNetwork.hpp"

#include "include/ActivationFunction.hpp"
#include "include/MNIST/mnist_reader_less.hpp"
#include "include/LossFunctions.hpp"

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
    DeepLearning::ANN<DeepLearning::Sigmoid> layer0;
    DeepLearning::ANN<DeepLearning::Sigmoid> layer1;
    DeepLearning::ANN<DeepLearning::Naive> layer2;

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
        int two = 2;
        int three = 3;
        layer0.feedForward(input,&two);
        layer1.feedForward();
        layer2.feedForward();

        layer2.backPropagation(target,&three);
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
    DeepLearning::ANN<DeepLearning::Naive> layer0;
    DeepLearning::ANN<DeepLearning::Naive> layer1;
    DeepLearning::ANN<DeepLearning::Naive> layer2;
    DeepLearning::ANN<DeepLearning::Naive> layer3;

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
        int three = 3;
        layer0.feedForward(input + i * 3, &three);
        layer1.feedForward();
        layer2.feedForward();
        layer3.feedForward();

        layer3.backPropagation(target + i * 3, &three);
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

    int three = 3;
    layer0.feedForward(validation_rhs, &three);
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
    DeepLearning::ANN<DeepLearning::Naive> layer;
    layer.init(2,3);
    layer.getLearningRate() = 0.1;

    std::cout<<"Epoch : "<<0<<" target = "<<target[0]<<" "<<target[1]<<" "<<target[2]<<std::endl;

    int two = 2;
    int three = 3;
    for(int i = 0; i < 100; ++i) {
        layer.feedForward(input,&two);
        layer.backPropagation(target,&three);
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

    ANN<Sigmoid> layer0;
    ANN<Sigmoid> layer1;
    ANN<Sigmoid> layer2;
    ANN<Sigmoid> layer3;
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

    ANN<Sigmoid>::G_descentType type = ANN<Sigmoid>::T_SGD;
    layer0.setDescentType(type);
    layer1.setDescentType(type);
    layer2.setDescentType(type);
    layer3.setDescentType(type);
    layer0.setErrorFunction(ANN<Sigmoid>::T_MSE);
    layer1.setErrorFunction(ANN<Sigmoid>::T_MSE);
    layer2.setErrorFunction(ANN<Sigmoid>::T_MSE);
    layer3.setErrorFunction(ANN<Sigmoid>::T_MSE);
    layer0.setOutputFilter(ANN<Sigmoid>::T_SOFTMAX);
    layer1.setOutputFilter(ANN<Sigmoid>::T_SOFTMAX);
    layer2.setOutputFilter(ANN<Sigmoid>::T_SOFTMAX);
    layer3.setOutputFilter(ANN<Sigmoid>::T_SOFTMAX);


    double input[784];
    double target[10];
    double average = 0.0;

    int ileng = 784;
    int oleng = 10;
    /// pre-train
    for(int iter = 0; iter < 10; ++iter)
    for(int i = 0; i < 100; ++i) {
        for(int j = 0; j < 784; ++j)
            input[j] = trainImage[i][j];
        for(int j = 0; j < 10; ++j)
            target[j] = 0;
        target[trainLabel[i]] = 1;
        layer0.feedForward(input,&ileng);
        layer1.feedForward();
        layer2.feedForward();
        layer3.feedForward();

        layer3.backPropagation(target,&oleng);
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
        layer0.feedForward(input,&ileng);
        layer1.feedForward();
        layer2.feedForward();
        layer3.feedForward();

        layer3.backPropagation(target,&oleng);
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

        layer0.feedForward(input,&ileng);
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

void MNISTTestConv()
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

    CNN<Sigmoid>    layer0;
    CNN<Sigmoid>    layer1;

    double input[784];
    double target[10];
    double average = 0.0;

    int filterSize[3] = {3,3,1};
    layer0.setRoll(CNN<Sigmoid>::R_CONVOLUTION);
    layer0.setIDim(28,28,1);
    layer0.init(784,64);
    layer0.initCNN(2,1,1,filterSize,1,1);

    int poolingSize[3] = {2,2,1};
    layer1.setRoll(CNN<Sigmoid>::R_POOLING);
    layer1.setPoolingType(CNN<Sigmoid>::P_MAX);
    layer1.setIDim(28,28,1);
    layer1.init(784,196);
    layer1.initCNN(2,1,2,poolingSize,1,0);

//    int ileng = 784;
//    int oleng = 10;

    /// train
    for(int i = 0; i < 1; ++i) {
        for(int j = 0; j < 784; ++j)
            input[j] = trainImage[i][j];
        for(int j = 0; j < 10; ++j)
            target[j] = 0;
        target[trainLabel[i]] = 1;
        layer0.feedForward(input,NULL);
        layer1.feedForward(input,NULL);
    }

    std::cout<<"layer output "<<layer0.getOutput().size()<<std::endl;
    std::cout<<"layer dim info = "<<layer0.getiwhd()[0]<<" "<<layer0.getiwhd()[1]<<" "<<layer0.getiwhd()[2]<<std::endl;
    std::cout<<"layer dim info = "<<layer0.getfwhd()[0]<<" "<<layer0.getfwhd()[1]<<" "<<layer0.getfwhd()[2]<<std::endl;
    std::cout<<"layer dim info = "<<layer0.getowhd()[0]<<" "<<layer0.getowhd()[1]<<" "<<layer0.getowhd()[2]<<std::endl;
    int dim = 28;
    for(int i = 0; i < dim; ++i) {
        for(int j = 0; j < dim; ++j) {
            std::cout<<layer0.getOutput()[j + i * dim]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    std::cout<<"pooling output"<<std::endl;
    int dim2 = 14;
    for(int i = 0; i < dim2; ++i) {
        for(int j = 0; j < dim2; ++j) {
            std::cout<<layer1.getOutput()[j + i * dim2]<<" ";
        }
        std::cout<<std::endl;
    }


//    int count = 0;
//    /// test
//    for(int i = 0; i < testImage.size(); ++i) {
//        for(int j = 0; j < 784; ++j)
//            input[j] = testImage[i][j];

//        layer0.feedForward(input,&ileng);
//        layer1.feedForward();
//        layer2.feedForward();
//        layer3.feedForward();

//        int max = -1;
//        double maxval = 0.0;
//        for(int j = 0; j < 10; ++j) {
//            if(maxval < layer3.getOutput()[j]) {
//                maxval = layer3.getOutput()[j];
//                max = j;
//            }
//        }

//        std::cout<<max<<" "<<(int)testLabel[i];
//        if(max == (int)testLabel[i]) {
//            std::cout<<"  Correct"<<std::endl;
//            count++;
//        }
//        else                 std::cout<<"  FAIL"<<std::endl;
//    }

//    std::cout<<"Accuracy = "<<(double)count / testImage.size() * 100.0<<"%"<<std::endl;



//    std::cout<<"MNIST test done"<<std::endl;
}

int main(int argc, char *argv[])
{

    __cilkrts_set_param("nworkers","4");
    std::cout<<"Hello Neural Network!"<<std::endl;

//    /// test Train
//    {
//        trainTest();
//    }

//    /// test multi layer train
//    {
//        multiLayerTrain();
//    }

//    /// test MNIST
//    {
//        mnistTest();
//    }

    /// test CNN
    {
        MNISTTestConv();
    }

    return 0;
}
