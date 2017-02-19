#include <iostream>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "include/Layer.hpp"
#include "include/ActivationFunction.hpp"
#include "include/MNIST/mnist_reader_less.hpp"
#include "include/LossFunctions.hpp"
#include "include/GradientDescent.hpp"



using namespace std;
using namespace DeepLearning;

double function2(int N, double* x) {
    return x[0] * x[0] + x[1] * x[1];
}


int main(int argc, char *argv[])
{
    DeepLearning::Layer<DeepLearning::Sigmoid> layer1;
    DeepLearning::Layer<DeepLearning::Sigmoid> layer2;
    DeepLearning::Layer<DeepLearning::Naive>   layer3;

    std::cout<<"Hello Neural Network!"<<std::endl;

    layer1.init(2,3);
    layer2.init(3,2);
    layer3.init(2,2);

    double input[2] = {1.0, 0.5};

    /// init layer1
    {
        DeepLearning::Layer<DeepLearning::Naive> layer;
        layer.init(2,3);
        double input[2] = {1, 2};
        rMat& weight = layer.getWeight();
        weight[0][0] = 1;   weight[0][1] = 3;   weight[0][2] = 5;
        weight[1][0] = 2;   weight[1][1] = 4;   weight[1][2] = 6;
        layer.getBias()[0] = 0.0;
        layer.getBias()[1] = 0.0;
        layer.getBias()[2] = 0.0;
        layer.feedForward(input,2);
        rVec& output = layer.getOutput();
        std::cout<<output[0]<<" "<<output[1]<<" "<<output[2]<<std::endl;
    }

    /// test softMax
    {
        double a[3] = { 0.3, 2.9, 4.0 };
        double b[3];
        SoftMax::getSoftMax(3,a,b);
        std::cout<<"---- softmax"<<std::endl;
        std::cout<<b[0]<<" "<<b[1]<<" "<<b[2]<<std::endl;
        std::cout<<"sum = "<<b[0] + b[1] + b[2]<<std::endl;
    }

    /// init layer1
    {
        rMat& weight = layer1.getWeight();
        weight[0][0] = 0.1;   weight[0][1] = 0.3;   weight[0][2] = 0.5;
        weight[1][0] = 0.2;   weight[1][1] = 0.4;   weight[1][2] = 0.6;
        layer1.getBias()[0] = 0.1;
        layer1.getBias()[1] = 0.2;
        layer1.getBias()[2] = 0.3;
    }

    /// init layer2
    {
        rMat& weight = layer2.getWeight();
        weight[0][0] = 0.1;   weight[0][1] = 0.4;
        weight[1][0] = 0.2;   weight[1][1] = 0.5;
        weight[2][0] = 0.3;   weight[2][1] = 0.6;
        layer2.getBias()[0] = 0.1;
        layer2.getBias()[1] = 0.2;
    }

    /// init layer3
    {
        rMat& weight = layer3.getWeight();
        weight[0][0] = 0.1;   weight[0][1] = 0.3;
        weight[1][0] = 0.2;   weight[1][1] = 0.4;
        layer3.getBias()[0] = 0.1;
        layer3.getBias()[1] = 0.2;
    }

    /// test MSE
    {
        double t[10] = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
        double y0[10] = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0};
        double y1[10] = { 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0};
        std::cout<<"MSE = "<<MSE(10,y0,t)<<std::endl;
        std::cout<<"MSE = "<<MSE(10,y1,t)<<std::endl;
    }

    /// test CEE
    {
        double t[10] = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
        double y0[10] = { 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0};
        double y1[10] = { 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0};
        std::cout<<"CEE = "<<CEE(10,y0,t)<<std::endl;
        std::cout<<"CEE = "<<CEE(10,y1,t)<<std::endl;
    }

    /// test gradient_numerical
    {
        std::cout<<"TEST numerical Gradient"<<std::endl;
        double input[2];;
        double grad[2];
        input[0] = 3.0;
        input[1] = 4.0;
        numerical_Gradient(function2,2,1,input,grad);
        std::cout<<grad[0]<<" "<<grad[1]<<std::endl;
        input[0] = 0.0;
        input[1] = 2.0;
        numerical_Gradient(function2,2,1,input,grad);
        std::cout<<grad[0]<<" "<<grad[1]<<std::endl;
        input[0] = 3.0;
        input[1] = 0.0;
        numerical_Gradient(function2,2,1,input,grad);
        std::cout<<grad[0]<<" "<<grad[1]<<std::endl;
    }

    /// test Gradient Descent
    {
        double input[2] = {-3.0, 4.0};
        gradientDescent(function2,0.1,100,2,1,input);
        std::cout<<std::endl<<"Gradeint Descent Result = "<<input[0]<<" "<<input[1]<<std::endl<<std::endl;
    }

    /// test Gradient Descent
    {
        double input[2] = {-3.0, 4.0};
        gradientDescent(function2,10.0,100,2,1,input);
        std::cout<<std::endl<<"Gradeint Descent Result2 = "<<input[0]<<" "<<input[1]<<std::endl<<std::endl;
    }

    /// test Gradient Descent
    {
        double input[2] = {-3.0, 4.0};
        gradientDescent(function2,1e-10,100,2,1,input);
        std::cout<<std::endl<<"Gradeint Descent Result3 = "<<input[0]<<" "<<input[1]<<std::endl<<std::endl;
    }

    /// test a layer
    {
        double input[2] = { 0.6, 0.9 };
        double target0[3] = { 1, 0, 0 };
        double target1[3] = { 0, 1, 0 };
        double target2[3] = { 0, 0, 1 };
        std::cout<<"Test a Layer"<<std::endl;
        DeepLearning::Layer<DeepLearning::Naive> layer;
        layer.init(2,3);
        rMat& weights = layer.getWeight();
        std::cout<<weights[0][0]<<" "<<weights[0][1]<<" "<<weights[0][2]<<std::endl;
        std::cout<<weights[1][0]<<" "<<weights[1][1]<<" "<<weights[1][2]<<std::endl;
        layer.feedForward(input,2);
        rVec& output = layer.getOutput();
        std::cout<<"output = "<<output[0]<<" "<<output[1]<<" "<<output[2]<<std::endl;

        rVec softmax(3,0);
        SoftMax::getSoftMax(3,output.data(),softmax.data());
        std::cout<<"softmax \n ==> "<<softmax[0]<<" "<<softmax[1]<<" "<<softmax[2]
                 <<"\nSUM = "<<softmax[0] + softmax[1] + softmax[2]<<std::endl;
        std::cout<<"cross entropy error\n ==> "<<CEE(3,softmax.data(),target0)<<" "
                                           <<CEE(3,softmax.data(),target1)<<" "
                                           <<CEE(3,softmax.data(),target2)<<std::endl;
    }



    layer1.feedForward(input,2);
    layer2.feedForward(layer1.getOutput().data(),layer1.getOutput().size());
    layer3.feedForward(layer2.getOutput().data(),layer2.getOutput().size());

    std::cout<<"---- 3 layer neural network validation"<<std::endl;
    std::cout<<"output = "<<layer3.getOutput()[0]<<" "<<layer3.getOutput()[1]<<std::endl;

    return 0;
}
