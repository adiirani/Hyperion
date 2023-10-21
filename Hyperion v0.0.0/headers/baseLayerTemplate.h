#ifndef BASELAYERTEMPLATE_H
#define BASELAYERTEMPLATE_H


#include "ActivationFunction.h"
#include <vector>
#include <string>


class layer {
protected:
    int inputSize;
    int outputSize;
    ActivationFunction activFunction;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    std::vector<double> activation(ActivationFunction activationType, std::vector<double>& input);

    std::vector<double> backtivation(ActivationFunction activationType, std::vector<double>& output);

public:
    // Constructor
    layer(int inputSz, int outputSz, ActivationFunction activationFunc);

    // Virtual methods to be implemented by subclasses
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) = 0;
    virtual std::string getData() = 0;
    virtual void setData(std::string token) = 0;
};


#endif