#ifndef RECURRENT_H
#define RECURRENT_H

#include "baseLayerTemplate.h"

class recurrent : public layer {
private:
    std::vector<double> state; // Memory state of the recurrent layer

public:
    recurrent(int inputSize, int outputSize, ActivationFunction activationFunc);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) override;
    std::string getData() override;
    void setData(std::string token) override;
};

#endif