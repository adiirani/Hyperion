#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "baseLayerTemplate.h"


class residual : public layer {
public:
    residual(int inputSize, ActivationFunction activationFunc);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) override;
    std::string getData() override;
    void setData(std::string token) override;
};

#endif