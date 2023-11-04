#ifndef ATTENTIVE_H
#define ATTENTIVE_H

#include "baseLayerTemplate.h"


class attentive : public layer {
private:
    std::vector<std::vector<double>> queryWeights;
    std::vector<std::vector<double>> keyWeights;

public:
    attentive(int nodeCount, ActivationFunction activationFunc);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) override;
    std::string getData() override;
    void setData(std::string token) override;
};

#endif