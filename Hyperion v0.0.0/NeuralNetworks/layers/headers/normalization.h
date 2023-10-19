#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "baseLayerTemplate.h"

class normalization : public layer {
public:
    normalization(int inputSize);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) override;
    std::string getData() override;
    void setData(std::string token) override;
};

#endif