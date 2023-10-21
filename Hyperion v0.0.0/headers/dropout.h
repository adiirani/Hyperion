#ifndef DROPOUT_H
#define DROPOUT_H

#include "baseLayerTemplate.h"

class dropout : public layer {
private:
    double dropoutRate;
    std::vector<bool> dropoutMask;

public:
    dropout(int inputSize, double dropoutRate);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) override;
    std::string getData() override;
    void setData(std::string token) override;
};

#endif