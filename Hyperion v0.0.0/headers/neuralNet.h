#ifndef NEURALNET_H
#define NEURALNET_H

#include "ActivationFunction.h"
#include "LayerType.h"
#include "LossFunction.h"
#include "attentive.h"
#include "convo.h"
#include "dropout.h"
#include "embedding.h"
#include "fullConn.h"
#include "normalization.h"
#include "pooling.h"
#include "recurrent.h"
#include "residual.h"
#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>
#include <random>
#include <ostream>
#include <memory>
#include <utility>
#include <sstream>


class neuralNet {
private:
    std::vector<std::unique_ptr<layer>> layers;

    std::vector<double> forwardPropagation(const std::vector<double>& input);
    std::vector<double> backwardPropagation(const std::vector<double>& target, const std::vector<double>& outputGradient, double learningRate);

public:
    neuralNet();

    void addLayer(int inSize, int outSize, LayerType type, ActivationFunction activationType, int numFilters = -1, int kernelFilterSize = -1, int step = -1, double dropoutRate = -1.0);
    void trainTest(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& labels, double split, LossFunction lossType);
    void save(const std::string& filename);
    void load(const std::string& filename);
};

#endif