#ifndef NEURALNET_H
#define NEURALNET_H

#include "../../enumsAndLoss/headers/ActivationFunctions.h"
#include "../../enumsAndLoss/headers/LayerType.h"
#include "../../enumsAndLoss/headers/LossFunction.h"
#include "../../layers/headers/attentive.h"
#include "../../layers/headers/convo.h"
#include "../../layers/headers/dropout.h"
#include "../../layers/headers/embedding.h"
#include "../../layers/headers/fullConn.h"
#include "../../layers/headers/normalization.h"
#include "../../layers/headers/pooling.h"
#include "../../layers/headers/recurrent.h"
#include "../../layers/headers/residual.h"
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
    neuralNet(LossFunction lossType);

    void addLayer(int inSize, int outSize, LayerType type, ActivationFunction activationType, int numFilters = -1, int kernelFilterSize = -1, int step = -1, double dropoutRate = -1.0);
    void trainTest(const std::vector<std::vector<double>>& data, double split, LossFunction lossType);
    void save(const std::string& filename);
    void load(const std::string& filename);
};

#endif