#ifndef FULLCONN_H
#define FULLCONN_H

#include "baseLayerTemplate.h"

class fullConn : public layer {
public:
	fullConn(int inputSize, int outputSize, ActivationFunction activationFunc) : layer(inputSize, outputSize, activationFunc) {}

	std::vector<double> forward(const std::vector<double>& input) override;

	std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) override;

	std::string getData() override;

	void setData(std::string token) override;

};

#endif // !FULLCONN_H
