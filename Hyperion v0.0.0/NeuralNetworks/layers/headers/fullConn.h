#ifndef FULLCONN_H
#define FULLCONN_H

#include "baseLayerTemplate.h"

class fullConn : public layer {
public:
	fullConn(int inputSz, int outputSz, ActivationFunction activationFunc) : layer(inputSz, outputSz, activationFunc);

	std::vector<double> forward(const std::vector<double>& input) override;

	std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) override;

	std::string getData() override;

	void setData(std::string token) override;

#endif // !FULLCONN_H
