#ifndef CONVO_H
#define CONVO_H

#include "baseLayerTemplate.h"

class convo : public layer {
private:
	int kernelSize;
	int numFilters;
	int stride;
	std::vector<double> filters; // 1D vector to store filters. Auto flattened to allow for MAXIMUM HACKABILITY

public:
	convo(int inputSize, int numFilters, int kernelSize, int stride, ActivationFunction activationFunc);

	std::vector<double> forward(const std::vector<double>& input) override;

	std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)override;

	std::string getData() override;

	void setData(std::string token) override;

}
#endif
