#ifndef POOLING_H
#define POOLING_H

#include "baseLayerTemplate.h"

class pooling : public layer {
private:
	int inputSize;
	int filterSize;
	int stride;
	std::vector<std::vector<double>> output;
	std::vector<std::vector<int>> maxIndices; // To store the indices of max values during forward pass

public:
	pooling(int inputSize, int filterSize, int stride);

	std::vector<double> forward(const std::vector<double>& input) override;

	std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) override;

	std::string getData() override;

	void setData(std::string token) override;


};

#endif // !POOLING_H
