#ifndef CONVO_H
#define CONVO_H

#include "baseLayerTemplate.h"

class convo : public layer {
private:
	int kernelSize;
	int numFilters;
	int filterSize;
	int stride;
	int xSize;
	int ySize;
	int zSize;
	std::vector<std::vector<std::vector<double>>> filters; // 3D vector to store filters.

public:
	convo(int nodeCount, int NumFilters, int KernelSize, int Stride, int XSize, int YSize,int ZSize, ActivationFunction activationFunc);

	std::vector<double> forward(const std::vector<double>& input) override;

	std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)override;

	std::string getData() override;

	void setData(std::string token) override;

};
#endif
