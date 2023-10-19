#include "./headers/convo.h"

#include <random>
#include <sstream>


class convo : public layer {
private:
	int kernelSize;
	int numFilters;
	int stride;
	std::vector<double> filters; // 1D vector to store filters. Auto flattened to allow for MAXIMUM HACKABILITY

public:
	convo(int inputSize, int numFilters, int kernelSize, int stride, ActivationFunction activationFunc)
		: layer(inputSize, numFilters, activationFunc), kernelSize(kernelSize), numFilters(numFilters), stride(stride) {
		int filterSize = kernelSize * kernelSize * inputSize;
		filters.resize(numFilters * filterSize);

		// Initialize filters with random values
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dist(0.0, 1.0);
		for (int i = 0; i < numFilters * filterSize; ++i) {
			filters[i] = dist(gen);
		}
	}
	//TODO: OPTIMISE THIS PART OF THE CODE.
	// highly doubt you need this many for loops for a simple dot product operation.
	// god knows what the time complexity on this is
	// O(n!*n)?
	// Forward pass
	std::vector<double> forward(const std::vector<double>& input) override {
		int inputDepth = input.size();
		int outputSize = (inputSize - kernelSize) / stride + 1;
		std::vector<double> output(numFilters * outputSize * outputSize, 0.0);

		for (int f = 0; f < numFilters; ++f) { //iterates through filters
			for (int x = 0; x < outputSize; x += stride) { //moves across x of image
				for (int y = 0; y < outputSize; y += stride) { //moves across y of image
					int filterOffset = f * (kernelSize * kernelSize * inputDepth);
					for (int i = 0; i < kernelSize; ++i) { //iterates through kernel x
						for (int j = 0; j < kernelSize; ++j) { //iterates through kernel y
							for (int d = 0; d < inputDepth; ++d) { //iterates through depths
								int inputOffset = d * inputSize * inputSize + (x + i) * inputSize + y + j;
								int outputOffset = f * outputSize * outputSize + x * outputSize + y;
								output[outputOffset] += filters[filterOffset + (i * kernelSize * inputDepth) + (j * inputDepth) + d] * input[inputOffset];
							}
						}
					}
				}
			}
		}

		return activation(activFunction, output);
	}

	// Backward pass
	std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)override {
		int inputDepth = input.size();
		int outputSize = (inputSize - kernelSize) / stride + 1;
		std::vector<double> inputGradient(inputDepth * inputSize * inputSize, 0.0);


		//basically same stuff as forward pass, just backpropagated.
		for (int f = 0; f < numFilters; ++f) {
			for (int x = 0; x < outputSize; x += stride) {
				for (int y = 0; y < outputSize; y += stride) {
					int filterOffset = f * (kernelSize * kernelSize * inputDepth);
					for (int i = 0; i < kernelSize; ++i) {
						for (int j = 0; j < kernelSize; ++j) {
							for (int d = 0; d < inputDepth; ++d) {
								int outputOffset = f * outputSize * outputSize + x * outputSize + y;
								int inputOffset = d * inputSize * inputSize + (x + i) * inputSize + y + j;
								for (int dx = 0; dx < outputSize; ++dx) {
									for (int dy = 0; dy < outputSize; ++dy) {
										inputGradient[inputOffset] += filters[filterOffset + (i * kernelSize * inputDepth) + (j * inputDepth) + d] * outputGradient[outputOffset + dx * outputSize + dy];
										filters[filterOffset + (i * kernelSize * inputDepth) + (j * inputDepth) + d] -= learningRate * input[inputOffset] * outputGradient[outputOffset + dx * outputSize + dy];
									}
								}
							}
						}
					}
				}
			}
		}

		inputGradient = backtivation(activFunction, inputGradient); // Apply backtivation
		return inputGradient;
	}

	std::string getData() override {
		// Serialize layer data into a string
		std::string data = "convo|";

		// Add input size, numFilters, kernelSize, and stride to the data
		data += std::to_string(inputSize) + "," + std::to_string(numFilters) + "," + std::to_string(kernelSize) + "," + std::to_string(stride) + "|";

		// Add activation function to the data
		data += std::to_string(static_cast<int>(activFunction)) + "|";

		// Serialize filters
		for (double filterValue : filters) {
			data += std::to_string(filterValue) + ",";
		}

		return data;
	}

	void setData(std::string token) override {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "convo") {
			// Deserialize input size, numFilters, kernelSize, and stride
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			std::istringstream sizeStream(sizeStr);
			std::string inputSizeStr, numFiltersStr, kernelSizeStr, strideStr;
			getline(sizeStream, inputSizeStr, ',');
			getline(sizeStream, numFiltersStr, ',');
			getline(sizeStream, kernelSizeStr, ',');
			getline(sizeStream, strideStr, '|');
			inputSize = std::stoi(inputSizeStr);
			numFilters = std::stoi(numFiltersStr);
			kernelSize = std::stoi(kernelSizeStr);
			stride = std::stoi(strideStr);

			// Deserialize activation function
			std::string activationFuncStr;
			getline(dataStream, activationFuncStr, '|');
			activFunction = static_cast<ActivationFunction>(std::stoi(activationFuncStr));

			// Deserialize filters
			std::string filterValuesStr;
			getline(dataStream, filterValuesStr, '|');
			std::istringstream filterValuesStream(filterValuesStr);
			filters.clear();
			std::string filterValueStr;
			while (getline(filterValuesStream, filterValueStr, ',')) {
				filters.push_back(std::stod(filterValueStr));
			}
		}
	}