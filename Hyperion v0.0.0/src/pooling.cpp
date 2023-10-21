#include "../headers/pooling.h"
#include <random>
#include <sstream>


	pooling::pooling(int inputSize, int filterSize, int stride) : layer(inputSize, inputSize, None), inputSize(inputSize), filterSize(filterSize), stride(stride) {
		// Initialize the output size based on input size and pooling parameters
		int outputSize = (inputSize - filterSize) / stride + 1;
		output.resize(outputSize, std::vector<double>(outputSize, 0.0));
		maxIndices.resize(outputSize, std::vector<int>(outputSize, 0));
	}

	// Forward pass
	std::vector<double> pooling::forward(const std::vector<double>& input)  {
		int maxPoolSize = filterSize * filterSize;
		std::vector<double> maxPooledValues;
		maxPooledValues.reserve(output.size() * output[0].size());

		for (int x = 0; x < output.size(); x++) {
			for (int y = 0; y < output.size(); y++) {
				double maxVal = -std::numeric_limits<double>::max();
				int maxX = -1;
				int maxY = -1;
				for (int i = 0; i < filterSize; i++) {
					for (int j = 0; j < filterSize; j++) {
						int inputIndex = (x * stride + i) * inputSize + (y * stride + j);
						double val = input[inputIndex];
						if (val > maxVal) {
							maxVal = val;
							maxX = x * stride + i;
							maxY = y * stride + j;
						}
					}
				}
				maxPooledValues.push_back(maxVal);
				maxIndices[x][y] = maxX * inputSize + maxY; // Store the indices of max values
			}
		}
		return maxPooledValues;
	}

	// Backward pass
	std::vector<double> pooling::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)  {
		std::vector<double> inputGradient(input.size(), 0.0);

		for (int x = 0; x < output.size(); x++) {
			for (int y = 0; y < output.size(); y++) {
				int maxIndex = maxIndices[x][y];
				inputGradient[maxIndex] += outputGradient[x * output.size() + y];
			}
		}

		return inputGradient;
	}

	std::string pooling::getData()  {
		// Serialize layer data into a string
		std::string data = "pooling|";

		// Add input size, filterSize, and stride to the data
		data += std::to_string(inputSize) + "," + std::to_string(filterSize) + "," + std::to_string(stride) + "|";

		return data;
	}

	void pooling::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "pooling") {
			// Deserialize input size, filterSize, and stride
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			std::istringstream sizeStream(sizeStr);
			std::string inputSizeStr, filterSizeStr, strideStr;
			getline(sizeStream, inputSizeStr, ',');
			getline(sizeStream, filterSizeStr, ',');
			getline(sizeStream, strideStr, '|');
			inputSize = std::stoi(inputSizeStr);
			filterSize = std::stoi(filterSizeStr);
			stride = std::stoi(strideStr);

			// Initialize the output size and maxIndices
			int outputSize = (inputSize - filterSize) / stride + 1;
			output.resize(outputSize, std::vector<double>(outputSize, 0.0));
			maxIndices.resize(outputSize, std::vector<int>(outputSize, 0));
		}
	}