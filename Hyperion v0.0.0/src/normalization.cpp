#include "../headers/normalization.h"
#include <random>
#include <sstream>

	normalization::normalization(int nodeCount) : layer(nodeCount, ActivationFunction::None) {
		// No additional parameters are required for a normalization layer.
	}

	std::vector<double> normalization::forward(const std::vector<double>& input) {
		// Implement the forward pass for the normalization layer
		std::vector<double> output(input.size(), 0.0);

		// Calculate the mean and standard deviation of the input
		double mean = 0.0;
		double variance = 0.0;

		for (int i = 0; i < input.size(); ++i) {
			mean += input[i];
		}

		mean /= input.size();

		for (int i = 0; i < input.size(); ++i) {
			variance += (input[i] - mean) * (input[i] - mean);
		}

		variance /= input.size();

		// Normalize the input
		for (int i = 0; i < input.size(); ++i) {
			output[i] = (input[i] - mean) / sqrt(variance);
		}

		return output;
	}

	std::vector<double> normalization::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)  {
		// Implement the backward pass for the normalization layer
		// Since normalization is stateless, the backward pass is straightforward
		std::vector<double> inputGradient(input.size(), 0.0);

		for (int i = 0; i < input.size(); ++i) {
			inputGradient[i] = outputGradient[i]; // Pass gradients through unchanged
		}

		return inputGradient;
	}

	std::string normalization::getData() {
		// Serialize layer data into a string
		std::string data = "normalization|";

		// Add input size to the data
		data += std::to_string(nodeCount) + "|";

		return data;
	}

	void normalization::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "normalization") {
			// Deserialize input size
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			nodeCount = std::stoi(sizeStr);
		}
	}