#include "../headers/residual.h"
#include <random>
#include <sstream>


	residual::residual(int inputSize, ActivationFunction activationFunc) : layer(inputSize, inputSize, activationFunc) {
		// No additional parameters are required for a residual layer.
	}

	std::vector<double> residual::forward(const std::vector<double>& input) {
		// Implement the forward pass for the residual layer
		std::vector<double> output(input.size(), 0.0);

		for (int i = 0; i < input.size(); ++i) {
			output[i] = input[i] + input[i]; // Adding input to the output
		}

		return activation(activFunction, output);
	}

	std::vector<double> residual::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)  {
		// Implement the backward pass for the residual layer
		std::vector<double> inputGradient(input.size(), 0.0);

		for (int i = 0; i < input.size(); ++i) {
			// Pass gradients through unchanged
			inputGradient[i] = outputGradient[i];
		}

		return inputGradient;
	}

	std::string residual::getData() {
		// Serialize layer data into a string
		std::string data = "residual|";

		// Add input size to the data
		data += std::to_string(inputSize) + "|";

		// Add activation function to the data
		data += std::to_string(static_cast<int>(activFunction)) + "|";

		return data;
	}

	void residual::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "residual") {
			// Deserialize input size
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			inputSize = std::stoi(sizeStr);

			// Deserialize activation function
			std::string activationStr;
			getline(dataStream, activationStr, '|');
			activFunction = static_cast<ActivationFunction>(std::stoi(activationStr));
		}
	}