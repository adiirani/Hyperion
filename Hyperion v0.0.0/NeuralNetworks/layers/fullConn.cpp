#include "./headers/fullConn.h"
#include <random>
#include <sstream>


	

	std::vector<double> fullConn::forward(const std::vector<double>& input) {
		std::vector<double> output(outputSize, 0.0);

		for (int j = 0; j < outputSize; ++j) {
			for (int i = 0; i < inputSize; ++i) {
				output[j] += input[i] * weights[i][j];
			}
			output[j] += biases[j];
		}

		return activation(activFunction, output);
	}

	std::vector<double> fullConn::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) {
		std::vector<double> inputGradient(inputSize, 0.0);

		// Compute gradients for weights and biases.
		for (int j = 0; j < outputSize; ++j) {
			for (int i = 0; i < inputSize; ++i) {
				inputGradient[i] += weights[i][j] * outputGradient[j];
				weights[i][j] -= learningRate * input[i] * outputGradient[j];
			}
			biases[j] -= learningRate * outputGradient[j];
		}

		inputGradient = backtivation(activFunction, inputGradient); // Apply backtivation
		return inputGradient;
	}

	std::string fullConn::getData() {
		// Serialize layer data into a string
		std::string data = "fullConn|";

		// Add input size and output size to the data
		data += std::to_string(inputSize) + "," + std::to_string(outputSize) + "|";

		// Add activation function to the data
		data += std::to_string(static_cast<int>(activFunction)) + "|";

		// Serialize weights
		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < outputSize; ++j) {
				data += std::to_string(weights[i][j]) + ",";
			}
		}
		data += "|";

		// Serialize biases
		for (int j = 0; j < outputSize; ++j) {
			data += std::to_string(biases[j]) + ",";
		}

		return data;
	}

	void fullConn::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "fullConn") {
			// Deserialize input size and output size
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			std::istringstream sizeStream(sizeStr);
			std::string inputSizeStr, outputSizeStr;
			getline(sizeStream, inputSizeStr, ',');
			getline(sizeStream, outputSizeStr, ',');
			inputSize = std::stoi(inputSizeStr);
			outputSize = std::stoi(outputSizeStr);

			// Deserialize activation function
			std::string activationFuncStr;
			getline(dataStream, activationFuncStr, '|');
			activFunction = static_cast<ActivationFunction>(std::stoi(activationFuncStr));

			// Deserialize weights
			for (int i = 0; i < inputSize; ++i) {
				for (int j = 0; j < outputSize; ++j) {
					std::string weightStr;
					getline(dataStream, weightStr, ',');
					weights[i][j] = std::stod(weightStr);
				}
			}

			// Deserialize biases
			for (int j = 0; j < outputSize; ++j) {
				std::string biasStr;
				getline(dataStream, biasStr, ',');
				biases[j] = std::stod(biasStr);
			}
		}
	}