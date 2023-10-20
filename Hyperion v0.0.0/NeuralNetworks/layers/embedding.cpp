#include "./headers/embedding.h"
#include <random>
#include <sstream>


	embedding::embedding(int inputSize, int outputSize) : layer(inputSize, outputSize, ActivationFunction::None) {
		// Initialize embedding weights
		embeddingWeights.resize(inputSize, std::vector<double>(outputSize));

		// Initialize the embedding weights with random values
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dist(0.0, 1.0);

		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < outputSize; ++j) {
				embeddingWeights[i][j] = dist(gen);
			}
		}
	}

	std::vector<double> embedding::forward(const std::vector<double>& input) {
		// Implement the forward pass for the embedding layer
		std::vector<double> output(outputSize, 0.0);

		for (int j = 0; j < outputSize; ++j) {
			for (int i = 0; i < inputSize; ++i) {
				output[j] += input[i] * embeddingWeights[i][j];
			}
		}

		return output;
	}

	std::vector<double> embedding::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) {
		// Implement the backward pass for the embedding layer
		std::vector<double> inputGradient(inputSize, 0.0);

		// Compute gradients for embedding weights
		for (int j = 0; j < outputSize; ++j) {
			for (int i = 0; i < inputSize; ++i) {
				inputGradient[i] += embeddingWeights[i][j] * outputGradient[j];
				embeddingWeights[i][j] -= learningRate * input[i] * outputGradient[j];
			}
		}

		return inputGradient;
	}

	std::string embedding::getData() {
		// Serialize layer data into a string
		std::string data = "embedding|";

		// Add input size and output size to the data
		data += std::to_string(inputSize) + "," + std::to_string(outputSize) + "|";

		// Serialize the embedding weights
		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < outputSize; ++j) {
				data += std::to_string(embeddingWeights[i][j]) + ",";
			}
		}
		data.pop_back(); // Remove the trailing comma
		data += "|";

		return data;
	}

	void embedding::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "embedding") {
			// Deserialize input size and output size
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			std::istringstream sizeStream(sizeStr);
			std::string inputSizeStr, outputSizeStr;
			getline(sizeStream, inputSizeStr, ',');
			getline(sizeStream, outputSizeStr, ',');
			inputSize = std::stoi(inputSizeStr);
			outputSize = std::stoi(outputSizeStr);

			// Deserialize the embedding weights
			std::string weightsStr;
			getline(dataStream, weightsStr, '|');
			std::istringstream weightsStream(weightsStr);
			for (int i = 0; i < inputSize; ++i) {
				for (int j = 0; j < outputSize; ++j) {
					std::string weightStr;
					getline(weightsStream, weightStr, ',');
					embeddingWeights[i][j] = std::stod(weightStr);
				}
			}
		}
	}