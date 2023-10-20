#include "./headers/attentive.h"
#include <random>
#include <sstream>


	attentive::attentive(int inputSize, ActivationFunction activationFunc) : layer(inputSize * 2, inputSize, activationFunc) {
		inputSize = inputSize;
		activFunction = activationFunc;

		// Initialize query and key weights
		queryWeights.resize(inputSize, std::vector<double>(inputSize, 0.0));
		keyWeights.resize(inputSize, std::vector<double>(inputSize, 0.0));

		// Initialize the weights with random values
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dist(0.0, 1.0);

		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < inputSize; ++j) {
				queryWeights[i][j] = dist(gen);
				keyWeights[i][j] = dist(gen);
			}
		}
	}

	std::vector<double> attentive::forward(const std::vector<double>& input) {
		// Implement the forward pass for the attentive layer
		std::vector<double> query(inputSize, 0.0);
		std::vector<double> key(inputSize, 0.0);

		// Split the input into query and key parts
		for (int i = 0; i < inputSize; ++i) {
			query[i] = input[i];
			key[i] = input[inputSize + i];
		}

		std::vector<double> output(inputSize, 0.0);

		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < inputSize; ++j) {
				output[i] += query[i] * queryWeights[i][j] + key[i] * keyWeights[i][j];
			}
		}

		return activation(activFunction, output);
	}


	std::vector<double> attentive::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)  {
		// Implement the backward pass for the attentive layer
		std::vector<double> query(inputSize, 0.0);
		std::vector<double> key(inputSize, 0.0);

		// Split the input into query and key parts
		for (int i = 0; i < inputSize; ++i) {
			query[i] = input[i];
			key[i] = input[inputSize + i];
		}

		std::vector<double> queryGradient(inputSize, 0.0);
		std::vector<double> keyGradient(inputSize, 0.0);

		// Compute gradients for query and key weights
		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < inputSize; ++j) {
				queryGradient[i] += queryWeights[i][j] * outputGradient[i];
				keyGradient[i] += keyWeights[i][j] * outputGradient[i];

				queryWeights[i][j] -= learningRate * query[i] * outputGradient[i];
				keyWeights[i][j] -= learningRate * key[i] * outputGradient[i];
			}
		}

		// Combine query and key gradients
		std::vector<double> inputGradient(inputSize * 2, 0.0);
		for (int i = 0; i < inputSize; ++i) {
			inputGradient[i] = queryGradient[i];
			inputGradient[inputSize + i] = keyGradient[i];
		}

		inputGradient = backtivation(activFunction, inputGradient); // Apply backtivation
		return inputGradient;
	}

	std::string attentive::getData()  {
		// Serialize layer data into a string
		std::string data = "attentive|";

		// Add input size to the data
		data += std::to_string(inputSize) + "|";

		// Add activation function to the data
		data += std::to_string(static_cast<int>(activFunction)) + "|";

		// Serialize the query weights
		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < inputSize; ++j) {
				data += std::to_string(queryWeights[i][j]) + "|";
			}
		}

		// Serialize the key weights
		for (int i = 0; i < inputSize; ++i) {
			for (int j = 0; j < inputSize; ++j) {
				data += std::to_string(keyWeights[i][j]) + "|";
			}
		}

		return data;
	}

	void attentive::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "attentive") {
			// Deserialize input size
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			inputSize = std::stoi(sizeStr);

			// Deserialize activation function
			std::string activationStr;
			getline(dataStream, activationStr, '|');
			activFunction = static_cast<ActivationFunction>(std::stoi(activationStr));

			// Deserialize the query weights
			queryWeights.resize(inputSize, std::vector<double>(inputSize, 0.0));
			for (int i = 0; i < inputSize; ++i) {
				for (int j = 0; j < inputSize; ++j) {
					std::string weightStr;
					getline(dataStream, weightStr, '|');
					queryWeights[i][j] = std::stod(weightStr);
				}
			}

			// Deserialize the key weights
			keyWeights.resize(inputSize, std::vector<double>(inputSize, 0.0));
			for (int i = 0; i < inputSize; ++i) {
				for (int j = 0; j < inputSize; ++j) {
					std::string weightStr;
					getline(dataStream, weightStr, '|');
					keyWeights[i][j] = std::stod(weightStr);
				}
			}
		}
	}