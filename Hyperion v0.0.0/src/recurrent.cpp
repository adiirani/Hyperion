#include "../headers/recurrent.h"
#include <random>
#include <sstream>


	recurrent::recurrent(int nodeCount, ActivationFunction activationFunc) : layer(nodeCount,activationFunc) {
		// Initialize the memory state
		state.resize(nodeCount, 0.0);
	}

	std::vector<double> recurrent::forward(const std::vector<double>& input) {
		// Implement the forward pass for the recurrent layer
		std::vector<double> output(nodeCount, 0.0);

		for (int j = 0; j < nodeCount; ++j) {
			for (int i = 0; i < nodeCount; ++i) {
				output[j] += input[i] * weights[i][j];
			}
			output[j] += biases[j];

			// Update the memory state
			state[j] = output[j];
		}

		return activation(activFunction, output);
	}

	std::vector<double> recurrent::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) {
		// Implement the backward pass for the recurrent layer
		std::vector<double> inputGradient(nodeCount, 0.0);

		// Compute gradients for weights and biases
		for (int j = 0; j < nodeCount; ++j) {
			for (int i = 0; i < nodeCount; ++i) {
				inputGradient[i] += weights[i][j] * outputGradient[j];
				weights[i][j] -= learningRate * input[i] * outputGradient[j];
			}
			biases[j] -= learningRate * outputGradient[j];
		}

		inputGradient = backtivation(activFunction, inputGradient); // Apply backtivation

		// Update the memory state in the backward pass
		state = inputGradient;

		return inputGradient;
	}

	std::string recurrent::getData() {
		// Serialize layer data into a string
		std::string data = "recurrent|";

		// Add input size, output size, and activation function to the data
		data += std::to_string(nodeCount) +  "|" + std::to_string(activFunction) + "|";

		// Serialize the memory state
		for (double value : state) {
			data += std::to_string(value) + ",";
		}
		data.pop_back(); // Remove the trailing comma
		data += "|";

		return data;
	}

	void recurrent::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "recurrent") {
			// Deserialize input size, output size, and activation function
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			std::istringstream sizeStream(sizeStr);
			std::string nodeCountStr, activationFuncStr;
			getline(sizeStream, nodeCountStr, '|');
			getline(sizeStream, activationFuncStr, '|');
			nodeCount = std::stoi(nodeCountStr);
			activFunction = static_cast<ActivationFunction>(std::stoi(activationFuncStr));

			// Deserialize the memory state
			std::string stateStr;
			getline(dataStream, stateStr, '|');
			std::istringstream stateStream(stateStr);
			for (int i = 0; i < nodeCount; ++i) {
				std::string valueStr;
				getline(stateStream, valueStr, ',');
				state[i] = std::stod(valueStr);
			}
		}
	}