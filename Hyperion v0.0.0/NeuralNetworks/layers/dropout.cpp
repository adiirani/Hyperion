#include "./headers/dropout.h"
#include <random>
#include <sstream>

	dropout::dropout(int inputSize, double dropoutRate) : layer(inputSize, inputSize, ActivationFunction::None), dropoutRate(dropoutRate) {
		// Initialize the dropout mask
		dropoutMask.resize(inputSize, true);
	}

	std::vector<double> dropout::forward(const std::vector<double>& input)  {
		// Implement the forward pass for the dropout layer
		std::vector<double> output(input.size(), 0.0);

		// Generate a random mask for dropout
		for (int i = 0; i < input.size(); ++i) {
			dropoutMask[i] = (rand() / double(RAND_MAX)) >= dropoutRate;
		}

		// Apply dropout to the input
		for (int i = 0; i < input.size(); ++i) {
			if (dropoutMask[i]) {
				output[i] = input[i] / (1.0 - dropoutRate);
			}
			else {
				output[i] = 0.0;
			}
		}

		return output;
	}

	std::vector<double> dropout::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)  {
		// Implement the backward pass for the dropout layer
		std::vector<double> inputGradient(input.size(), 0.0);

		// Apply dropout mask to the gradients
		for (int i = 0; i < input.size(); ++i) {
			if (dropoutMask[i]) {
				inputGradient[i] = outputGradient[i] / (1.0 - dropoutRate);
			}
			else {
				inputGradient[i] = 0.0;
			}
		}

		return inputGradient;
	}

	std::string dropout::getData()  {
		// Serialize layer data into a string
		std::string data = "dropout|";

		// Add input size and dropout rate to the data
		data += std::to_string(inputSize) + "|";
		data += std::to_string(dropoutRate) + "|";

		// Serialize the dropout mask
		for (bool mask : dropoutMask) {
			data += mask ? "1|" : "0|";
		}

		return data;
	}

	void dropout::setData(std::string token)  {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "dropout") {
			// Deserialize input size
			std::string sizeStr;
			getline(dataStream, sizeStr, '|');
			inputSize = std::stoi(sizeStr);
			outputSize = inputSize; // Output size is the same as input size

			// Deserialize dropout rate
			std::string rateStr;
			getline(dataStream, rateStr, '|');
			dropoutRate = std::stod(rateStr);

			// Deserialize the dropout mask
			dropoutMask.resize(inputSize);
			for (int i = 0; i < inputSize; ++i) {
				std::string maskStr;
				getline(dataStream, maskStr, '|');
				dropoutMask[i] = (maskStr == "1");
			}
		}
	}