#include "../headers/dropout.h"
#include <random>
#include <sstream>

	dropout::dropout(int nodeCount, double dropoutRate) : layer(nodeCount, ActivationFunction::None), dropoutRate(dropoutRate) {
		// Initialize the dropout mask
		dropoutMask.resize(nodeCount, true);
	}

	std::vector<double> dropout::forward(const std::vector<double>& input)  { 
		// Implement the forward pass for the dropout layer
		std::vector<double> output(input.size(), 0.0);

		// Generate a random mask for dropout
		for (int i = 0; i < input.size(); ++i) {
			output[i] = (((rand() / double(RAND_MAX)) >= dropoutRate) & 1) * input[i] / (1.0 - dropoutRate); //not john carmack level but it will suffice
			dropoutMask[i] = output[i] > 0; //this is a surprise tool that will help us later
		}


		return output;
	}


	//backward pass
	std::vector<double> dropout::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate)  {
		// Implement the backward pass for the dropout layer
		std::vector<double> inputGradient(input.size(), 0.0);

		// Apply dropout mask to the gradients
		for (int i = 0; i < input.size(); ++i) {
			inputGradient[i] = (dropoutMask[i] & 1) * (outputGradient[i] / (1.0 - dropoutRate)); //even more bithax
		}

		
		return inputGradient;
	}

	std::string dropout::getData()  {
		// Serialize layer data into a string
		std::string data = "dropout|";

		// Add input size and dropout rate to the data
		data += std::to_string(nodeCount) + "|";
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
			nodeCount = std::stoi(sizeStr);

			// Deserialize dropout rate
			std::string rateStr;
			getline(dataStream, rateStr, '|');
			dropoutRate = std::stod(rateStr);

			// Deserialize the dropout mask
			dropoutMask.resize(nodeCount);
			for (int i = 0; i < nodeCount; ++i) {
				std::string maskStr;
				getline(dataStream, maskStr, '|');
				dropoutMask[i] = (maskStr == "1");
			}
		}
	}