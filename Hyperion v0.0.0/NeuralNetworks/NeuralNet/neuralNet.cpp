#include "./headers/neuralNet.h"


/// <summary>
/// MAIN NEURAL NETWORK CLASS
/// Uses stack magic to build a neural network. Sort of like that cake game in purble place but actually useful.
/// USAGE !!!PAY ATTENTION DUBMSASES!!!
///		neuralNet myNeuralNetwork(<2d layers vector containing node count, layer type and activation function type>); Declares neural network, with set layers if specified
///		neuralNet.add(<node count>,<layer type>, <activation function type>); adds neural network layer
///		neuralNet.traintest(<data>,<split>); runs it on data from a custom data class TB specified. specify split in terms of val from 0 to 1, so like 0.5 for 50%
///		neuralNet.save(<filename>); saves Neural Network and destroys the object.
///		neuralNet.load(<filename>
///		neuralNet.destroy(); destroys Neural Network without saving (if you made skynet with this library and need to kill it use this, though i dont even understand why you would do so with such a shit library.)
/// </summary>

class neuralNet {
private:
	std::vector<std::unique_ptr<layer>> layers;

	std::vector<double> forwardPropagation(const std::vector<double>& input) {
		std::vector<double> layerInput = input;
		for (const auto& currLayer : layers) {
			layerInput = currLayer->forward(layerInput);
		}
		return layerInput;  // Return the final output.
	}

	std::vector<double> backwardPropagation(const std::vector<double>& target, const std::vector<double>& outputGradient, double learningRate) {
		int numLayers = layers.size();
		std::vector<double> layerError = outputGradient;
		for (int i = numLayers - 1; i >= 0; --i) {
			auto& currLayer = layers[i];
			layerError = currLayer->backward(target, layerError, learningRate);
		}
		return layerError;  // Return the error propagated to the input layer.
	}

public:
	neuralNet(LossFunction lossType) {
		std::cout << "initializing neural network.";

	}

	void addLayer(int inSize, int outSize, LayerType type, ActivationFunction activationType, int numFilters = -1, int kernelFilterSize = -1, int step = -1, double dropoutRate = -1.0) {
		std::unique_ptr<layer> currentLayer; // Declare a pointer to the base class.

		switch (type) {
		case FullConn:
			currentLayer = std::make_unique<fullConn>(inSize, outSize, activationType);
			break;
		case Convo:
			currentLayer = std::make_unique<convo>(inSize, numFilters, kernelFilterSize, step, activationType);
			break;
		case Pooling:
			currentLayer = std::make_unique<pooling>(inSize, kernelFilterSize, step);
			break;
		case Recurrent:
			currentLayer = std::make_unique<recurrent>(inSize, outSize, activationType);
			break;
		case Embedding:
			currentLayer = std::make_unique<embedding>(inSize, outSize);
			break;
		case Normalization:
			currentLayer = std::make_unique<normalization>(inSize);
			break;
		case Dropout:
			currentLayer = std::make_unique<dropout>(inSize, dropoutRate);
			break;
		case Attention:
			currentLayer = std::make_unique<attentive>(inSize, activationType);
			break;
		case Residual:
			currentLayer = std::make_unique<residual>(inSize, activationType);
			break;
		default:
			std::cerr << "Didn't specify a valid layer type." << std::endl;
			return; // Exit the function if the type is invalid.
		}


		layers.push_back(std::move(currentLayer));
	}

	void trainTest(const std::vector<std::vector<double>>& data, double split, LossFunction lossType) {
		if (split <= 0.0 || split >= 1.0) {
			throw std::invalid_argument("Invalid split value. The split value should be in the range (0, 1).");
		}

		// Split the data into training and testing sets
		int numDataPoints = data.size();
		int splitIndex = static_cast<int>(split * numDataPoints);
		std::vector<std::vector<double>> trainingData(data.begin(), data.begin() + splitIndex);
		std::vector<std::vector<double>> testingData(data.begin() + splitIndex, data.end());

		std::cout << "Training neural network with " << trainingData.size() << " data points..." << std::endl;

		double learningRate = 0.01;
		int numEpochs = 100;

		for (int epoch = 1; epoch <= numEpochs; ++epoch) {
			double totalLoss = 0.0;

			for (const auto& dataPoint : trainingData) {
				std::vector<double> input(dataPoint.begin(), dataPoint.end() - 1);
				std::vector<double> target(dataPoint.end() - 1, dataPoint.end());
				std::vector<double> predicted = forwardPropagation(input);

				double loss = lossFunc(predicted, target, lossType);
				totalLoss += loss;

				std::vector<double> gradient = backwardPropagation(input, target, learningRate);
			}

			std::cout << "Epoch " << epoch << " Loss: " << totalLoss << std::endl;
		}

		std::cout << "Testing neural network with " << testingData.size() << " data points..." << std::endl;

		int numCorrect = 0;

		for (const auto& dataPoint : testingData) {
			std::vector<double> input(dataPoint.begin(), dataPoint.end() - 1);
			std::vector<double> target(dataPoint.end() - 1, dataPoint.end());
			std::vector<double> predicted = forwardPropagation(input);

			if (target[0] == 1 && predicted[0] >= 0.5) {
				numCorrect++;
			}
			else if (target[0] == 0 && predicted[0] < 0.5) {
				numCorrect++;
			}
		}

		double accuracy = static_cast<double>(numCorrect) / testingData.size();

		std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
	}

	void save(const std::string& filename) {
		std::ofstream file(filename, std::ios::out | std::ios::binary);
		if (file.is_open()) {
			// Serialize and save the layers
			for (const auto& layer : layers) {
				std::string serializedData = layer->getData();
				file << serializedData << "\n";
			}
			file.close();
			std::cout << "Neural network saved to " << filename << std::endl;
		}
		else {
			std::cerr << "Unable to open the file for saving." << std::endl;
		}
	}

	// Load the neural network from a file
	void load(const std::string& filename) {
		std::ifstream file(filename);
		if (file.is_open()) {
			layers.clear(); // Clear existing layers
			std::string line;

			while (std::getline(file, line)) {
				std::istringstream layerInfo(line);
				std::string layerTypeStr;
				std::getline(layerInfo, layerTypeStr, '|');
				LayerType layerType = static_cast<LayerType>(std::stoi(layerTypeStr));

				int inSize, outSize, numFilters, kernelFilterSize, step;
				double dropoutRate;
				ActivationFunction activationType = ActivationFunction::None;

				if (layerInfo >> inSize >> outSize >> numFilters >> kernelFilterSize >> step >> dropoutRate) {
					int activationTypeInt;
					if (layerInfo >> activationTypeInt) {
						activationType = static_cast<ActivationFunction>(activationTypeInt);
					}

					// Add the layer to the network
					addLayer(inSize, outSize, layerType, activationType, numFilters, kernelFilterSize, step, dropoutRate);
				}
			}

			file.close();
			std::cout << "Neural network loaded from " << filename << std::endl;
		}
		else {
			std::cerr << "Unable to open the file for loading." << std::endl;
		}
	}
};

