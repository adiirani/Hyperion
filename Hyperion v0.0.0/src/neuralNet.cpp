#include "../headers/neuralNet.h"


/// <summary>
/// MAIN NEURAL NETWORK CLASS
/// Uses stack magic to build a neural network. Sort of like that cake game in purble place but actually useful.
/// USAGE !!!PAY ATTENTION!!!
///		neuralNet myNeuralNetwork(<2d layers vector containing node count, layer type and activation function type>); Declares neural network, with set layers if specified
///		neuralNet.add(<node count>,<layer type>, <activation function type>); adds neural network layer
///		neuralNet.traintest(<data>,<split>); runs it on data from a custom data class TB specified. specify split in terms of val from 0 to 1, so like 0.5 for 50%
///		neuralNet.save(<filename>); saves Neural Network and destroys the object.
///		neuralNet.load(<filename>
///		neuralNet.destroy(); destroys Neural Network without saving (if you made skynet with this library and need to kill it use this, though i dont even understand why you would do so with such a bad library.)
/// </summary>

	std::vector<double> neuralNet::backwardPropagation(const std::vector<double>& target, const std::vector<double>& outputGradient, double learningRate) {
		int numLayers = layers.size();
		std::vector<double> layerError = outputGradient;
		for (int i = numLayers - 1; i >= 0; --i) {
			auto& currLayer = layers[i];
			layerError = currLayer->backward(target, layerError, learningRate);
		}
		return layerError;  // Return the error propagated to the input layer.
	}
	neuralNet::neuralNet() {
		std::cout << "Initializing neural network.";
	}

	std::vector<double> neuralNet::forwardPropagation(const std::vector<double>& input) {
		std::vector<double> layerInput = input;
		for (const auto& currLayer : layers) {
			if (currLayer->nodeCount < layerInput.size()) { //averages input for layers of smaller nodecounts
				int cell = std::div(layerInput.size(), currLayer->nodeCount).quot;
				std::vector<double> layerTemp;
				for (int x = 0; x < currLayer->nodeCount; x++) {
					int total = 0;
					for (int y = 0; y < cell; y++) {
						total += layerInput[y];
					}
					total /= cell;
					layerTemp.push_back(total);
				}
				layerInput = layerTemp;
			}
			layerInput = currLayer->forward(layerInput);
		}
		return layerInput;  // Return the final output.
	}

	void neuralNet::addLayer(int nodeCount, LayerType type, ActivationFunction activationType, double dropoutRate, int xSize = 0, int ySize = 0, int zSize = 0,int numFilters = 0, int kernelFilterSize = 0, int step = 0) {
		std::unique_ptr<layer> currentLayer; // Declare a pointer to the base class.

		switch (type) {
		case FullConn:
			currentLayer = std::make_unique<fullConn>(nodeCount, activationType);
			break;
		case Convo:
			currentLayer = std::make_unique<convo>(nodeCount, numFilters, kernelFilterSize,xSize,ySize,zSize, step, activationType);
			break;
		case Pooling:
			currentLayer = std::make_unique<pooling>(nodeCount,xSize,ySize,zSize, kernelFilterSize, step);
			break;
		case Recurrent:
			currentLayer = std::make_unique<recurrent>(nodeCount, activationType);
			break;
		case Embedding:
			currentLayer = std::make_unique<embedding>(nodeCount);
			break;
		case Normalization:
			currentLayer = std::make_unique<normalization>(nodeCount);
			break;
		case Dropout:
			currentLayer = std::make_unique<dropout>(nodeCount, dropoutRate);
			break;
		case Attention:
			currentLayer = std::make_unique<attentive>(nodeCount, activationType);
			break;
		case Residual:
			currentLayer = std::make_unique<residual>(nodeCount, activationType);
			break;
		default:
			std::cerr << "Didn't specify a valid layer type." << std::endl;
			return; // Exit the function if the type is invalid.
		}


		layers.push_back(std::move(currentLayer));
	}

	void neuralNet::trainTest(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& labels, double split, LossFunction lossType) {
    		if (split <= 0.0 || split >= 1.0) {
        		throw std::invalid_argument("Invalid split value. The split value should be in the range (0, 1).");
    		}

		// Split the data into training and testing sets
		int numDataPoints = data.size();
		int splitIndex = static_cast<int>(split * numDataPoints);
		std::vector<std::vector<double>> trainingData(data.begin(), data.begin() + splitIndex);
		std::vector<std::vector<double>> trainingLabels(labels.begin(), labels.begin() + splitIndex);
		std::vector<std::vector<double>> testingData(data.begin() + splitIndex, data.end());
		std::vector<std::vector<double>> testingLabels(labels.begin() + splitIndex, labels.end());
		
		std::cout << "Training neural network with " << trainingData.size() << " data points..." << std::endl;
		
		double learningRate = 0.01;
		int numEpochs = 100;
		
		for (int epoch = 1; epoch <= numEpochs; ++epoch) {
			double totalLoss = 0.0;
		
			for (size_t i = 0; i < trainingData.size(); i++) {
			    std::vector<double> input = trainingData[i];
			    std::vector<double> target = trainingLabels[i];
			    std::vector<double> predicted = forwardPropagation(input);
			
			    double loss = lossFunc(predicted, target, lossType);
			    totalLoss += loss;
			
			    std::vector<double> gradient = backwardPropagation(target, predicted, learningRate);
			}
		
			std::cout << "Epoch " << epoch << " Loss: " << totalLoss << std::endl;
		}
		
		std::cout << "Testing neural network with " << testingData.size() << " data points..." << std::endl;
		
		int numCorrect = 0;
		
		for (size_t i = 0; i < testingData.size(); i++) {
			std::vector<double> input = testingData[i];
			std::vector<double> target = testingLabels[i];
			std::vector<double> predicted = forwardPropagation(input);
		
			if (target[0] == 1 && predicted[0] >= 0.5) {
		    		numCorrect++;
			} else if (target[0] == 0 && predicted[0] < 0.5) {
		    		numCorrect++;
			}


		}
		
		double accuracy = static_cast<double>(numCorrect) / testingData.size();
		
		std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
	}


	void neuralNet::save(const std::string& filename) {
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
	void neuralNet::load(const std::string& filename) {
		std::ifstream file(filename);
		if (file.is_open()) {
			layers.clear(); // Clear existing layers
			std::string line;

			while (std::getline(file, line)) {
				std::istringstream layerInfo(line);
				std::string layerTypeStr;
				std::getline(layerInfo, layerTypeStr, '|');
				LayerType layerType = static_cast<LayerType>(std::stoi(layerTypeStr));

				addLayer(0, layerType, None, 0, 0, 0, 0, 0, 0, 0); //much more efficient than the LLM pass XD
				layers.back()->setData(line);
			}

			file.close();
			std::cout << "Neural network loaded from " << filename << std::endl;
		}
		else {
			std::cerr << "Unable to open the file for loading." << std::endl;
		}
	}

