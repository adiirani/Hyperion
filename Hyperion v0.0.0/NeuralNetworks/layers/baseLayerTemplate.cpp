#include "headers/baseLayerTemplate.h"

#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>

class layer {
protected:
	int inputSize;
	int outputSize;
	ActivationFunction activFunction;
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;

	std::vector<double> activation(ActivationFunction activationType, std::vector<double>& input) {
		std::vector<double> output = input;
		double sumExp = 0.0;
		double maxVal = *std::max_element(output.begin(), output.end());

		switch (activationType) {
		case ReLU:
			for (double& x : output) {
				x = std::max(0.0, x);
			}
			break;
		case Sigmoid:
			for (double& x : output) {
				x = 1 / (1 + exp(x));
			}
			break;
		case Tanh:
			for (double& x : output) {
				x = (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
			}
			break;
		case SoftMax:
			for (double& x : output) {
				x = std::exp(x - maxVal);
				sumExp += x;
			}
			for (double& x : output) {
				x /= sumExp;
			}
			break;
		default:
			break;
		}
		return output;
	}

	std::vector<double> backtivation(ActivationFunction activationType, std::vector<double>& output) {
		std::vector<double> input = output;
		double maxVal = *std::max_element(input.begin(), input.end());
		double sumExp = 0.0;

		switch (activationType) {
		case ReLU:
			for (double& x : input) {
				x = (x > 0) ? 1.0 : 0.0;
			}
			break;
		case Sigmoid:
			for (double& x : input) {
				x = x * (1.0 - x);
			}
			break;
		case Tanh:
			for (double& x : input) {
				x = 1.0 - x * x;
			}
			break;
		case SoftMax:

			for (double& x : input) {
				x = x * (1.0 - x); // This is a simplified derivative of softmax, used for illustration.
			}
			break;
		default:
			break;
		}
		return input;
	}


public:
	layer(int inputSz, int outputSz, ActivationFunction activationFunc) { //even i dont know where this is going and i made this shit...
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dist(0.0, 1.0);
		ActivationFunction layerActivation = activationFunc;
		weights.resize(inputSz, std::vector<double>(outputSz));
		biases.resize(outputSz);
		inputSize = inputSz;
		outputSize = outputSz;
		activFunction = activationFunc;

		//init random biases according to normal distribution.
		for (int y = 0; y < outputSz; ++y) {
			for (int x = 0; x < inputSz; ++x) {
				weights[x][y] = dist(gen);
			}
			biases[y] = dist(gen); //i dont know if this hack works to make efficiency but...
		}

	}

	virtual std::vector<double> forward(const std::vector<double>& input) = 0;
	virtual std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) = 0;
	virtual std::string getData() = 0;
	virtual void setData(std::string token) = 0;

};