#include "../headers/convo.h"

#include <random>
#include <sstream>


/// <summary>
/// Note: CONVOLUTIONAL AND POOLING LAYERS EXIST SOLELY AS PROOF OF CONCEPTS. THE OTHER LAYERS ARE MORE TESTED.
/// </summary>


	convo::convo(int nodeCount, int NumFilters, int KernelSize, int XSize, int YSize, int ZSize, int Stride, ActivationFunction activationFunc)
		: layer(nodeCount, activationFunc), kernelSize(kernelSize), numFilters(numFilters), stride(stride) {
		int numFilters = NumFilters;
		int kernelSize = KernelSize;
		int xSize = XSize;
		int ySize = YSize;
		int zSize = ZSize;
		int stride = Stride;
		int filterSize = kernelSize * kernelSize * zSize;

		filters.resize(numFilters, std::vector<std::vector<double>>(kernelSize, std::vector<double>(kernelSize * zSize)));

		// Initialize filters with random values using normal dist
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dist(0.0, 1.0);

		for (int x = 0; x < numFilters; ++x) {
			for (int y = 0; y < kernelSize; ++y) {
				for (int z = 0; z < kernelSize * zSize; ++z) {
					filters[x][y][z] = dist(gen);
				}
			}
		}
	}
	//TODO: OPTIMISE THIS PART OF THE CODE.
	// still not optimized
	// but this is a start
	// i would rely on eigen for matrix operations buuuut...
	std::vector<double> convo::forward(const std::vector<double>& input) {
		int outputSize = (nodeCount - kernelSize) / stride + 1;
		std::vector<double> output(numFilters * outputSize * outputSize, 0.0);
		std::vector<std::vector<std::vector<double>>> unflattenedImage;

		//unflatten
		for (int x = 0; x < xSize; ++x) {
			std::vector<std::vector<double>> yArr;
			for (int y = 0; y < ySize; ++y) {
				std::vector<double> pixel = { input[x * xSize + y],input[x * xSize + y + 1],input[x * xSize + y + 2] };
				yArr.push_back(pixel);
			}
			unflattenedImage.push_back(yArr);
		}


		//perform convo
		for(int filter = 0; filter < numFilters; ++filter){
			for (int imgX = 0; imgX < nodeCount; ++imgX ) {
				for (int imgY = 0; imgX < nodeCount; ++imgY) {
					for (int kerX = 0; kerX < filterSize; kerX += stride) {
						for (int kerY = 0; kerY < filterSize; kerY += stride) {
							for (int kerZ = 0; kerZ < zSize; ++kerZ) {
								output[filter * outputSize * outputSize + imgX * outputSize + imgY] +=  //autoflattens cause why not
									filters[filter][kerX][kerY * zSize + kerZ] * unflattenedImage[imgX + kerX][imgY + kerY][kerZ];
							}
						}
					}
				}
			}
		}

		return activation(activFunction, output);
	}

	// Backward pass (written by LLM)
	std::vector<double> convo::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) {
		int outputSize = (nodeCount - kernelSize) / stride + 1;
		int filterSize = kernelSize * kernelSize * zSize;

		std::vector<double> inputGradient(input.size(), 0.0);
		std::vector<std::vector<std::vector<double>>> unflattenedImage(xSize, std::vector<std::vector<double>>(ySize, std::vector<double>(zSize, 0.0)));

		for (int x = 0; x < xSize; ++x) {
			for (int y = 0; y < ySize; ++y) {
				for (int z = 0; z < zSize; ++z) {
					unflattenedImage[x][y][z] = input[z + x * ySize * zSize + y * zSize];
				}
			}
		}

		// Initialize filter weight gradients
		std::vector<std::vector<std::vector<double>>> filterGradients(numFilters, std::vector<std::vector<double>>(kernelSize, std::vector<double>(kernelSize * zSize, 0.0)));

		// Backpropagation
		for (int filter = 0; filter < numFilters; ++filter) {
			for (int x = 0; x < outputSize; ++x) {
				for (int y = 0; y < outputSize; ++y) {
					for (int i = 0; i < kernelSize; ++i) {
						for (int j = 0; j < kernelSize; ++j) {
							for (int d = 0; d < zSize; ++d) {
								// Compute the input gradient using the chain rule
								for (int dx = 0; dx < outputSize; ++dx) {
									for (int dy = 0; dy < outputSize; ++dy) {
										inputGradient[d + x * ySize * zSize + y * zSize] +=
											filters[filter][i][j * zSize + d] * outputGradient[filter * outputSize * outputSize + x * outputSize + y];
									}
								}

								// Compute the filter weight gradients
								for (int dx = 0; dx < outputSize; ++dx) {
									for (int dy = 0; dy < outputSize; ++dy) {
										filterGradients[filter][i][j * zSize + d] +=
											unflattenedImage[x + i][y + j][d] * outputGradient[filter * outputSize * outputSize + x * outputSize + y];
									}
								}
							}
						}
					}
				}
			}
		}

		// Update filter weights
		for (int filter = 0; filter < numFilters; ++filter) {
			for (int i = 0; i < kernelSize; ++i) {
				for (int j = 0; j < kernelSize; ++j) {
					for (int d = 0; d < zSize; ++d) {
						filters[filter][i][j * zSize + d] -= learningRate * filterGradients[filter][i][j * zSize + d];
					}
				}
			}
		}

		return inputGradient;
	}

	std::string convo::getData() {
		// Serialize layer data into a string
		std::string data = "convo|";

		// Add input size, numFilters, kernelSize, and stride to the data
		data += std::to_string(nodeCount) + "|" + std::to_string(numFilters) + "|" + std::to_string(kernelSize) + "|" + std::to_string(stride) + "|";
		

		// Add activation function to the data
		data += std::to_string(static_cast<int>(activFunction)) + "|";

		// Serialize filters
		for (int filter = 0; filter < numFilters; ++filter) {
			for (int i = 0; i < kernelSize; ++i) {
				for (int j = 0; j < kernelSize; ++j) {
					for (int d = 0; d < zSize; ++d) {
						data += std::to_string(filters[filter][i][j * zSize + d]);
						data += ",";
					}
				}
			}
		}

		data += "|" + std::to_string(xSize) + "," + std::to_string(ySize) + "," + std::to_string(zSize);
		return data;
	}

	void convo::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "convo") {
			// Deserialize input size, numFilters, kernelSize, and stride
			std::string sizeStr, xStr, yStr,zStr;
			getline(dataStream, sizeStr, '|');
			std::istringstream sizeStream(sizeStr);
			std::string nodeCountStr, numFiltersStr, kernelSizeStr, strideStr;
			getline(sizeStream, nodeCountStr, '|');
			getline(sizeStream, numFiltersStr, '|');
			getline(sizeStream, kernelSizeStr, '|');
			getline(sizeStream, strideStr, '|');
			nodeCount = std::stoi(nodeCountStr);
			numFilters = std::stoi(numFiltersStr);
			kernelSize = std::stoi(kernelSizeStr);
			stride = std::stoi(strideStr);

			// Deserialize activation function
			std::string activationFuncStr;
			getline(dataStream, activationFuncStr, '|');
			activFunction = static_cast<ActivationFunction>(std::stoi(activationFuncStr));

			// Deserialize filters
			filters.resize(numFilters, std::vector<std::vector<double>>(kernelSize, std::vector<double>(kernelSize * zSize, 0.0)));

			for (int filter = 0; filter < numFilters; ++filter) {
				for (int i = 0; i < kernelSize; ++i) {
					for (int j = 0; j < kernelSize; ++j) {
						for (int d = 0; d < zSize; ++d) {
							std::string filterValueStr;
							if (getline(dataStream, filterValueStr, ',')) {
								filters[filter][i][j * zSize + d] = std::stod(filterValueStr);
							}
						}
					}
				}
			}

			std::getline(dataStream, xStr, '|');
			std::getline(dataStream, yStr, '|');
			std::getline(dataStream, zStr, '|');
			xSize = std::stoi(xStr);
			ySize = std::stoi(yStr);
			zSize = std::stoi(zStr);
		}
	}