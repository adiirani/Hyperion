#include "../headers/pooling.h"
#include <random>
#include <sstream>


	/// <summary>
	/// Note: CONVOLUTIONAL AND POOLING LAYERS EXIST SOLELY AS PROOF OF CONCEPTS. THE OTHER LAYERS ARE MORE TESTED.
	/// </summary>


	pooling::pooling(int nodeCount,int XSize, int YSize, int ZSize, int FilterSize, int Stride) : layer(nodeCount, None){
		int xSize = xSize;
		int ySize = ySize;
		int zSize = zSize;
		int filterSize = filterSize;
		int stride = stride;
		// Initialize the output size based on input size and pooling parameters
		int outputSize = (nodeCount - filterSize) / stride + 1;
		output.resize(outputSize, std::vector<double>(outputSize, 0.0));
		maxIndices.resize(outputSize, std::vector<int>(outputSize, 0));
	}
	 
	// Forward pass
	std::vector<double> pooling::forward(const std::vector<double>& input) {
		int maxPoolSize = filterSize * filterSize;
		std::vector<std::vector<std::vector<double>>> maxPooledValues;
		maxPooledValues.reserve(output.size() * output[0].size());
		std::vector<double> reflattenedValues;

		std::vector<std::vector<std::vector<double>>> unflattenedImage;

		for (int x = 0; x < xSize; ++x) {
			std::vector<std::vector<double>> yArr;
			for (int y = 0; y < ySize; ++y) {
				std::vector<double> pixel = { input[x * xSize + y],input[x * xSize + y + 1],input[x * xSize + y + 2] };
				yArr.push_back(pixel);
			}
			unflattenedImage.push_back(yArr);
		}


		for (int x = 0; x < xSize; x += stride) {
			std::vector<std::vector<double>> row;
			for (int y = 0; y < ySize; y += stride) {
				std::vector<double> maxPixel(filterSize * filterSize, 0.0);

				for (int i = 0; i < filterSize; i++) {
					for (int j = 0; j < filterSize; j++) {
						for (int z = 0; z < filterSize * filterSize; z++) {
							int xi = x + i;
							int yj = y + j;
							if (xi < xSize && yj < ySize) {
								int index = (xi * xSize * filterSize * filterSize) + (yj * filterSize * filterSize) + z;
								double val = unflattenedImage[xi][yj][z];
								if (val > maxPixel[z]) {
									maxPixel[z] = val;
								}
							}
						}
					}
				}
				row.push_back(maxPixel);
			}
			maxPooledValues.push_back(row);
		}

		for (const auto& row : maxPooledValues) {
			for (const auto& pixel : row) {
				for (const double value : pixel) {
					reflattenedValues.push_back(value);
				}
			}
		}

		return reflattenedValues;
	}

	// Backward pass
	std::vector<double> pooling::backward(const std::vector<double>& input, const std::vector<double>& outputGradient, double learningRate) {
		std::vector<double> inputGradient(input.size(), 0.0);

		for (int x = 0; x < output.size(); x++) {
			for (int y = 0; y < output.size(); y++) {
				int maxIndex = maxIndices[x][y];
				inputGradient[maxIndex] += outputGradient[x * output.size() + y];
			}
		}

		return inputGradient;
	}

	std::string pooling::getData()  {
		// Serialize layer data into a string
		std::string data = "pooling|";

		// Add input size, filterSize, and stride to the data
		data += std::to_string(nodeCount) + "|" + std::to_string(filterSize) + "|" + std::to_string(stride) + "|";
		data += "|" + std::to_string(xSize) + "," + std::to_string(ySize) + "," + std::to_string(zSize);

		return data;
	}

	void pooling::setData(std::string token) {
		// Deserialize and set layer data from the string
		std::istringstream dataStream(token);
		std::string layerType;
		getline(dataStream, layerType, '|');

		if (layerType == "pooling") {
			// Deserialize input size, filterSize, and stride
			std::string sizeStr,xStr,yStr,zStr;
			getline(dataStream, sizeStr, '|');
			std::istringstream sizeStream(sizeStr);
			std::string nodeCountStr, filterSizeStr, strideStr;
			getline(sizeStream, nodeCountStr, '|');
			getline(sizeStream, filterSizeStr, '|');
			getline(sizeStream, strideStr, '|');
			nodeCount = std::stoi(nodeCountStr);
			filterSize = std::stoi(filterSizeStr);
			stride = std::stoi(strideStr);

			// Initialize the output size and maxIndices
			int outputSize = (nodeCount - filterSize) / stride + 1;
			output.resize(outputSize, std::vector<double>(outputSize, 0.0));
			maxIndices.resize(outputSize, std::vector<int>(outputSize, 0));

			std::getline(dataStream, xStr, '|');
			std::getline(dataStream, yStr, '|');
			std::getline(dataStream, zStr, '|');
			xSize = std::stoi(xStr);
			ySize = std::stoi(yStr);
			zSize = std::stoi(zStr);
		}
	}