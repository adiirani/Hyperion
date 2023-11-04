#include "../headers/PreProcessing.h"

    std::vector<double> encoders::label(const std::vector<std::string>& input) {
        std::unordered_map<std::string, double> encoding;
        std::vector<double> result;

        int index = 1;
        for (const std::string& category : input) {
            if (encoding.find(category) == encoding.end()) {
                encoding[category] = index++;
            }
            result.push_back(encoding[category]);
        }

        return result;
    }

    std::vector<std::vector<double>> encoders::oneHot(const std::vector<std::string>& input) {
        std::unordered_map<std::string, double> encoding;
        std::vector<std::vector<double>> result;

        int index = 0;
        for (const std::string& category : input) {
            if (encoding.find(category) == encoding.end()) {
                encoding[category] = index++;
            }
            std::vector<double> ohVector(encoding.size(), 0.0);
            ohVector[encoding[category]] = 1.0;
            result.push_back(ohVector);
        }

        return result;
    }

    std::vector<double> encoders::ordinal(const std::vector<std::string>& input, const std::vector<std::string>& order) {
        std::map<std::string, double> encoding;
        std::vector<double> results;

        for (int i = 0; i < order.size(); i++) {
            encoding[order[i]] = i;
        }

        for (auto& val : input) {
            results.push_back(encoding[val]);
        }

        return results;
    }

    std::vector<std::string> vectorizers::tokenize(const std::string& input, const std::set<std::string>& stopWords) {
        std::string tokenText = input;
        std::transform(tokenText.begin(), tokenText.end(), tokenText.begin(), ::tolower);

        std::vector<std::string> tokens;
        size_t pos = 0;
        while ((pos = tokenText.find(" ")) != std::string::npos) {
            std::string token = tokenText.substr(0, pos);
            if (stopWords.find(token) == stopWords.end()) {
                tokens.push_back(token);
            }
            tokenText.erase(0, pos + 1);
        }
        if (!tokenText.empty() && stopWords.find(tokenText) == stopWords.end()) {
            tokens.push_back(tokenText);
        }
        return tokens;
    }

    size_t vectorizers::hash(const std::string& input) {
        std::hash<std::string> hashmaker;
        return hashmaker(input);
    }

    std::vector<std::vector<double>> vectorizers::count(const std::vector<std::string>& input, const std::set<std::string>& stopWords) {
        std::vector<std::vector<double>> result;

        for (const std::string& text : input) {
            std::vector<double> documentVector;
            std::unordered_map<std::string, double> termCounts;

            std::vector<std::string> tokenized = tokenize(text, stopWords);

            for (const std::string& token : tokenized) {
                termCounts[token]++;
            }

            for (const std::string& token : tokenized) {
                double count = termCounts[token];
                documentVector.push_back(count);

                // Update the reverse dictionary
                if (reverseDictionary.find(count) == reverseDictionary.end()) {
                    reverseDictionary[count] = std::vector<std::string>{ token };
                }
                else {
                    reverseDictionary[count].push_back(token);
                }
            }

            result.push_back(documentVector);
        }

        return result;
    }

    std::vector<std::vector<double>> vectorizers::tfidf(const std::vector<std::string>& input, const std::set<std::string>& stopWords) {
        // Step 1: Create a vocabulary and calculate term frequencies (TF) for each term in each document
        std::unordered_map<std::string, double> termIndex; // Vocabulary
        std::vector<std::vector<double>> termFrequencies;   // Term frequencies for each document

        for (const std::string& text : input) {
            std::vector<std::string> tokenized = tokenize(text, stopWords);
            std::vector<double> documentTermFrequencies(termIndex.size(), 0.0); // Initialize with the size of the current vocabulary

            for (const std::string& token : tokenized) {
                // Step 1a: Build vocabulary
                if (termIndex.find(token) == termIndex.end()) {
                    termIndex[token] = termIndex.size();
                    // Expand the size of documentTermFrequencies to match the updated vocabulary size
                    documentTermFrequencies.resize(termIndex.size(), 0.0);
                }

                // Step 1b: Calculate term frequencies (TF)
                int termIdx = termIndex[token];
                documentTermFrequencies[termIdx]++;
            }

            termFrequencies.push_back(documentTermFrequencies);
        }

        return termFrequencies;
    }

    std::vector<std::vector<double>> vectorizers::hashing(const std::vector<std::string>& input, const std::set<std::string>& stopWords) { //written by LLM
        std::vector<std::vector<double>> result;

        for (const std::string& document : input) {
            std::vector<double> documentVector;

            // Tokenize the document (same as before)
            std::vector<std::string> tokenized = tokenize(document, stopWords);

            // Calculate the hashed feature vector
            std::vector<double> hashedFeatures;

            for (const std::string& token : tokenized) {
                // Hash the token to a double
                double featureIdx = static_cast<double>(hash(token));
                hashedFeatures.push_back(featureIdx);
            }

            // Update the reverse dictionary
            for (double featureIdx : hashedFeatures) {
                if (reverseDictionary.find(featureIdx) == reverseDictionary.end()) {
                    reverseDictionary[featureIdx] = std::vector<std::string>{};
                }
                reverseDictionary[featureIdx].push_back("hash_" + std::to_string(featureIdx));
            }

            result.push_back(hashedFeatures);
        }

        return result;
    }

    std::unordered_map<double, std::vector<std::string>> vectorizers::getReverseDictionary() const { //written by LLM
        return reverseDictionary;
    }

    std::vector<double> image::loadAndConvertImage(const std::string& filename, int& imageWidth, int& imageHeight) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Unable to open the image file." << std::endl;
            return {};
        }

        // Read the image header (assuming a simple BMP format)
        char header[54];
        file.read(header, 54);

        // Extract image dimensions
        imageWidth = *reinterpret_cast<int*>(&header[18]);
        imageHeight = *reinterpret_cast<int*>(&header[22]);

        // Initialize the vector to hold the image data
        std::vector<double> imageVector(imageWidth * imageHeight * 3, 0.0);

        // Read the image data (RGB values)
        for (int i = 0; i < imageHeight; ++i) {
            for (int j = 0; j < imageWidth; ++j) {
                char pixel[3];
                file.read(pixel, 3);

                // Store RGB values as doubles in the vector
                imageVector[(i * imageWidth + j) * 3] = static_cast<double>(static_cast<unsigned char>(pixel[2])); // R
                imageVector[(i * imageWidth + j) * 3 + 1] = static_cast<double>(static_cast<unsigned char>(pixel[1])); // G
                imageVector[(i * imageWidth + j) * 3 + 2] = static_cast<double>(static_cast<unsigned char>(pixel[0])); // B
            }
        }

        return imageVector;
    }

