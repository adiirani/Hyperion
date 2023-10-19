#include "./headers/PreProcessing.h"

class encoders {
public:
    std::vector<int> label(const std::vector<std::string>& input) {
        std::unordered_map<std::string, int> encoding;
        std::vector<int> result;

        int index = 1;
        for (const std::string& category : input) {
            if (encoding.find(category) == encoding.end()) {
                encoding[category] = index++;
            }
            result.push_back(encoding[category]);
        }

        return result;
    }

    std::vector<std::vector<int>> oneHot(const std::vector<std::string>& input) {
        std::unordered_map<std::string, int> encoding;
        std::vector<std::vector<int>> result;

        int index = 0;
        for (const std::string& category : input) {
            if (encoding.find(category) == encoding.end()) {
                encoding[category] = index++;
            }
            std::vector<int> ohVector(encoding.size(), 0);
            ohVector[encoding[category]] = 1;
            result.push_back(ohVector);
        }

        return result;
    }

    std::vector<int> ordinal(const std::vector<std::string>& input, const std::vector<std::string>& order) {
        std::map<std::string, int> encoding;
        std::vector<int> results;

        for (int i = 0; i < order.size(); i++) {
            encoding[order[i]] = i;
        }

        for (auto& val : input) {
            results.push_back(encoding[val]);
        }

        return results;
    }
};

class vectorizers {
private:
    std::unordered_map<int, std::vector<std::string>> reverseDictionary;

    std::vector<std::string> tokenize(const std::string& input, const std::set<std::string>& stopWords) {
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

    size_t hash(const std::string& input) {
        std::hash<std::string> hashmaker;
        return hashmaker(input);
    }

public:
    std::vector<std::vector<int>> count(const std::vector<std::string>& input, const std::set<std::string>& stopWords) {
        std::vector<std::vector<int>> result;

        for (const std::string& text : input) {
            std::vector<int> documentVector;
            std::unordered_map<std::string, int> termCounts;

            std::vector<std::string> tokenized = tokenize(text, stopWords);

            for (const std::string& token : tokenized) {
                termCounts[token]++;
            }

            for (const std::string& token : tokenized) {
                int count = termCounts[token];
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

    std::vector<std::vector<double>> tfidf(const std::vector<std::string>& input, const std::set<std::string>& stopWords) {
        // Step 1: Create a vocabulary and calculate term frequencies (TF) for each term in each document
        std::unordered_map<std::string, int> termIndex; // Vocabulary
        std::vector<std::vector<int>> termFrequencies;  // Term frequencies for each document

        for (const std::string& text : input) {
            std::vector<int> documentTermFrequencies(termIndex.size(), 0);
            std::vector<std::string> tokenized = tokenize(text, stopWords);

            for (const std::string& token : tokenized) {
                // Step 1a: Build vocabulary
                if (termIndex.find(token) == termIndex.end()) {
                    termIndex[token] = termIndex.size();
                }

                // Step 1b: Calculate term frequencies (TF)
                int termIdx = termIndex[token];
                documentTermFrequencies[termIdx]++;
            }

            termFrequencies.push_back(documentTermFrequencies);
        }

        // Step 2: Calculate document frequencies (DF) for each term
        std::vector<int> documentFrequencies(termIndex.size(), 0);

        for (const auto& docTermFreqs : termFrequencies) {
            for (int i = 0; i < docTermFreqs.size(); ++i) {
                if (docTermFreqs[i] > 0) {
                    documentFrequencies[i]++;
                }
            }
        }

        // Step 3: Compute TF-IDF values
        std::vector<std::vector<double>> result;

        for (const auto& docTermFreqs : termFrequencies) {
            std::vector<double> tfidfVector(termIndex.size(), 0.0);

            for (int i = 0; i < docTermFreqs.size(); ++i) {
                // Step 3a: Calculate TF-IDF values
                double tf = static_cast<double>(docTermFreqs[i]) / docTermFreqs.size();
                double idf = std::log(static_cast<double>(input.size()) / (1.0 + documentFrequencies[i]));
                tfidfVector[i] = tf * idf;
            }

            result.push_back(tfidfVector);
        }

        return result;
    }

    std::vector<std::vector<int>> hashing(const std::vector<std::string>& input, const std::set<std::string>& stopWords) {
        std::vector<std::vector<int>> result;

        for (const std::string& document : input) {
            std::vector<int> documentVector;

            // Tokenize the document (same as before)
            std::vector<std::string> tokenized = tokenize(document, stopWords);

            // Calculate the hashed feature vector
            std::vector<int> hashedFeatures;

            for (const std::string& token : tokenized) {
                // Hash the token to an integer
                int featureIdx = static_cast<int>(hash(token));
                hashedFeatures.push_back(featureIdx);
            }

            // Update the reverse dictionary
            for (int featureIdx : hashedFeatures) {
                if (reverseDictionary.find(featureIdx) == reverseDictionary.end()) {
                    reverseDictionary[featureIdx] = std::vector<std::string>{};
                }
                reverseDictionary[featureIdx].push_back("hash_" + std::to_string(featureIdx));
            }

            result.push_back(hashedFeatures);
        }

        return result;
    }

    std::unordered_map<int, std::vector<std::string>> getReverseDictionary() const {
        return reverseDictionary;
    }
};

class image {
public:
    std::vector<double> loadAndConvertImage(const std::string& filename, int& imageWidth, int& imageHeight) {
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
};