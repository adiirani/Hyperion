#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>
#include <cmath> // Include the cmath header for log function

class encoders {
public:
    std::vector<int> label(const std::vector<std::string>& input);
    std::vector<std::vector<int>> oneHot(const std::vector<std::string>& input);
    std::vector<int> ordinal(const std::vector<std::string>& input, const std::vector<std::string>& order);
};

class vectorizers {
private:
    std::unordered_map<int, std::vector<std::string>> reverseDictionary;

    std::vector<std::string> tokenize(const std::string& input, const std::set<std::string>& stopWords);
    size_t hash(const std::string& input);

public:
    std::vector<std::vector<int>> count(const std::vector<std::string>& input, const std::set<std::string>& stopWords);
    std::vector<std::vector<double>> tfidf(const std::vector<std::string>& input, const std::set<std::string>& stopWords);
    std::vector<std::vector<int>> hashing(const std::vector<std::string>& input, const std::set<std::string>& stopWords);
    std::unordered_map<int, std::vector<std::string>> getReverseDictionary() const;
};

class image {
public:
    std::vector<double> loadAndConvertImage(const std::string& filename, int& imageWidth, int& imageHeight);
};

#endif