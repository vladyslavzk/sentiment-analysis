#ifndef SENTIMENTANALYSIS_TEXTPREPROCESSOR_H
#define SENTIMENTANALYSIS_TEXTPREPROCESSOR_H

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <iostream>
#include <fstream>
#include <cctype>
#include <algorithm>
#include <sstream>

#include "Dataset.h"

class TextPreprocessor {
private:
    static std::string removePunctuationAndDigits(const std::string &text);

    static std::vector<std::string> tokenizeWords(const std::string &text);

    static std::vector<std::string> removeStopwords(const std::vector<std::string> &tokens, const std::unordered_set<std::string> &stopwords);

    static std::string simpleStemming(const std::string &tokens);
    static std::vector<std::string> provideStemming(const std::vector<std::string> &tokens);

public:
    static std::unordered_set<std::string> readStopwords(const std::string &filename);
    static std::vector<std::string> preprocess(const std::string &text, const std::unordered_set<std::string> &);
    static std::vector<double> createFeatureVector(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int>& vocabulary);

    };


#endif //SENTIMENTANALYSIS_TEXTPREPROCESSOR_H
