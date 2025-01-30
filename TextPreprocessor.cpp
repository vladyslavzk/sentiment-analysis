#include "TextPreprocessor.h"
#include <math.h>

// Function to remove punctuation and digits in a given string and convert to lowercase
// This function cleans the text by removing unwanted characters and standardizing the case
std::string TextPreprocessor::removePunctuationAndDigits(const std::string &text) {
    std::string result;
    for (char c : text) {
        if (!std::ispunct(c) && !std::isdigit(c)) {
            result += std::tolower(c);
        }
    }
    return result;
}

// Function to tokenize a string into a vector of words (tokens)
// Splits the input string into individual words based on whitespace
std::vector<std::string> TextPreprocessor::tokenizeWords(const std::string &text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;

    while (ss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

// Function to read stopwords from a file and store them in an unordered_set
// Stopwords are commonly used words that are removed during text processing
std::unordered_set<std::string> TextPreprocessor::readStopwords(const std::string &filename) {
    std::unordered_set<std::string> stopwords;
    std::ifstream file(filename);
    std::string word;

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return stopwords;
    }

    while (std::getline(file, word)) {
        stopwords.insert(word);
    }

    return stopwords;
}

// Function to remove stopwords from a list of tokens
// Filters out words that are in the stopwords list
std::vector<std::string> TextPreprocessor::removeStopwords(const std::vector<std::string> &tokens, const std::unordered_set<std::string> &stopwords) {
    std::vector<std::string> filteredTokens;
    for (const std::string &token : tokens) {
        if (stopwords.find(token) == stopwords.end()) {
            filteredTokens.push_back(token);
        }
    }
    return filteredTokens;
}

// Function to perform simple stemming on a single token
// Reduces words to their base form by removing common suffixes
std::string TextPreprocessor::simpleStemming(const std::string &token) {
    std::string result = token;

    // Remove "ing" at the end
    if (result.size() >= 3 && result.compare(result.size() - 3, 3, "ing") == 0) {
        result = result.substr(0, result.size() - 3);
    }

    // Replace "tion" at the end with "te"
    if (result.size() >= 4 && result.compare(result.size() - 4, 4, "tion") == 0) {
        result = result.substr(0, result.size() - 4) + "te";
    }

    // Remove "ed" at the end
    if (result.size() >= 2 && result.compare(result.size() - 2, 2, "ed") == 0) {
        result = result.substr(0, result.size() - 2);
    }

    // Remove "ly" at the end
    if (result.size() >= 2 && result.compare(result.size() - 2, 2, "ly") == 0) {
        result = result.substr(0, result.size() - 2);
    }

    // Remove "s" at the end
    if (result.size() >= 2 && result.compare(result.size() - 1, 1, "s") == 0) {
        result = result.substr(0, result.size() - 1);
    }

    // Remove "es" at the end
    if (result.size() >= 2 && result.compare(result.size() - 2, 2, "es") == 0) {
        result = result.substr(0, result.size() - 2);
    }

    // Remove "ness" at the end
    if (result.size() >= 4 && result.compare(result.size() - 4, 4, "ness") == 0) {
        result = result.substr(0, result.size() - 4);
    }

    return result;
}

// Function to apply stemming on a list of tokens
// Applies the simpleStemming function to each token in the list
std::vector<std::string> TextPreprocessor::provideStemming(const std::vector<std::string> &tokens) {
    std::vector<std::string> result;

    result.reserve(tokens.size());
    for (const std::string &token : tokens) {
        result.push_back(simpleStemming(token));
    }

    return result;
}

// Preprocessing function that applies all preprocessing steps to a given text
// Combines all preprocessing steps: removing punctuation, tokenizing, removing stopwords, and stemming
std::vector<std::string> TextPreprocessor::preprocess(const std::string &text, const std::unordered_set<std::string> &stopwords) {
    std::string preprocessed_text = text;
    std::vector<std::string> tokens;

    preprocessed_text = removePunctuationAndDigits(preprocessed_text);
    tokens = tokenizeWords(preprocessed_text);
    tokens = removeStopwords(tokens, stopwords);
    tokens = provideStemming(tokens);

    return tokens;
}

// Function to create a feature vector from tokens based on a given vocabulary
// Converts a list of tokens into a numerical vector where each position corresponds to a word in the vocabulary
std::vector<double> TextPreprocessor::createFeatureVector(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int>& vocabulary) {
    std::vector<double> featureVector(vocabulary.size(), 0.0);

    for (const auto& token : tokens) {
        auto it = vocabulary.find(token);
        if (it != vocabulary.end()) {
            featureVector[it->second] += 1.0;
        }
    }

    return featureVector;
}
