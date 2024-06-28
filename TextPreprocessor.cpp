#include "TextPreprocessor.h"


// Function to remove punctuation and digits in a given string and lowercase it
std::string TextPreprocessor::removePunctuationAndDigits(const std::string &text) {
    std::string result;
    for (char c: text) {
        if (!std::ispunct(c) && !std::isdigit(c)) {
            result += std::tolower(c);
        }
    }
    return result;
}


// Function to tokenize a string to get a vector with a tokens (single words)
std::vector<std::string> TextPreprocessor::tokenizeWords(const std::string &text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;

    while (ss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}


// Function to read stopwords from a file into an unordered_set
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


// Function to remove stopwords from tokens
std::vector<std::string> TextPreprocessor::removeStopwords(const std::vector<std::string> &tokens,
                                                           const std::unordered_set<std::string> &stopwords) {
    std::vector<std::string> filteredTokens;
    for (const std::string &token: tokens) {
        if (stopwords.find(token) == stopwords.end()) {
            filteredTokens.push_back(token);
        }
    }
    return filteredTokens;
}


// Function for stemming on a single token
std::string TextPreprocessor::simpleStemming(const std::string &tokens) {
    std::string result = tokens;

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


// Provide stemming on a vector of tokens
std::vector<std::string> TextPreprocessor::provideStemming(const std::vector<std::string> &tokens) {
    std::vector<std::string> result;

    result.reserve(tokens.size());
    for (const std::string &token: tokens) {
        result.push_back(simpleStemming(token));
    }

    return result;
}


// Preprocessing function
std::vector<std::string>
TextPreprocessor::preprocess(const std::string &text, const std::unordered_set<std::string> &stopwords) {
    std::string preprocessed_text = text;
    std::vector<std::string> tokens;

    preprocessed_text = removePunctuationAndDigits(preprocessed_text);

    tokens = tokenizeWords(preprocessed_text);
    tokens = removeStopwords(tokens, stopwords);
    tokens = provideStemming(tokens);

    return tokens;
}

