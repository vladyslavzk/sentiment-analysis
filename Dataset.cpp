#include "Dataset.h"

// Getter function to access the data stored in the Dataset object
// Returns a reference to the vector containing DSText objects
vector<DSText> &Dataset::getData() {
    return data;
}

// Adds a new text sample to the dataset along with its corresponding label
// The tokens are stored as a DSText object, which is then added to the data vector
void Dataset::addTokens(vector<std::string> tokens, int label) {
    data.push_back(DSText(tokens, label));
}

// Creates a vocabulary from all tokens present in the dataset
// The vocabulary is an unordered_map where each token is assigned a unique index
std::unordered_map<std::string, int> Dataset::createVocabulary() {
    std::unordered_map<std::string, int> vocabulary;
    int index = 0;

    for (const auto& sample : getData()) {
        for (const auto& token : sample.getTokens()) {
            // If the token is not already in the vocabulary, add it with the next available index
            if (vocabulary.find(token) == vocabulary.end()) {
                vocabulary[token] = index++;
            }
        }
    }

    return vocabulary;
}
