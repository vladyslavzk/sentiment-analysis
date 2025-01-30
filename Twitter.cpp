#include "Twitter.h"
#include <filesystem>

// Function to load data from a file into a Dataset object
// Optionally limits the number of sentences loaded (useful for testing or smaller datasets)
void Twitter::loadData(const std::string &filename, Dataset &dataset, int n_sentences) {
    std::ifstream file;
    file.open(filename);
    std::string line;
    int sentence_count = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string tweetID, entity, sentiment, text;
        int label;

        // Extract fields from the CSV line
        std::getline(iss, tweetID, ',');
        std::getline(iss, entity, ',');
        std::getline(iss, sentiment, ',');
        std::getline(iss, text);

        // Convert sentiment string to a label (1 for Positive, 0 for Negative)
        if (sentiment == "Positive") {
            label = 1;
        } else if (sentiment == "Negative") {
            label = 0;
        } else {
            continue; // Skip sentences that are not labeled as Positive or Negative
        }

        // Preprocess the text and add it to the dataset
        std::vector<std::string> tokens = TextPreprocessor::preprocess(text, stopwords);
        dataset.addTokens(tokens, label);

        sentence_count++;

        // Stop loading if the specified number of sentences is reached
        if (n_sentences > 0 && sentence_count >= n_sentences) {
            break;
        }
    }
    file.close();
}

// Function to load training data from a file
// Calls loadData with the training dataset and optional sentence limit
void Twitter::loadTrainData(std::string filename, int n_sentences) {
    std::cout << "Loading train dataset..." << std::endl;
    loadData(filename, train, n_sentences);
    std::cout << "Loading train dataset complete" << std::endl;
}

// Function to load validation data from a file
// Calls loadData with the validation dataset and optional sentence limit
void Twitter::loadDevData(std::string filename, int n_sentences) {
    std::cout << "Loading validation dataset..." << std::endl;
    loadData(filename, dev, n_sentences);
    std::cout << "Loading validation dataset complete" << std::endl;
}

// Function to load stopwords from a file
// Uses the TextPreprocessor to read stopwords and stores them in an unordered_set
void Twitter::loadStopwords(std::string filename) {
    std::cout << "Loading stopwords..." << std::endl;
    stopwords = TextPreprocessor::readStopwords(filename);
    std::cout << "Loading stopwords complete" << std::endl;
}

// Getter function to access the training dataset
Dataset &Twitter::getTrainData() {
    return train;
}

// Getter function to access the validation dataset
Dataset &Twitter::getDevData() {
    return dev;
}
