#include "Twitter.h"
#include <filesystem>

void Twitter::loadData(const std::string &filename, Dataset &dataset) {
    ifstream file;

    file.open(filename);
    string line;

    while (getline(file, line)) {
        istringstream iss(line);
        string tweetID, entity, sentiment, text;
        int label;

        getline(iss, tweetID, ',');
        getline(iss, entity, ',');
        getline(iss, sentiment, ',');
        getline(iss, text);

        if (sentiment == "Positive") {
            label = 0;
        } else if (sentiment == "Negative") {
            label = 1;
        } else {
            label = 2;
        }

        vector<string> tokens = TextPreprocessor::preprocess(text, stopwords);
        dataset.addTokens(tokens, label);
    }
    file.close();
}

void Twitter::loadTrainData(std::string filename) {
    loadData(filename, train);
}

void Twitter::loadDevData(std::string filename) {
    loadData(filename, dev);
}

void Twitter::loadStopwords(string filename) {
    stopwords = TextPreprocessor::readStopwords("../data/stopwords.txt");
}

Dataset &Twitter::getTrainData() {
    return train;
}

Dataset &Twitter::getDevData() {
    return dev;
}