#ifndef SENTIMENTANALYSIS_TWITTER_H
#define SENTIMENTANALYSIS_TWITTER_H

#include "Dataset.h"
#include "TextPreprocessor.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>


class Twitter {
private:
    Dataset train;
    Dataset dev;
    unordered_set<string> stopwords;
    void loadData(const string &filename, Dataset &dataset, int n_sentences=-1);

public:
    void loadStopwords(string filename);
    void loadTrainData(string filename, int n_sentences=-1);
    void loadDevData(string filename, int n_sentences=-1);


    Dataset &getTrainData();
    Dataset &getDevData();
};

#endif //SENTIMENTANALYSIS_TWITTER_H
