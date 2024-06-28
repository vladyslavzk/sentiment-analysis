//
// Created by Vlad on 6/24/2024.
//

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


public:
    void loadStopwords(string filename);

    void loadData(const string &filename, Dataset &dataset);

    void loadTrainData(string filename);

    void loadDevData(string filename);


    Dataset &getTrainData();

    Dataset &getDevData();

};

#endif //SENTIMENTANALYSIS_TWITTER_H
