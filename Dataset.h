#ifndef SENTIMENTANALYSIS_DATASET_H
#define SENTIMENTANALYSIS_DATASET_H

#include "DSText.h"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>


class Dataset {
private:
    vector<DSText> data;

public:
    vector<DSText> &getData();
    void addTokens(vector<string> tokens, int label);
    std::unordered_map<std::string, int> createVocabulary();

};


#endif //SENTIMENTANALYSIS_DATASET_H
