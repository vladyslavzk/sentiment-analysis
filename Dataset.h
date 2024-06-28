#ifndef SENTIMENTANALYSIS_DATASET_H
#define SENTIMENTANALYSIS_DATASET_H

#include "DSText.h"

#include <iostream>
#include <vector>
#include <string>


class Dataset {
private:
    vector<DSText> data;

public:
    void addTokens(vector<string> tokens, int label);

    vector<DSText> &getData();
};


#endif //SENTIMENTANALYSIS_DATASET_H
