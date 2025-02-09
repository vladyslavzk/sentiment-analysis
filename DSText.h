#ifndef SENTIMENTANALYSIS_DSTEXT_H
#define SENTIMENTANALYSIS_DSTEXT_H

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

using namespace std;


class DSText {
private:
    vector<string> tokens;
    int label; // 0 - negative, 1 - positive

public:
    DSText(vector<string> &txt, int lbl) : tokens(txt), label(lbl) {}

    vector<string> getTokens() const;

    int getLabel() const;
};


#endif //SENTIMENTANALYSIS_DSTEXT_H
