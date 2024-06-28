#include "Dataset.h"

vector<DSText> &Dataset::getData() {
    return data;
}

void Dataset::addTokens(vector<std::string> tokens, int label) {
    data.push_back(DSText(tokens, label));
}
