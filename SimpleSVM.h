#ifndef SimpleSVM_H
#define SimpleSVM_H

#include <vector>
#include <unordered_map>
#include "Dataset.h"

class SimpleSVM {
private:
    std::vector<double> weights;
    double bias;

    double predictRaw(const std::vector<double>& features);
    void updateWeights(const std::vector<double>& features, double label, double learningRate, double regularizationParam);

public:
    SimpleSVM();
    void train(Dataset& dataset, const std::unordered_map<std::string, int>& vocabulary, Dataset& devData, double learningRate, int epochs, double regularizationParam, bool verbose=true);
    int predict(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int>& vocabulary);
    double evaluate(Dataset& dataset, const std::unordered_map<std::string, int>& vocabulary);

    // Functions for saving and loading model weights
    void saveWeights(const std::string& filename) const;
    bool loadWeights(const std::string& filename);
};

#endif // SimpleSVM_H
