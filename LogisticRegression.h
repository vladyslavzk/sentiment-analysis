#ifndef SENTIMENTANALYSIS_LOGISTICREGRESSION_H
#define SENTIMENTANALYSIS_LOGISTICREGRESSION_H

#include "Twitter.h"
#include <cmath>
#include <random>
#include <numeric>
#include <unordered_map>

class LogisticRegression {
private:
    std::vector<double> weights;
    double bias;
    int numFeatures;

    std::vector<double> createFeatureVectorBoW(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int>& vocabulary);
    double sigmoid(double z);
    void updateWeights(const std::vector<double>& features, double error, double gradient, double learningRate);
    double clip(double value, double epsilon = 1e-10);

public:
    LogisticRegression(int numFeatures);
    void train(Dataset& dataset, const std::unordered_map<std::string, int>& vocabulary, Dataset& devDataset, double learningRate, int epochs, bool verbose=true);
    int predict(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int>& vocabulary);
    double evaluate(Dataset& dataset, const std::unordered_map<std::string, int>& vocabulary);

    // Functions for saving and loading model weights
    void saveWeights(const std::string& filename) const;
    bool loadWeights(const std::string& filename);

    // Getters for validation purposes
    int getNumFeatures() const { return numFeatures; }
};

#endif // SENTIMENTANALYSIS_LOGISTICREGRESSION_H
