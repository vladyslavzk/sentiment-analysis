#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <unordered_map>
#include <string>
#include "Dataset.h"
#include "TextPreprocessor.h"

class NeuralNetwork {
private:
    std::vector<std::vector<double>> weightsInputHidden; // Weights between input and hidden layer
    std::vector<double> biasHidden;                      // Biases for the hidden layer
    std::vector<double> weightsHiddenOutput;             // Weights between hidden and output layer
    double biasOutput;                                   // Bias for the output layer
    int inputSize;                                       // Number of input features
    int hiddenSize;                                      // Number of neurons in the hidden layer

    double sigmoid(double x);
    double sigmoidDerivative(double x);
    double relu(double x);
    double reluDerivative(double x);
    double forward(const std::vector<double>& input, std::vector<double>& hiddenLayerOutput);
    void backward(const std::vector<double>& input, const std::vector<double>& hiddenLayerOutput, double output, double target, double learningRate);

public:
    NeuralNetwork(int inputSize, int hiddenSize); // Constructor only initializes network structure

    // Training function now accepts hyperparameters like learningRate and epochs
    void train(Dataset& trainData, std::unordered_map<std::string, int> vocabulary, int epochs, double learningRate, Dataset& devData, bool verbose = true);
    int predict(const std::vector<double>& input);
    double evaluate(Dataset& devData, std::unordered_map<std::string, int> vocabulary);

    // Functions for saving and loading weights
    void saveWeights(const std::string& filename) const;
    bool loadWeights(const std::string& filename);

    // Getters for validation purposes
    int getInputSize() const { return inputSize; }
    int getHiddenSize() const { return hiddenSize; }
};

#endif // NEURALNETWORK_H
