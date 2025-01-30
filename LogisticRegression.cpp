#include "LogisticRegression.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

// Constructor to initialize the LogisticRegression object
// Initializes weights to zeros and bias to 0.0
LogisticRegression::LogisticRegression(int numFeatures)
        : numFeatures(numFeatures), bias(0.0) {
    weights = std::vector<double>(numFeatures, 0.0);
}

// Function to create a feature vector using Bag of Words
// Converts a list of tokens into a fixed-size vector based on the vocabulary
std::vector<double> LogisticRegression::createFeatureVectorBoW(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int>& vocabulary) {
    std::vector<double> featureVector(vocabulary.size(), 0.0);

    // Increment the position in the vector corresponding to each token in the vocabulary
    for (const auto& token : tokens) {
        if (vocabulary.find(token) != vocabulary.end()) {
            featureVector[vocabulary.at(token)] += 1.0;
        }
    }

    return featureVector;
}

// Sigmoid function to map predictions to probabilities
// Converts the linear combination of features into a probability between 0 and 1
double LogisticRegression::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// Clipping function to avoid issues with log(0)
// Ensures that the prediction is never exactly 0 or 1
double LogisticRegression::clip(double value, double epsilon) {
    return std::max(epsilon, std::min(1 - epsilon, value));
}

// Function to update weights and bias using gradient descent
// Applies the computed gradients to adjust the model parameters
void LogisticRegression::updateWeights(const std::vector<double>& features, double error, double gradient, double learningRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += learningRate * error * gradient * features[i];
    }
    bias += learningRate * error * gradient;
}

// Function to predict the label for a single sample
// Uses the sigmoid function to compute the probability and returns the binary prediction
int LogisticRegression::predict(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int>& vocabulary) {
    auto featureVector = createFeatureVectorBoW(tokens, vocabulary);
    double linearCombination = std::inner_product(featureVector.begin(), featureVector.end(), weights.begin(), 0.0) + bias;
    double prediction = sigmoid(linearCombination);

    return prediction >= 0.5 ? 1 : 0;
}

// Function to evaluate the accuracy of the model on a validation dataset
// Compares predicted labels with true labels and calculates accuracy
double LogisticRegression::evaluate(Dataset& dataset, const std::unordered_map<std::string, int>& vocabulary) {
    int correctPredictions = 0;
    int totalPredictions = 0;

    for (const auto &sample : dataset.getData()) {
        auto tokens = sample.getTokens();
        int trueLabel = sample.getLabel();

        int predictedLabel = predict(tokens, vocabulary);

        if (predictedLabel == trueLabel) {
            correctPredictions++;
        }

        totalPredictions++;
    }

    return 100.0 * correctPredictions / totalPredictions;
}

// Function to train the logistic regression model
// Iteratively adjusts weights and bias using gradient descent, with an optional verbose output
void LogisticRegression::train(Dataset& dataset, const std::unordered_map<std::string, int>& vocabulary, Dataset& devDataset, double learningRate, int epochs, bool verbose) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < dataset.getData().size(); ++i) {
            const auto& sample = dataset.getData()[i];
            auto tokens = sample.getTokens();
            int label = sample.getLabel(); // Assume labels are 0 for negative and 1 for positive

            // Create feature vector using BoW
            auto featureVector = createFeatureVectorBoW(tokens, vocabulary);

            // Compute linear combination of features and weights
            double linearCombination = std::inner_product(featureVector.begin(), featureVector.end(), weights.begin(), 0.0) + bias;

            // Apply sigmoid function
            double prediction = sigmoid(linearCombination);

            // Clip prediction to avoid log(0)
            prediction = clip(prediction);

            // Compute the error
            double error = label - prediction;
            totalLoss += -label * std::log(prediction) - (1 - label) * std::log(1 - prediction);

            // Update weights and bias
            double gradient = prediction * (1 - prediction);
            updateWeights(featureVector, error, gradient, learningRate);

//            if (verbose) {
//                std::cout << "\rEpoch " << epoch + 1 << "/" << epochs
//                          << ", Sample " << i + 1 << "/" << dataset.getData().size() << std::flush;
//            }
        }

        if (verbose) {
            std::cout << "\nEpoch " << epoch + 1 << ", Loss: " << totalLoss << std::endl;
            double accuracy = evaluate(devDataset, vocabulary);
            std::cout << "Validation Accuracy: " << accuracy << "%" << std::endl;
        }
    }
}

// Function to save model weights and bias to a file
// Stores the model parameters in a binary file for later use
void LogisticRegression::saveWeights(const std::string& filename) const {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file for saving weights: " << filename << std::endl;
        return;
    }

    // Save model parameters (numFeatures)
    outFile.write(reinterpret_cast<const char*>(&numFeatures), sizeof(numFeatures));

    // Save weights
    for (double weight : weights) {
        outFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    }

    // Save bias
    outFile.write(reinterpret_cast<const char*>(&bias), sizeof(bias));

    outFile.close();
}

// Function to load model weights and bias from a file
// Restores the model parameters from a binary file
bool LogisticRegression::loadWeights(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);

    if (!inFile.is_open()) {
        std::cerr << "Error opening file for loading weights: " << filename << std::endl;
        return false;
    }

    // Load model parameters and check for consistency
    int loadedNumFeatures;

    inFile.read(reinterpret_cast<char*>(&loadedNumFeatures), sizeof(loadedNumFeatures));

    if (loadedNumFeatures != numFeatures) {
        std::cerr << "Model parameters do not match: "
                  << "Expected numFeatures: " << numFeatures
                  << ", but got numFeatures: " << loadedNumFeatures << "." << std::endl;
        inFile.close();
        return false;
    }

    // Load weights
    for (double& weight : weights) {
        inFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));
    }

    // Load bias
    inFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));

    inFile.close();
    return true;
}
