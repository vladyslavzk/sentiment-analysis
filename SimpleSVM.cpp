#include "SimpleSVM.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "TextPreprocessor.h"

// Constructor: Initializes the bias to 0.0
SimpleSVM::SimpleSVM() : bias(0.0) {}

// Predict the raw output (margin) before applying the decision rule
// Returns the dot product of weights and features plus the bias
double SimpleSVM::predictRaw(const std::vector<double>& features) {
    return std::inner_product(features.begin(), features.end(), weights.begin(), 0.0) + bias;
}

// Update the weights and bias based on the SVM hinge loss
// Uses the margin to determine whether the current prediction is correct or not
void SimpleSVM::updateWeights(const std::vector<double>& features, double label, double learningRate, double regularizationParam) {
    double margin = label * predictRaw(features);

    if (margin >= 1) {
        // No error, just apply regularization
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learningRate * regularizationParam * weights[i];
        }
    } else {
        // Prediction error, apply regularization and update rule
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += learningRate * (label * features[i] - regularizationParam * weights[i]);
        }
        bias += learningRate * label;
    }
}

// Train the SVM model using the dataset and vocabulary
// Adjusts the weights and bias over multiple epochs using the hinge loss function
void SimpleSVM::train(Dataset& dataset, const std::unordered_map<std::string, int>& vocabulary, Dataset& devData, double learningRate, int epochs, double regularizationParam, bool verbose) {
    // Resize weights to match the number of features (vocabulary size)
    weights.resize(vocabulary.size(), 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (const auto& sample : dataset.getData()) {
            auto tokens = sample.getTokens();
            int label = sample.getLabel() == 1 ? 1 : -1; // Convert label to +1 or -1

            // Create the feature vector for the current sample
            auto featureVector = TextPreprocessor::createFeatureVector(tokens, vocabulary);

            // Update weights and bias based on the current sample
            updateWeights(featureVector, label, learningRate, regularizationParam);

            // Calculate hinge loss for the current sample
            double margin = label * predictRaw(featureVector);
            totalLoss += std::max(0.0, 1.0 - margin); // Hinge loss
        }

        if (verbose) {
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss << std::endl;
            double accuracy = evaluate(devData, vocabulary);
            std::cout << "Validation Accuracy: " << accuracy << "%" << std::endl;
        }
    }
}

// Predict the label (0 or 1) for a given sample based on the tokens
// Converts the raw prediction (margin) to a binary class label
int SimpleSVM::predict(const std::vector<std::string>& tokens, const std::unordered_map<std::string, int>& vocabulary) {
    auto featureVector = TextPreprocessor::createFeatureVector(tokens, vocabulary);
    return predictRaw(featureVector) >= 0.0 ? 1 : 0;
}

// Evaluate the accuracy of the SVM model on a validation dataset
// Compares predicted labels with true labels to calculate the accuracy
double SimpleSVM::evaluate(Dataset& dataset, const std::unordered_map<std::string, int>& vocabulary) {
    int correct_predictions = 0;
    int total_predictions = 0;

    for (const auto& sample : dataset.getData()) {
        int predicted_label = predict(sample.getTokens(), vocabulary);
        if (predicted_label == sample.getLabel()) {
            correct_predictions++;
        }
        total_predictions++;
    }

    return 100.0 * correct_predictions / total_predictions;
}

// Save the current weights and bias to a binary file
// Useful for saving the trained model to disk for later use
void SimpleSVM::saveWeights(const std::string& filename) const {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file for saving weights: " << filename << std::endl;
        return;
    }

    // Save weights
    for (double weight : weights) {
        outFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    }

    // Save bias
    outFile.write(reinterpret_cast<const char*>(&bias), sizeof(bias));

    outFile.close();
}

// Load the weights and bias from a binary file
// Allows the model to be restored from a previously saved state
bool SimpleSVM::loadWeights(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);

    if (!inFile.is_open()) {
        std::cerr << "Error opening file for loading weights: " << filename << std::endl;
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
