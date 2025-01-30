#include "NeuralNetwork.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <random>


// Constructor initializes the structure and randomizes weights and biases
NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize)
        : inputSize(inputSize), hiddenSize(hiddenSize), biasOutput(0.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    weightsInputHidden.resize(hiddenSize, std::vector<double>(inputSize));
    biasHidden.resize(hiddenSize);
    weightsHiddenOutput.resize(hiddenSize);

    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weightsInputHidden[i][j] = d(gen) * sqrt(2.0 / inputSize); // He initialization
        }
        biasHidden[i] = 0.0;
        weightsHiddenOutput[i] = d(gen) * sqrt(2.0 / hiddenSize); // He initialization
    }
}


// Sigmoid activation function
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of the sigmoid function for backpropagation
double NeuralNetwork::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// ReLU activation function for hidden layers
double NeuralNetwork::relu(double x) {
    return std::max(0.0, x);
}

// Derivative of the ReLU function for backpropagation
double NeuralNetwork::reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Forward pass through the network
// Computes the output of the network given an input vector
double NeuralNetwork::forward(const std::vector<double>& input, std::vector<double>& hiddenLayerOutput) {
    // Compute hidden layer output
    for (int i = 0; i < hiddenSize; ++i) {
        hiddenLayerOutput[i] = biasHidden[i];
        for (int j = 0; j < inputSize; ++j) {
            hiddenLayerOutput[i] += input[j] * weightsInputHidden[i][j];
        }
        hiddenLayerOutput[i] = relu(hiddenLayerOutput[i]);
    }

    // Compute output layer output
    double output = biasOutput;
    for (int i = 0; i < hiddenSize; ++i) {
        output += hiddenLayerOutput[i] * weightsHiddenOutput[i];
    }
    return sigmoid(output);
}

// Backward propagation step to update weights and biases
// Takes learningRate as a parameter to adjust the extent of updates
void NeuralNetwork::backward(const std::vector<double>& input, const std::vector<double>& hiddenLayerOutput, double output, double target, double learningRate) {
    // Compute output layer error
    double outputError = output - target;
    double outputGradient = outputError * sigmoidDerivative(output);

    // Update weights and bias for output layer
    for (int i = 0; i < hiddenSize; ++i) {
        weightsHiddenOutput[i] -= learningRate * outputGradient * hiddenLayerOutput[i];
    }
    biasOutput -= learningRate * outputGradient;

    // Backpropagate error to hidden layer
    std::vector<double> hiddenErrors(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        hiddenErrors[i] = outputGradient * weightsHiddenOutput[i] * reluDerivative(hiddenLayerOutput[i]);
    }

    // Update weights and biases for hidden layer
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weightsInputHidden[i][j] -= learningRate * hiddenErrors[i] * input[j];
        }
        biasHidden[i] -= learningRate * hiddenErrors[i];
    }
}

// Train the neural network using the training data
// Iteratively adjusts weights using gradient descent and backpropagation
void NeuralNetwork::train(Dataset& trainData, std::unordered_map<std::string, int> vocabulary, int epochs, double learningRate, Dataset& devData, bool verbose) {
    std::vector<std::vector<double>> trainFeatures;
    std::vector<int> trainLabels;

    // Convert training data to feature vectors
    for (const auto& sample : trainData.getData()) {
        trainFeatures.push_back(TextPreprocessor::createFeatureVector(sample.getTokens(), vocabulary));
        trainLabels.push_back(sample.getLabel());
    }

    // Training loop over the specified number of epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < trainFeatures.size(); ++i) {
            std::vector<double> hiddenLayerOutput(hiddenSize);
            double output = forward(trainFeatures[i], hiddenLayerOutput);
            double target = trainLabels[i];

            // Calculate loss (Mean Squared Error)
            double loss = (output - target) * (output - target);
            totalLoss += loss;

            // Perform backpropagation
            backward(trainFeatures[i], hiddenLayerOutput, output, target, learningRate);

//            // Print progress
//            std::cout << "\rEpoch " << epoch + 1 << "/" << epochs
//                      << ", Sample " << i + 1 << "/" << trainFeatures.size() << std::flush;
        }

        totalLoss /= trainFeatures.size();

        if (verbose) {
            std::cout << "\nEpoch " << epoch + 1 << ", Loss: " << totalLoss << std::endl;
            double accuracy = evaluate(devData, vocabulary);
            std::cout << "Validation Accuracy: " << accuracy << "%" << std::endl;
        }
    }
}

// Prediction function for a single input
// Returns 1 for positive and 0 for negative based on the output
int NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> hiddenLayerOutput(hiddenSize);
    double output = forward(input, hiddenLayerOutput);
    return output >= 0.5 ? 1 : 0;
}

// Evaluate the accuracy of the network on the validation dataset
// Compares predicted labels with true labels to calculate accuracy
double NeuralNetwork::evaluate(Dataset& devData, std::unordered_map<std::string, int> vocabulary) {
    std::vector<std::vector<double>> devFeatures;
    std::vector<int> devLabels;

    // Convert validation data to feature vectors
    for (const auto& sample : devData.getData()) {
        devFeatures.push_back(TextPreprocessor::createFeatureVector(sample.getTokens(), vocabulary));
        devLabels.push_back(sample.getLabel());
    }

    int correctPredictions = 0;

    // Iterate over all validation samples and make predictions
    for (size_t i = 0; i < devFeatures.size(); ++i) {
        int prediction = predict(devFeatures[i]);
        if (prediction == devLabels[i]) {
            correctPredictions++;
        }
    }

    double accuracy = 100.0 * correctPredictions / devFeatures.size();

    return accuracy;
}

// Save the model weights and biases to a binary file
// Includes progress indication to inform the user of the saving process
void NeuralNetwork::saveWeights(const std::string& filename) const {
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file for saving weights: " << filename << std::endl;
        return;
    }

    // Save model parameters (inputSize, hiddenSize)
    outFile.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
    outFile.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(hiddenSize));

    // Save input-hidden weights
    for (const auto& row : weightsInputHidden) {
        for (double weight : row) {
            outFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
    }

    // Save hidden biases
    for (double bias : biasHidden) {
        outFile.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
    }

    // Save hidden-output weights
    for (double weight : weightsHiddenOutput) {
        outFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    }

    // Save output bias
    outFile.write(reinterpret_cast<const char*>(&biasOutput), sizeof(biasOutput));

    outFile.close();
    std::cout << "Model weights saved to " << filename << std::endl;
}

// Load model weights and biases from a binary file
// Includes checks to ensure the model structure matches the saved model
bool NeuralNetwork::loadWeights(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);

    if (!inFile.is_open()) {
        std::cerr << "Error opening file for loading weights: " << filename << std::endl;
        return false;
    }

    // Load model parameters and check for consistency
    int loadedInputSize, loadedHiddenSize;
    inFile.read(reinterpret_cast<char*>(&loadedInputSize), sizeof(loadedInputSize));
    inFile.read(reinterpret_cast<char*>(&loadedHiddenSize), sizeof(loadedHiddenSize));

    if (loadedInputSize != inputSize || loadedHiddenSize != hiddenSize) {
        std::cerr << "Model parameters do not match: "
                  << "Expected (inputSize: " << inputSize << ", hiddenSize: " << hiddenSize << "), "
                  << "but got (inputSize: " << loadedInputSize << ", hiddenSize: " << loadedHiddenSize << ")." << std::endl;
        inFile.close();
        return false;
    }

    // Load input-hidden weights
    for (auto& row : weightsInputHidden) {
        for (double& weight : row) {
            inFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        }
    }

    // Load hidden biases
    for (double& bias : biasHidden) {
        inFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));
    }

    // Load hidden-output weights
    for (double& weight : weightsHiddenOutput) {
        inFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));
    }

    // Load output bias
    inFile.read(reinterpret_cast<char*>(&biasOutput), sizeof(biasOutput));

    inFile.close();
    std::cout << "Model weights loaded from " << filename << std::endl;
    return true;
}
