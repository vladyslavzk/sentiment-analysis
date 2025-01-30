#include <iostream>
#include <string>
#include <fstream>
#include "LogisticRegression.h"
#include "SimpleSVM.h"
#include "NaiveBayes.h"
#include "NeuralNetwork.h"
#include "Twitter.h"


void displayMenu() {
    std::cout << "Welcome to the Sentiment Analysis Program!" << std::endl;
    std::cout << "Please choose a method for sentiment analysis:" << std::endl;
    std::cout << "1. Naive Bayes" << std::endl;
    std::cout << "2. Logistic Regression" << std::endl;
    std::cout << "3. Support Vector Machine (SVM)" << std::endl;
    std::cout << "4. Neural Network" << std::endl;
    std::cout << "0. Exit" << std::endl;
}

std::string promptSaveLoadModel() {
    std::string option;
    while (true) {
        std::cout << "Would you like to load a pretrained model or train a new one?" << std::endl;
        std::cout << "1. Load pretrained model" << std::endl;
        std::cout << "2. Train a new model" << std::endl;
        std::cout << "Enter your choice (1/2): ";
        std::cin >> option;
        if (option == "1" || option == "2") {
            break;
        } else {
            std::cout << "Invalid input. Please enter 1 or 2." << std::endl;
        }
    }
    return option;
}

std::string promptFilename(const std::string& action) {
    std::string filename;
    std::cout << "Enter the filename to " << action << " the model: ";
    std::cin >> filename;
    return filename;
}

std::pair<int, int> promptDatasetSize() {
    std::pair<int, int> n_sentences;
    std::cout << "Enter the number of sentences to load from train dataset (-1 for all sentences): ";
    std::cin >> n_sentences.first;
    std::cout << "Enter the number of sentences to load from validation dataset (-1 for all sentences): ";
    std::cin >> n_sentences.second;
    return n_sentences;
}

void predictTextSentiment(NaiveBayes& model) {
    std::string text;
    std::cout << "Enter a text to analyze sentiment (type 'exit' to return to the main menu): ";
    std::cin.ignore();
    while (true) {
        std::getline(std::cin, text);
        if (text == "exit") {
            break;
        }
        auto tokens = TextPreprocessor::preprocess(text, TextPreprocessor::readStopwords("../data/stopwords.txt"));
        int prediction = model.predict(text);
        std::string sentiment = prediction == 1 ? "Positive" : "Negative";
        std::cout << "Predicted sentiment: " << sentiment << std::endl;
        std::cout << "Enter another text to analyze sentiment (type 'exit' to return to the main menu): ";
    }
}


void predictTextSentiment(LogisticRegression& model, const std::unordered_map<std::string, int>& vocabulary) {
    std::string text;
    std::cout << "Enter a text to analyze sentiment (type 'exit' to return to the main menu): ";
    std::cin.ignore();  // to ignore any leftover newline character
    while (true) {
        std::getline(std::cin, text);
        if (text == "exit") {
            break;
        }
        auto tokens = TextPreprocessor::preprocess(text, TextPreprocessor::readStopwords("../data/stopwords.txt"));
        int prediction = model.predict(tokens, vocabulary);
        std::string sentiment = prediction == 1 ? "Positive" : "Negative";
        std::cout << "Predicted sentiment: " << sentiment << std::endl;
        std::cout << "Enter another text to analyze sentiment (type 'exit' to return to the main menu): ";
    }
}

void predictTextSentiment(SimpleSVM& model, const std::unordered_map<std::string, int>& vocabulary) {
    std::string text;
    std::cout << "Enter a text to analyze sentiment (type 'exit' to return to the main menu): ";
    std::cin.ignore();
    while (true) {
        std::getline(std::cin, text);
        if (text == "exit") {
            break;
        }
        auto tokens = TextPreprocessor::preprocess(text, TextPreprocessor::readStopwords("../data/stopwords.txt"));
        int prediction = model.predict(tokens, vocabulary);
        std::string sentiment = prediction == 1 ? "Positive" : "Negative";
        std::cout << "Predicted sentiment: " << sentiment << std::endl;
        std::cout << "Enter another text to analyze sentiment (type 'exit' to return to the main menu): ";
    }
}

void predictTextSentiment(NeuralNetwork& model, const std::unordered_map<std::string, int>& vocabulary) {
    std::string text;
    std::cout << "Enter a text to analyze sentiment (type 'exit' to return to the main menu): ";
    std::cin.ignore();
    while (true) {
        std::getline(std::cin, text);
        if (text == "exit") {
            break;
        }
        auto tokens = TextPreprocessor::preprocess(text, TextPreprocessor::readStopwords("../data/stopwords.txt"));
        auto featureVector = TextPreprocessor::createFeatureVector(tokens, vocabulary);
        int prediction = model.predict(featureVector);
        std::string sentiment = prediction == 1 ? "Positive" : "Negative";
        std::cout << "Predicted sentiment: " << sentiment << std::endl;
        std::cout << "Enter another text to analyze sentiment (type 'exit' to return to the main menu): ";
    }
}

void trainNaiveBayes(Twitter& twitter, Dataset& trainData, Dataset& devData) {
    NaiveBayes nb;
    std::cout << "Training Naive Bayes..." << std::endl;
    nb.train(trainData);
    double accuracy = nb.evaluate(devData);
    std::cout << "Validation Accuracy: " << accuracy << "%" << std::endl;

    predictTextSentiment(nb);
}


void trainLogisticRegression(Twitter& twitter, Dataset& trainData, Dataset& devData) {
    LogisticRegression lr(trainData.createVocabulary().size());
    double learningRate = 0.01;
    int epochs = 100;

    std::string option = promptSaveLoadModel();
    if (option == "1") {
        if (lr.loadWeights("../saved_models/lr_weights.bin")) {
            std::cout << "Model loaded successfully." << std::endl;
        } else {
            std::cout << "Failed to load the model. Proceeding with training." << std::endl;
        }
    } else {
        std::cout << "Set learning rate (default 0.1): ";
        std::cin >> learningRate;

        std::cout << "Set number of epochs (default 100): ";
        std::cin >> epochs;

        std::cout << "Training Logistic Regression..." << std::endl;
        lr.train(trainData, trainData.createVocabulary(), devData, learningRate, epochs, true);

        std::cout << "Save the model? (yes/no): ";
        std::string save;
        std::cin >> save;
        if (save == "yes") {
            lr.saveWeights("../saved_models/lr_weights.bin");
        }
    }
    predictTextSentiment(lr, trainData.createVocabulary());
}

void trainSVM(Twitter& twitter, Dataset& trainData, Dataset& devData) {
    SimpleSVM svm;
    double learningRate = 0.01;
    int epochs = 100;
    double regularizationParam = 0.01;

    std::string option = promptSaveLoadModel();
    if (option == "1") {
        if (svm.loadWeights("../saved_models/svm_weights.bin")) {
            std::cout << "Model loaded successfully." << std::endl;
        } else {
            std::cout << "Failed to load the model. Proceeding with training." << std::endl;
        }
    } else {
        std::cout << "Set learning rate (default 0.01): ";
        std::cin >> learningRate;

        std::cout << "Set number of epochs (default 100): ";
        std::cin >> epochs;

        std::cout << "Set regularization parameter (default 0.00001): ";
        std::cin >> regularizationParam;

        std::cout << "Training SVM..." << std::endl;
        svm.train(trainData, trainData.createVocabulary(), devData, learningRate, epochs, regularizationParam, true);

        std::cout << "Save the model? (yes/no): ";
        std::string save;
        std::cin >> save;
        if (save == "yes") {
            svm.saveWeights("../saved_models/svm_weights.bin");
        }
    }
    predictTextSentiment(svm, trainData.createVocabulary());
}

void trainNeuralNetwork(Twitter& twitter, Dataset& trainData, Dataset& devData) {
    int inputSize = trainData.createVocabulary().size();
    int hiddenSize = 10;
    double learningRate = 0.01;
    int epochs = 100;

    std::string option = promptSaveLoadModel();
    NeuralNetwork nn(inputSize, hiddenSize);

    if (option == "1") {
        if (nn.loadWeights("../saved_models/nn_weights.bin")) {
            std::cout << "Model loaded successfully." << std::endl;
        } else {
            std::cout << "Failed to load the model. Proceeding with training." << std::endl;
        }
    } else {
        std::cout << "Set hidden layer size (default 10): ";
        std::cin >> hiddenSize;

        std::cout << "Set learning rate (default 0.01): ";
        std::cin >> learningRate;

        std::cout << "Set number of epochs (default 100): ";
        std::cin >> epochs;

        std::cout << "Training Neural Network..." << std::endl;
        nn.train(trainData, trainData.createVocabulary(), epochs, learningRate, devData, true);

        std::cout << "Save the model? (yes/no): ";
        std::string save;
        std::cin >> save;
        if (save == "yes") {
            nn.saveWeights("../saved_models/nn_weights.bin");
        }
    }
    predictTextSentiment(nn, trainData.createVocabulary());
}

int main() {
    Twitter twitter;
    twitter.loadStopwords("../data/stopwords.txt");

    std::string trainFile = "../data/twitter_training.csv";
    std::string devFile = "../data/twitter_validation.csv";

    std::pair<int, int> n_sentences = promptDatasetSize();
    twitter.loadTrainData(trainFile, n_sentences.first);
    twitter.loadDevData(devFile, n_sentences.second);

    Dataset& trainData = twitter.getTrainData();
    Dataset& devData = twitter.getDevData();

    while (true) {
        displayMenu();

        int choice;
        std::cout << "Enter your choice: ";
        std::cin >> choice;

        switch (choice) {
            case 1:
                trainNaiveBayes(twitter, trainData, devData);
                break;
            case 2:
                trainLogisticRegression(twitter, trainData, devData);
                break;
            case 3:
                trainSVM(twitter, trainData, devData);
                break;
            case 4:
                trainNeuralNetwork(twitter, trainData, devData);
                break;
            case 0:
                std::cout << "Exiting..." << std::endl;
                return 0;
            default:
                std::cout << "Invalid choice. Please try again." << std::endl;
                break;
        }
    }
}
