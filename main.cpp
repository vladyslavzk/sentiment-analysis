#include <iostream>
#include "NaiveBayes.h"

// Define the label strings corresponding to the indices
const std::string labels[] = { "positive", "negative", "neutral" };

int main() {

    // Loading datasets
    Twitter twitter;
    twitter.loadStopwords("../data/stopwords.txt");
    twitter.loadTrainData("../data/twitter_training.csv");
    twitter.loadDevData("../data/twitter_validation.csv");

    // Training a model
    NaiveBayes nb;
    nb.loadStopwords("../data/stopwords.txt");
    double laplace = 0.5;
    nb.train(twitter.getTrainData(), laplace);

    // Evaluation a model with a validation dataset
    cout << "Accuracy on a validation dataset is " << nb.evaluate(twitter.getDevData()) << "%"<<endl;

    // Examples
    string input_text = "I'm excited to play this game"; // positive
    int predicted_index = nb.predict(input_text);
    std::cout << "Predicted label for \"" << input_text << "\": " << labels[predicted_index] << std::endl;

    input_text = "I hate this game"; // negative
    predicted_index = nb.predict(input_text);
    std::cout << "Predicted label for \"" << input_text << "\": " << labels[predicted_index] << std::endl;

    input_text = "He is playing on a computer"; // neutral
    predicted_index = nb.predict(input_text);
    std::cout << "Predicted label for \"" << input_text << "\": " << labels[predicted_index] << std::endl;

    return 0;
}