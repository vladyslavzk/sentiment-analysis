#include "NaiveBayes.h"

// Load stopwords from a file and store them in an unordered_set
void NaiveBayes::loadStopwords(string filename) {
    stopwords = TextPreprocessor::readStopwords("../data/stopwords.txt");
}

// Calculate the word counts for positive and negative classes in the training dataset
// Builds a vocabulary and tracks the total number of words for each class
void NaiveBayes::calculateWordCounts(Dataset &train) {
    for (auto tokens : train.getData()) {
        int label = tokens.getLabel();
        for (const auto &token : tokens.getTokens()) {
            vocabulary.insert(token); // Add the token to the vocabulary

            if (label == 1) {
                wordCountPositive[token]++;
                totalPositiveWords++;
            } else if (label == 0) {
                wordCountNegative[token]++;
                totalNegativeWords++;
            }
        }
    }
}

// Calculate the likelihood of each word given the class (positive/negative)
// Applies Laplace smoothing to handle words that may not be in the training data
unordered_map<string, double> NaiveBayes::calculateLikelihood(unordered_map<string, int> wordCount, int totalWords, double laplacian_smoothing) {
    unordered_map<string, double> likelihood;
    double vocabulary_size = vocabulary.size();

    for (auto item : wordCount) {
        string token = item.first;
        double count = item.second;

        likelihood[token] = (count + laplacian_smoothing) / (totalWords + vocabulary_size * laplacian_smoothing);
    }

    // Ensure all vocabulary words are in the likelihood map
    for (const auto &token : vocabulary) {
        if (likelihood.find(token) == likelihood.end()) {
            likelihood[token] = laplacian_smoothing / (totalWords + vocabulary_size * laplacian_smoothing);
        }
    }

    return likelihood;
}

// Convert the likelihoods to log-likelihoods to avoid numerical underflow issues
// This is done because probabilities can become very small, leading to potential precision problems
unordered_map<string, double> NaiveBayes::calculateLogLikelihood(unordered_map<string, double> likelihood) {
    unordered_map<string, double> log_likelihood;

    for (auto item : likelihood) {
        string token = item.first;
        double prob = item.second;
        log_likelihood[token] = log(prob);
    }
    return log_likelihood;
}

// Calculate the prior probabilities for positive and negative classes based on the dataset
// The prior is the probability of each class without any additional information
void NaiveBayes::calculateLogPrior(Dataset &dataset) {
    int total_tweets = dataset.getData().size();
    int positive_tweets = count_if(dataset.getData().begin(), dataset.getData().end(),
                                   [](const DSText &text) { return text.getLabel() == 1; });
    int negative_tweets = count_if(dataset.getData().begin(), dataset.getData().end(),
                                   [](const DSText &text) { return text.getLabel() == 0; });

    log_prior_positive = log(static_cast<double>(positive_tweets) / total_tweets);
    log_prior_negative = log(static_cast<double>(negative_tweets) / total_tweets);
}

// Train the Naive Bayes model by calculating word counts, likelihoods, and priors
// Uses Laplace smoothing to handle cases where a word is not observed in a class
void NaiveBayes::train(Dataset &train, double laplace) {
    totalPositiveWords = 0;
    totalNegativeWords = 0;

    calculateWordCounts(train);
    auto likelihood_positive = calculateLikelihood(wordCountPositive, totalPositiveWords, laplace);
    auto likelihood_negative = calculateLikelihood(wordCountNegative, totalNegativeWords, laplace);

    log_likelihood_positive = calculateLogLikelihood(likelihood_positive);
    log_likelihood_negative = calculateLogLikelihood(likelihood_negative);

    calculateLogPrior(train);
}

// Predict the sentiment of a given text by calculating the log-probabilities for each class
// The class with the higher log-probability is chosen as the prediction
int NaiveBayes::predict(const std::string &text) {
    vector<string> tokens = TextPreprocessor::preprocess(text, stopwords);

    double positive_score = log_prior_positive;
    double negative_score = log_prior_negative;

    // Sum log-likelihoods for each word in the text
    for (const auto &token : tokens) {
        if (log_likelihood_positive.find(token) != log_likelihood_positive.end()) {
            positive_score += log_likelihood_positive[token];
        }

        if (log_likelihood_negative.find(token) != log_likelihood_negative.end()) {
            negative_score += log_likelihood_negative[token];
        }
    }

    // Return the label with the higher score
    return positive_score >= negative_score ? 1 : 0;
}

// Evaluate the accuracy of the model on a given dataset
// Compares predicted labels with true labels to compute the accuracy
double NaiveBayes::evaluate(Dataset &dataset) {
    int correct_predictions = 0;
    int total_predictions = 0;

    for (const DSText &text : dataset.getData()) {
        std::string combined_text;
        for (const std::string &token : text.getTokens()) {
            combined_text += token + " ";
        }
        int predicted_label = predict(combined_text);
        if (predicted_label == text.getLabel()) {
            correct_predictions++;
        }
        total_predictions++;
    }

    return 100 * static_cast<double>(correct_predictions) / total_predictions;
}
