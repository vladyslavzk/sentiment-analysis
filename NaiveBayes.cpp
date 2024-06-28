#include "NaiveBayes.h"


void NaiveBayes::loadStopwords(string filename) {
    stopwords = TextPreprocessor::readStopwords("../data/stopwords.txt");
}


// Create the word count table. Build a table that shows the occurrences of each word in the positive and negative tweets
void NaiveBayes::calculateWordCounts(Dataset &train) {
    for (auto tokens: train.getData()) {
        int label = tokens.getLabel();
        for (const auto &token: tokens.getTokens()) {
            vocabulary.insert(token);

            if (label == 0) {
                wordCountPositive[token]++;
            } else if (label == 1) {
                wordCountNegative[token]++;
            } else {
                wordCountNeutral[token]++;
            }
        }
    }
}


unordered_map<string, double> NaiveBayes::calculateLikelihood(unordered_map<string, int> wordCount, double laplacian_smoothing) {
    unordered_map<string, double> likelihood;
    double vocabulary_size = int(vocabulary.size());

    for (auto item: wordCount) {
        string token = item.first;
        double count = item.second;

        likelihood[token] = (count + laplacian_smoothing) / (vocabulary_size + wordCount.size() * laplacian_smoothing);
    }
    return likelihood;
}


unordered_map<string, double> NaiveBayes::calculateLogLikelihood(unordered_map<string, double> likelihood) {
    unordered_map<string, double> log_likelihood;

    for (auto item: likelihood) {
        string token = item.first;
        double prob = item.second;
        log_likelihood[token] = log(prob);
    }
    return log_likelihood;
}


void NaiveBayes::calculateLogPrior(Dataset &dataset) {
    int total_tweets = dataset.getData().size();
    int positive_tweets = count_if(dataset.getData().begin(), dataset.getData().end(),
                                   [](const DSText &text) { return text.getLabel() == 0; });
    int negative_tweets = count_if(dataset.getData().begin(), dataset.getData().end(),
                                   [](const DSText &text) { return text.getLabel() == 1; });
    int neutral_tweets = count_if(dataset.getData().begin(), dataset.getData().end(),
                                  [](const DSText &text) { return text.getLabel() == 2; });

    log_prior_positive = log(static_cast<double>(positive_tweets) / total_tweets);
    log_prior_negative = log(static_cast<double>(negative_tweets) / total_tweets);
    log_prior_neutral = log(static_cast<double>(neutral_tweets) / total_tweets);

}


// Function for training a model with a given dataset
void NaiveBayes::train(Dataset &train, double laplace) {
    calculateWordCounts(train);
    unordered_map<string, double> likelihood_positive = calculateLikelihood(wordCountPositive, laplace);
    unordered_map<string, double> likelihood_negative = calculateLikelihood(wordCountNegative, laplace);
    unordered_map<string, double> likelihood_neutral = calculateLikelihood(wordCountNeutral, laplace);

    log_likelihood_positive = calculateLogLikelihood(likelihood_positive);
    log_likelihood_negative = calculateLogLikelihood(likelihood_negative);
    log_likelihood_neutral = calculateLogLikelihood(likelihood_neutral);

    calculateLogPrior(train);
}


// Function for predicting a sentiment with a given text
int NaiveBayes::predict(const std::string &text) {
    vector<string> tokens = TextPreprocessor::preprocess(text, stopwords);

    double positive_score = log_prior_positive;
    double negative_score = log_prior_negative;
    double neutral_score = log_prior_neutral;

    for (const auto &token: tokens) {
        if (log_likelihood_positive.find(token) != log_likelihood_positive.end()) {
            positive_score += log_likelihood_positive[token];
        }
        if (log_likelihood_negative.find(token) != log_likelihood_negative.end()) {
            negative_score += log_likelihood_negative[token];
        }
        if (log_likelihood_neutral.find(token) != log_likelihood_neutral.end()) {
            neutral_score += log_likelihood_neutral[token];
        }
    }

    if (positive_score >= negative_score && positive_score >= neutral_score) {
        return 0;
    } else if (negative_score >= positive_score && negative_score >= neutral_score) {
        return 1;
    } else {
        return 2;
    }
}


// Function for calculating accuracy of predicted sentiment on a given dataset
double NaiveBayes::evaluate(Dataset &dataset) {
    int correct_predictions = 0;
    int total_predictions = 0;

    for (const DSText &text: dataset.getData()) {
        std::string combined_text;
        for (const std::string &token: text.getTokens()) {
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



