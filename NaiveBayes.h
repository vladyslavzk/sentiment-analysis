#ifndef SENTIMENTANALYSIS_NAIVEBAYES_H
#define SENTIMENTANALYSIS_NAIVEBAYES_H

#include "Twitter.h"
#include <unordered_map>
#include <unordered_set>
#include <math.h>

class NaiveBayes {
private:

    unordered_set<string> stopwords;
    unordered_map<string, int> wordCountPositive;
    unordered_map<string, int> wordCountNegative;
    unordered_map<string, int> totalWordCount;
    unordered_set<string> vocabulary;

    int totalPositiveWords;
    int totalNegativeWords;

    double log_prior_positive;
    double log_prior_negative;

    std::unordered_map<std::string, double> log_likelihood_positive;
    std::unordered_map<std::string, double> log_likelihood_negative;

    void calculateWordCounts(Dataset &train);

    unordered_map<string, double>
    calculateLikelihood(unordered_map<string, int> wordCount, int totalWords, double laplacian_smoothing = 1.0);

    static unordered_map<string, double> calculateLogLikelihood(unordered_map<string, double> likelihood);

    void calculateLogPrior(Dataset &dataset);


public:

    void loadStopwords(string filename);

    void train(Dataset &train, double laplace = 1.0);

    int predict(const std::string &text);

    double evaluate(Dataset &dataset);
};


#endif //SENTIMENTANALYSIS_NAIVEBAYES_H
