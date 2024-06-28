cmake_minimum_required(VERSION 3.27)
project(sentimentanalysis)

set(CMAKE_CXX_STANDARD 17)

add_executable(sentimentanalysis main.cpp
        DSText.cpp
        DSText.h
        Dataset.cpp
        Dataset.h
        Twitter.cpp
        Twitter.h
        TextPreprocessor.cpp
        TextPreprocessor.h
        NaiveBayes.cpp
        NaiveBayes.h
)
