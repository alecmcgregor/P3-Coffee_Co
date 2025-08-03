#pragma once
#include <iostream>
#include <vector>
using namespace std;

class LinearRegression {
private:
    vector<vector<double>> dataset;
    vector<double> weights;
    double bias = 0.0;
    vector<vector<double>> matrix;
    vector<double> output;
    int features = 0;
    int samples = 0;
public:
    LinearRegression(const string& path);
    void train(const double& alpha, const int& iterations);
    double predict(const vector<double>& sample);
    void printWeights();
};