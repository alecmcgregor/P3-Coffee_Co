//
// Created by alecm on 7/31/2025.
//
#include "LinearRegression.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
using namespace std;

LinearRegression::LinearRegression(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "File could not be opened" << endl;
        return;
    }
    string header;
    getline(file, header);
    string line;
    while (getline(file, line)) {
        vector<double> row;
        istringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        dataset.push_back(row);
        output.push_back(row[1]);
        vector<double> temp;
        for (int i = 0; i < row.size(); i++) {
            if (i != 1) {
                temp.push_back(row[i]);
            }
        }
        matrix.push_back(temp);
    }
    file.close();
    samples = matrix.size();
    features = matrix[0].size();
    weights.resize(features, 0.0);
}

void LinearRegression::train(const double& alpha, const int& iterations) {
    double mseTolerance = 1e-6;
    double gradientTolerance = 1e-4;
    double prev_mse = 0.0;

    for (int k = 0; k < iterations; k++) {
        vector<double> predicted;
        for (int i = 0; i < samples; i++) {
            double result = 0.0;
            for (int j = 0; j < features; j++) {
                result += matrix[i][j] * weights[j];
            }
            result += bias;
            predicted.push_back(result);
        }

        double mse = 0.0;
        for (int i = 0; i < samples; i++) {
            mse +=  (output[i] - predicted[i]) * (output[i] - predicted[i]);
        }
        mse /= samples;

        cout << k <<"    " << mse << endl;

        vector<double> gradient(features, 0.0);
        double avgerr = 0.0;
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                gradient[j] += (predicted[i] - output[i]) * matrix[i][j];
            }
            avgerr += predicted[i] - output[i];
        }
        for (int i = 0; i < features; i++) {
            gradient[i] *= 2.0 / samples;
        }
        avgerr *= 2.0 / samples;

        double num = 0.0;
        for (int i = 0; i < features; i++) {
            num += gradient[i] * gradient[i];
        }
        num = sqrt(num);
        if (k > 0 && (abs(mse - prev_mse) < mseTolerance || num < gradientTolerance)) {
            break;
        }
        prev_mse = mse;

        bias -= alpha * avgerr;
        for (int i = 0; i < features; i++) {
            weights[i] -= alpha * gradient[i];
        }
    }
}

double LinearRegression::predict(const vector<double>& sample) {
    double result = 0.0;
    for (int i = 0; i < features; i++) {
        result += sample[i] *weights[i];
    }
    result += bias;
    return result;
}

void LinearRegression::printWeights() {
    for (int i = 0; i < features; i++) {
        cout << weights[i] << endl;
    }
    cout << bias << endl;
}