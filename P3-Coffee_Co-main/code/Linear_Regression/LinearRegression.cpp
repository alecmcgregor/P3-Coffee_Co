#include "LinearRegression.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <limits>

using namespace std;

LinearRegression::LinearRegression(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Error: File could not be opened.\n";
        return;
    }

    string header;
    getline(file, header); // Skip column labels

    string line;
    while (getline(file, line)) {
        istringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ',')) {
            row.push_back(cell.empty() ? 0.0 : stod(cell));
        }

        if (row.size() < 2) continue;

        dataset.push_back(row);
        output.push_back(row[1]);

        vector<double> features_row;
        features_row.reserve(row.size() - 1);
        for (size_t i = 0; i < row.size(); ++i) {
            if (i != 1) features_row.push_back(row[i]); // exclude flavor score
        }
        matrix.push_back(features_row);
    }

    file.close();

    samples = matrix.size();
    features = matrix[0].size();
    weights.assign(features, 0.0);
}

void LinearRegression::train(const double& alpha, const int& iterations) {
    const double mseTolerance = 1e-6;
    const double gradientTolerance = 1e-4;
    double prev_mse = std::numeric_limits<double>::max();

    for (int iter = 0; iter < iterations; ++iter) {
        vector<double> predicted(samples);
        for (int i = 0; i < samples; ++i) {
            predicted[i] = inner_product(matrix[i].begin(), matrix[i].end(), weights.begin(), bias);
        }

        vector<double> errors(samples);
        double mse = 0.0;
        for (int i = 0; i < samples; ++i) {
            errors[i] = predicted[i] - output[i];
            mse += errors[i] * errors[i];
        }
        mse /= samples;

        vector<double> gradient(features, 0.0);
        double bias_grad = 0.0;
        for (int i = 0; i < samples; ++i) {
            bias_grad += errors[i];
            for (int j = 0; j < features; ++j) {
                gradient[j] += errors[i] * matrix[i][j];
            }
        }

        bias_grad = 2.0 * bias_grad / samples;
        for (double& g : gradient) g = 2.0 * g / samples;

        double grad_norm = sqrt(inner_product(gradient.begin(), gradient.end(), gradient.begin(), 0.0));

        if (iter > 0 && (abs(mse - prev_mse) < mseTolerance || grad_norm < gradientTolerance)) {
            cout << "\nConverged at iteration " << iter << endl;
            break;
        }

        for (int j = 0; j < features; ++j) {
            weights[j] -= alpha * gradient[j];
        }
        bias -= alpha * bias_grad;

        prev_mse = mse;

        if (iter % 500 == 0 || iter == iterations - 1) {
            cout << "\rIteration: " << iter << " | RMSE: " << sqrt(mse) << flush;
        }
    }

    cout << "\nFinal RMSE: " << sqrt(prev_mse) << endl;
}

double LinearRegression::predict(const vector<double>& sample) const {
    return inner_product(sample.begin(), sample.end(), weights.begin(), bias);
}

void LinearRegression::printWeights() {
    cout << "Feature Coefficients:\n";
    const vector<string> labels = {
        "Aroma", "Aftertaste", "Acidity", "Body",
        "Balance", "Uniformity", "Sweetness", "Moisture"
    };

    for (int i = 0; i < features && i < (int)labels.size(); ++i) {
        cout << setw(12) << labels[i];
    }
    cout << "\n";

    for (double w : weights) {
        cout << setw(12) << w;
    }

    cout << "\nBias: " << bias << endl;
}

void LinearRegression::saveResults() const {
    // Save predictions vs actual
    std::ofstream predfile("linear_predictions.csv");
    predfile << "Predicted,Actual\n";
    for (int i = 0; i < samples; ++i) {
        double yhat = predict(matrix[i]);
        predfile << yhat << "," << output[i] << "\n";
    }
    predfile.close();

    // Save weights
    std::ofstream wfile("linear_weights.csv");
    for (double w : weights) {
        wfile << w << "\n";
    }
    wfile.close();
}