#include "LinearRegression.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
using namespace std;

LinearRegression::LinearRegression(const string& path) {
    //Read the file in
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "File could not be opened" << endl;
        return;
    }

    //ignore the column labels so that you're only reading in the numeric data
    string header;
    getline(file, header);

    //read all the data in from each row and col and push everything in to the dataset, features into the matrix, and flavor values into the output
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

    //initialize number of samples, features, and vector of weights
    samples = matrix.size();
    features = matrix[0].size();
    weights.resize(features, 0.0);
}

void LinearRegression::train(const double& alpha, const int& iterations) {
    //Initialize tolerances to make sure the model doesn't converge
    double mseTolerance = 1e-6;
    double gradientTolerance = 1e-4;
    double prev_mse = 0.0;

    //Run the training loop for the number of iterations inputted
    for (int k = 0; k < iterations; k++) {

        //generate all the predicted values and place them in a vector
        vector<double> predicted;
        for (int i = 0; i < samples; i++) {
            double result = 0.0;
            for (int j = 0; j < features; j++) {
                result += matrix[i][j] * weights[j];
            }
            result += bias;
            predicted.push_back(result);
        }

        // calculate the mean squared error
        double mse = 0.0;
        for (int i = 0; i < samples; i++) {
            mse +=  (output[i] - predicted[i]) * (output[i] - predicted[i]);
        }
        mse /= samples;

        //calculate the gradient of the weights and the bias
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

        // Compute the L2 norm to determine how much the weights are actually changing, then check to make sure that the model loss stabilizes or the gradient flattens out
        double num = 0.0;
        for (int i = 0; i < features; i++) {
            num += gradient[i] * gradient[i];
        }
        num = sqrt(num);
        if (k > 0 && (abs(mse - prev_mse) < mseTolerance || num < gradientTolerance)) {
            break;
        }
        prev_mse = mse;

        //update the bias and the weights at the end of the loop
        bias -= alpha * avgerr;
        for (int i = 0; i < features; i++) {
            weights[i] -= alpha * gradient[i];
        }

        //Progress display to know that the code is running
        cout << "\rIteration: " << k << flush;
    }
    cout << endl;
    cout << "RMSE: " << sqrt(prev_mse) << endl;
}

double LinearRegression::predict(const vector<double>& sample) {
    //predicts the output for a single given sample, aka returns the predicted flavor score for a given bean sample
    double result = 0.0;
    for (int i = 0; i < features; i++) {
        result += sample[i] *weights[i];
    }
    result += bias;
    return result;
}

void LinearRegression::printWeights() {
    // iterate over the weights vector and print out the values as well as the bias
    cout << "Feature Coefficients: " << endl;
    cout << setw(12) << "Aroma" << setw(12) << "Aftertaste" << setw(12) << "Acidity" << setw(12) << "Body" << setw(12) << "Balance" << setw(12) << "Uniformity" << setw(12) << "Sweetness" << setw(12) << "Moisture" << endl;
    for (int i = 0; i < features; i++) {
        cout << setw(12) << weights[i];
    }
    cout << endl;
    cout << "Bias: " << bias << endl;
}