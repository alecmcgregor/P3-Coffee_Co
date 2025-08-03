#include "LinearRegression.h"
#include <iostream>
using namespace std;

int main(int argc, char* argv []) {
    const string path = argv[1];
    LinearRegression model(path);
    const string alpha = argv[2];
    const string iterations = argv[3];
    model.train(stod(alpha), stoi(iterations));
    model.printWeights();
    return 0;
}