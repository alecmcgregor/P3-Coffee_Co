#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include "DecisionTree.h"

int main() {
    std::ifstream file("coffee.csv");
    if (!file.is_open()) {
        std::cerr << "Failed to open coffee.csv\n";
        return 1;
    }

    std::string header;
    std::getline(file, header);
    std::vector<std::string> cols;
    std::stringstream hs(header);
    std::string col;
    while (std::getline(hs, col, ',')) cols.push_back(col);

    auto idx_of = [&](const std::string& name) {
        auto it = std::find(cols.begin(), cols.end(), name);
        return (it == cols.end() ? -1 : int(std::distance(cols.begin(), it)));
    };

    std::vector<std::string> feats = {
        "Data.Scores.Aroma","Data.Scores.Aftertaste","Data.Scores.Acidity",
        "Data.Scores.Body","Data.Scores.Balance","Data.Scores.Uniformity",
        "Data.Scores.Sweetness","Data.Scores.Moisture"
    };
    std::vector<int> fidx;
    for (auto& f : feats) {
        int i = idx_of(f);
        if (i < 0) { std::cerr << "Column " << f << " missing\n"; return 1; }
        fidx.push_back(i);
    }
    int tidx = idx_of("Data.Scores.Flavor");
    if (tidx < 0) { std::cerr << "Column Flavor missing\n"; return 1; }

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ls(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ls, cell, ',')) row.push_back(cell);
        if (row.size() <= size_t(tidx)) continue;
        bool bad = false;
        std::vector<double> xv;
        for (int i : fidx) {
            if (row[i].empty()) { bad = true; break; }
            xv.push_back(std::stod(row[i]));
        }
        if (bad || row[tidx].empty()) continue;
        X.push_back(xv);
        y.push_back(std::stod(row[tidx]));
    }
    file.close();

    int n = X.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937{std::random_device{}()});
    int train_n = int(0.8 * n);

    std::vector<std::vector<double>> Xtr, Xte;
    std::vector<double> ytr, yte;
    for (int i = 0; i < n; ++i) {
        if (i < train_n)      { Xtr.push_back(X[idx[i]]); ytr.push_back(y[idx[i]]); }
        else                  { Xte.push_back(X[idx[i]]); yte.push_back(y[idx[i]]); }
    }

    DecisionTreeRegressor model(6, 5);
    model.fit(Xtr, ytr);
    auto pred = model.predict(Xte);

    double ss_res = 0, ss_tot = 0, mean_y = 0;
    for (double v : yte) mean_y += v;
    mean_y /= yte.size();
    for (size_t i = 0; i < yte.size(); ++i) {
        ss_res += (yte[i] - pred[i]) * (yte[i] - pred[i]);
        ss_tot += (yte[i] - mean_y) * (yte[i] - mean_y);
    }
    double rmse = std::sqrt(ss_res / yte.size());
    double r2   = 1 - ss_res / ss_tot;

    std::cout << "Test RMSE: " << rmse << "\n";
    std::cout << "Test R2: "   << r2   << "\n";
    return 0;
}
