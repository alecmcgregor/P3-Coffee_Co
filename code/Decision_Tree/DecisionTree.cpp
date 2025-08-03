#include "DecisionTree.h"
#include <algorithm>
#include <limits>

DecisionTreeRegressor::DecisionTreeRegressor(int max_depth, int min_samples_split)
    : root_(nullptr), max_depth_(max_depth), min_samples_split_(min_samples_split) {}

void DecisionTreeRegressor::fit(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y) {
    root_ = buildTree(X, y, 0);
}

double DecisionTreeRegressor::predict(const std::vector<double>& x) const {
    return predictSample(root_.get(), x);
}

std::vector<double> DecisionTreeRegressor::predict(
    const std::vector<std::vector<double>>& X) const {
    std::vector<double> preds;
    preds.reserve(X.size());
    for (const auto& x : X) preds.push_back(predict(x));
    return preds;
}

std::unique_ptr<DecisionTreeRegressor::Node>
DecisionTreeRegressor::buildTree(const std::vector<std::vector<double>>& X,
                                  const std::vector<double>& y,
                                  int depth) {
    auto node = std::make_unique<Node>();
    if (depth >= max_depth_ || y.size() < (size_t)min_samples_split_ || variance(y) == 0.0) {
        node->is_leaf = true;
        node->prediction = mean(y);
        return node;
    }

    int best_feature = -1;
    double best_thresh = 0.0;
    double best_mse = std::numeric_limits<double>::infinity();
    std::vector<size_t> best_left, best_right;
    int n_features = X[0].size();

    for (int f = 0; f < n_features; ++f) {
        std::vector<double> vals;
        for (size_t i = 0; i < X.size(); ++i) vals.push_back(X[i][f]);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        for (size_t j = 1; j < vals.size(); ++j) {
            double th = 0.5 * (vals[j - 1] + vals[j]);
            std::vector<size_t> left_idx, right_idx;
            for (size_t i = 0; i < X.size(); ++i) {
                (X[i][f] <= th ? left_idx : right_idx).push_back(i);
            }
            if (left_idx.empty() || right_idx.empty()) continue;
            double mse = weightedMSE(y, left_idx, right_idx);
            if (mse < best_mse) {
                best_mse = mse;
                best_feature = f;
                best_thresh = th;
                best_left = left_idx;
                best_right = right_idx;
            }
        }
    }

    if (best_feature == -1) {
        node->is_leaf = true;
        node->prediction = mean(y);
        return node;
    }

    std::vector<std::vector<double>> Xl, Xr;
    std::vector<double> yl, yr;
    for (size_t idx : best_left) { Xl.push_back(X[idx]); yl.push_back(y[idx]); }
    for (size_t idx : best_right) { Xr.push_back(X[idx]); yr.push_back(y[idx]); }

    node->feature_index = best_feature;
    node->threshold = best_thresh;
    node->left = buildTree(Xl, yl, depth + 1);
    node->right = buildTree(Xr, yr, depth + 1);
    return node;
}

double DecisionTreeRegressor::predictSample(const Node* node,
                                            const std::vector<double>& x) const {
    if (node->is_leaf) return node->prediction;
    return (x[node->feature_index] <= node->threshold)
           ? predictSample(node->left.get(), x)
           : predictSample(node->right.get(), x);
}

double DecisionTreeRegressor::mean(const std::vector<double>& vals) {
    double sum = 0;
    for (double v : vals) sum += v;
    return sum / vals.size();
}

double DecisionTreeRegressor::variance(const std::vector<double>& vals) {
    double m = mean(vals), var = 0;
    for (double v : vals) var += (v - m) * (v - m);
    return var / vals.size();
}

double DecisionTreeRegressor::weightedMSE(const std::vector<double>& y,
                                          const std::vector<size_t>& li,
                                          const std::vector<size_t>& ri) {
    std::vector<double> yl, yr;
    for (auto i : li) yl.push_back(y[i]);
    for (auto i : ri) yr.push_back(y[i]);
    double mse_l = variance(yl), mse_r = variance(yr);
    double n = y.size(), nl = yl.size(), nr = yr.size();
    return (nl/n)*mse_l + (nr/n)*mse_r;
}