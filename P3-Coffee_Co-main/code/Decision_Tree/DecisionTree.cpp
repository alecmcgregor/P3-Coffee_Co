#include "DecisionTree.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>
#include <fstream>

struct DecisionTree::Node {
    bool is_leaf = false;
    int feature_index = -1;
    double threshold = 0.0;
    double prediction = 0.0;
    std::unique_ptr<Node> left, right;
};

DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : max_depth_(max_depth), min_samples_split_(min_samples_split) {}

DecisionTree::~DecisionTree() = default;

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y) {
    if (X.empty()) return;
    int n_features = static_cast<int>(X[0].size());
    feature_importances_.assign(n_features, 0.0);
    root_ = buildTree(X, y, 0);

    // Normalize feature importances
    double total_importance = std::accumulate(
        feature_importances_.begin(), feature_importances_.end(), 0.0);
    if (total_importance > 0.0) {
        for (double& v : feature_importances_) v /= total_importance;
    }
}

std::vector<double> DecisionTree::feature_importances() const {
    return feature_importances_;
}

double DecisionTree::predict(const std::vector<double>& x) const {
    return predictSample(root_.get(), x);
}

std::vector<double> DecisionTree::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> preds;
    preds.reserve(X.size());
    for (const auto& row : X)
        preds.push_back(predict(row));
    return preds;
}

std::unique_ptr<DecisionTree::Node>
DecisionTree::buildTree(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y,
                        int depth) {
    auto node = std::make_unique<Node>();
    double curr_var = variance(y);

    // Stopping conditions
    if (depth >= max_depth_ || y.size() < static_cast<size_t>(min_samples_split_) || curr_var == 0.0) {
        node->is_leaf = true;
        node->prediction = mean(y);
        return node;
    }

    int best_feature = -1;
    double best_threshold = 0.0, best_mse = std::numeric_limits<double>::infinity();
    std::vector<size_t> best_left, best_right;

    int num_features = static_cast<int>(X[0].size());

    for (int f = 0; f < num_features; ++f) {
        std::vector<double> unique_vals;
        for (const auto& row : X) unique_vals.push_back(row[f]);
        std::sort(unique_vals.begin(), unique_vals.end());
        unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());

        int num_vals = static_cast<int>(unique_vals.size());
        if (num_vals < 2) continue;

        const int max_thresholds = 10;
        for (int j = 1; j <= max_thresholds && j < num_vals; ++j) {
            int lo = static_cast<int>(std::floor(j * (num_vals - 1) / static_cast<double>(max_thresholds + 1)));
            int hi = lo + 1;
            if (hi >= num_vals) continue;
            double threshold = 0.5 * (unique_vals[lo] + unique_vals[hi]);

            std::vector<size_t> left_idx, right_idx;
            for (size_t i = 0; i < X.size(); ++i) {
                (X[i][f] <= threshold ? left_idx : right_idx).push_back(i);
            }

            if (left_idx.empty() || right_idx.empty()) continue;

            double mse = weightedMSE(y, left_idx, right_idx);
            if (mse < best_mse) {
                best_mse = mse;
                best_feature = f;
                best_threshold = threshold;
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

    feature_importances_[best_feature] += (curr_var - best_mse);

    std::vector<std::vector<double>> X_left, X_right;
    std::vector<double> y_left, y_right;

    for (auto i : best_left) {
        X_left.push_back(X[i]);
        y_left.push_back(y[i]);
    }
    for (auto i : best_right) {
        X_right.push_back(X[i]);
        y_right.push_back(y[i]);
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = buildTree(X_left, y_left, depth + 1);
    node->right = buildTree(X_right, y_right, depth + 1);

    return node;
}

double DecisionTree::predictSample(const Node* node, const std::vector<double>& x) const {
    while (!node->is_leaf) {
        node = (x[node->feature_index] <= node->threshold)
            ? node->left.get() : node->right.get();
    }
    return node->prediction;
}

double DecisionTree::mean(const std::vector<double>& vals) {
    return std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
}

double DecisionTree::variance(const std::vector<double>& vals) {
    double m = mean(vals);
    double var = 0.0;
    for (double v : vals) var += (v - m) * (v - m);
    return var / vals.size();
}

double DecisionTree::weightedMSE(const std::vector<double>& y,
                                 const std::vector<size_t>& li,
                                 const std::vector<size_t>& ri) {
    std::vector<double> yl, yr;
    yl.reserve(li.size());
    yr.reserve(ri.size());

    for (auto i : li) yl.push_back(y[i]);
    for (auto i : ri) yr.push_back(y[i]);

    double mse_left = variance(yl);
    double mse_right = variance(yr);
    double total = static_cast<double>(y.size());

    return (yl.size() / total) * mse_left + (yr.size() / total) * mse_right;
}