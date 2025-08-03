#include "DecisionTree.h"
#include <algorithm>
#include <limits>
#include <cmath>

struct DecisionTree::Node {
    bool is_leaf;
    int feature_index;
    double threshold;
    double prediction;
    std::unique_ptr<Node> left, right;
    Node()
      : is_leaf(false),
        feature_index(-1),
        threshold(0.0),
        prediction(0.0)
    {}
};

DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : root_(nullptr),
      max_depth_(max_depth),
      min_samples_split_(min_samples_split)
{}

DecisionTree::~DecisionTree() = default;

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y) {
    if (X.empty()) return;
    int n_features = (int)X[0].size();
    feature_importances_.assign(n_features, 0.0);
    root_ = buildTree(X, y, 0);

    // normalize
    double total = 0.0;
    for (double v : feature_importances_) total += v;
    if (total > 0.0) {
        for (double &v : feature_importances_) v /= total;
    }
}

std::vector<double> DecisionTree::feature_importances() const {
    return feature_importances_;
}

double DecisionTree::predict(const std::vector<double>& x) const {
    return predictSample(root_.get(), x);
}

std::vector<double> DecisionTree::predict(
    const std::vector<std::vector<double>>& X) const
{
    std::vector<double> preds;
    preds.reserve(X.size());
    for (auto& row : X)
        preds.push_back(predict(row));
    return preds;
}

std::unique_ptr<DecisionTree::Node>
DecisionTree::buildTree(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y,
                        int depth)
{
    auto node = std::make_unique<Node>();
    double curr_var = variance(y);

    // stopping:
    if (depth >= max_depth_ ||
        y.size() < size_t(min_samples_split_) ||
        curr_var == 0.0)
    {
        node->is_leaf = true;
        node->prediction = mean(y);
        return node;
    }

    int best_f = -1;
    double best_t = 0.0;
    double best_mse = std::numeric_limits<double>::infinity();
    std::vector<size_t> best_l, best_r;
    int F = (int)X[0].size();

    // For each feature
    for (int f = 0; f < F; ++f) {
        // collect and unique‐sort values
        std::vector<double> vals;
        vals.reserve(X.size());
        for (auto& r : X) vals.push_back(r[f]);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        int M = (int)vals.size();
        if (M < 2) continue;

        // **Threshold sampling**: at most K splits
        const int K = 10;
        for (int j = 1; j <= K && j < M; ++j) {
            int lo = std::floor(j * (M - 1) / double(K + 1));
            int hi = lo + 1;
            double th = 0.5 * (vals[lo] + vals[hi]);

            std::vector<size_t> left_idx, right_idx;
            for (size_t i = 0; i < X.size(); ++i) {
                (X[i][f] <= th ? left_idx : right_idx).push_back(i);
            }
            if (left_idx.empty() || right_idx.empty()) continue;

            double mse = weightedMSE(y, left_idx, right_idx);
            if (mse < best_mse) {
                best_mse = mse;
                best_f   = f;
                best_t   = th;
                best_l   = left_idx;
                best_r   = right_idx;
            }
        }
    }

    if (best_f < 0) {
        node->is_leaf = true;
        node->prediction = mean(y);
        return node;
    }

    // accumulate importance = variance reduction
    feature_importances_[best_f] += (curr_var - best_mse);

    // split data
    std::vector<std::vector<double>> Xl, Xr;
    std::vector<double> yl, yr;
    Xl.reserve(best_l.size());  yl.reserve(best_l.size());
    Xr.reserve(best_r.size());  yr.reserve(best_r.size());
    for (auto i : best_l) { Xl.push_back(X[i]); yl.push_back(y[i]); }
    for (auto i : best_r) { Xr.push_back(X[i]); yr.push_back(y[i]); }

    node->feature_index = best_f;
    node->threshold     = best_t;
    node->left          = buildTree(Xl, yl, depth + 1);
    node->right         = buildTree(Xr, yr, depth + 1);
    return node;
}

double DecisionTree::predictSample(const Node* node,
                                   const std::vector<double>& x) const
{
    if (node->is_leaf) return node->prediction;
    if (x[node->feature_index] <= node->threshold)
        return predictSample(node->left.get(), x);
    else
        return predictSample(node->right.get(), x);
}

double DecisionTree::mean(const std::vector<double>& vals) {
    double s = 0;
    for (double v : vals) s += v;
    return s / vals.size();
}

double DecisionTree::variance(const std::vector<double>& vals) {
    double m = mean(vals), v = 0;
    for (double x : vals) v += (x - m)*(x - m);
    return v / vals.size();
}

double DecisionTree::weightedMSE(const std::vector<double>& y,
                                 const std::vector<size_t>& li,
                                 const std::vector<size_t>& ri)
{
    std::vector<double> yl, yr;
    yl.reserve(li.size()); yr.reserve(ri.size());
    for (auto i : li) yl.push_back(y[i]);
    for (auto i : ri) yr.push_back(y[i]);
    double mse_l = variance(yl), mse_r = variance(yr);
    double n = double(y.size()), nl = double(yl.size()), nr = double(yr.size());
    return (nl/n)*mse_l + (nr/n)*mse_r;
}
