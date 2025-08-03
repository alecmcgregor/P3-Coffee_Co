#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>

class DecisionTreeRegressor {
public:
    DecisionTreeRegressor(int max_depth = 5, int min_samples_split = 10);
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);
    double predict(const std::vector<double>& x) const;
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

private:
    struct Node {
        bool is_leaf;
        int feature_index;
        double threshold;
        double prediction;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        Node() : is_leaf(false), feature_index(-1), threshold(0.0), prediction(0.0) {}
    };
    std::unique_ptr<Node> root_;
    int max_depth_;
    int min_samples_split_;

    std::unique_ptr<Node> buildTree(const std::vector<std::vector<double>>& X,
                                    const std::vector<double>& y,
                                    int depth);
    double predictSample(const Node* node,
                         const std::vector<double>& x) const;

    static double mean(const std::vector<double>& vals);
    static double variance(const std::vector<double>& vals);
    static double weightedMSE(const std::vector<double>& y,
                              const std::vector<size_t>& left_idx,
                              const std::vector<size_t>& right_idx);
};

#endif // DECISION_TREE_H