#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>

class DecisionTree {
public:
    // max_depth: maximum tree depth; min_samples_split: minimum samples to consider a split
    DecisionTree(int max_depth = 12, int min_samples_split = 20);
    ~DecisionTree();

    // Train on feature matrix X and target vector y
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);

    // Predict one sample or many
    double predict(const std::vector<double>& x) const;
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

    // Get normalized feature importances
    std::vector<double> feature_importances() const;

private:
    struct Node;
    std::unique_ptr<Node> root_;
    int max_depth_;
    int min_samples_split_;
    std::vector<double> feature_importances_;

    // Recursive tree construction
    std::unique_ptr<Node> buildTree(const std::vector<std::vector<double>>& X,
                                    const std::vector<double>& y,
                                    int depth);

    // Recursive single‐sample prediction
    double predictSample(const Node* node,
                         const std::vector<double>& x) const;

    // Helpers
    static double mean(const std::vector<double>& vals);
    static double variance(const std::vector<double>& vals);
    static double weightedMSE(const std::vector<double>& y,
                              const std::vector<size_t>& left_idx,
                              const std::vector<size_t>& right_idx);
};

#endif // DECISION_TREE_H
