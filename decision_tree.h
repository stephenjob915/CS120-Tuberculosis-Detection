#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <string>

struct Node {
    bool is_leaf;
    int predicted_class;

    int feature_index;
    double threshold;

    Node* left;
    Node* right;

    Node();
};

class DecisionTree {
public:
    Node* root;
    int max_depth;
    int min_samples;

    DecisionTree(int depth = 5, int min_s = 2);

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<int>& y);

    int predict(const std::vector<double>& x);

    // --- NEW ---
    void save(const std::string& filename);
    void load(const std::string& filename);

private:
    double gini(const std::vector<int>& labels);
    int most_common(const std::vector<int>& labels);

    Node* build(const std::vector<std::vector<double>>& X,
                const std::vector<int>& y,
                int depth);

    int predict_one(const std::vector<double>& x, Node* node);

    // --- NEW ---
    void saveNode(std::ofstream& out, Node* node);
    Node* loadNode(std::ifstream& in);
};

#endif
