#include "decision_tree.h"
#include <limits>
#include <cmath>
using namespace std;

// ============ Node Constructor ============

Node::Node()
    : is_leaf(false), predicted_class(-1),
      feature_index(-1), threshold(0.0),
      left(nullptr), right(nullptr) {}

// ============ DecisionTree ============

DecisionTree::DecisionTree(int depth, int min_s)
    : root(nullptr), max_depth(depth), min_samples(min_s) {}

double DecisionTree::gini(const vector<int>& labels) {
    int count0 = 0, count1 = 0;
    for (int y : labels) (y == 0) ? count0++ : count1++;

    double p0 = (double)count0 / labels.size();
    double p1 = (double)count1 / labels.size();

    return 1.0 - (p0*p0 + p1*p1);
}

int DecisionTree::most_common(const vector<int>& labels) {
    int count0 = 0, count1 = 0;
    for (int y : labels) (y == 0) ? count0++ : count1++;
    return (count1 > count0) ? 1 : 0;
}

Node* DecisionTree::build(const vector<vector<double>>& X,
                          const vector<int>& y,
                          int depth)
{
    Node* node = new Node();

    // Stop conditions
    if (depth >= max_depth || y.size() <= min_samples || gini(y) == 0.0) {
        node->is_leaf = true;
        node->predicted_class = most_common(y);
        return node;
    }

    // Search for best split
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_gini = numeric_limits<double>::infinity();

    for (int f = 0; f < X[0].size(); f++) {
        for (int i = 0; i < X.size(); i++) {
            double t = X[i][f];

            vector<int> left_y, right_y;
            for (int j = 0; j < X.size(); j++) {
                if (X[j][f] < t)
                    left_y.push_back(y[j]);
                else
                    right_y.push_back(y[j]);
            }

            if (left_y.empty() || right_y.empty()) continue;

            double g = (left_y.size() * gini(left_y) +
                        right_y.size() * gini(right_y))
                        / y.size();

            if (g < best_gini) {
                best_gini = g;
                best_feature = f;
                best_threshold = t;
            }
        }
    }

    if (best_feature == -1) { // No valid split
        node->is_leaf = true;
        node->predicted_class = most_common(y);
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

    // Split data
    vector<vector<double>> left_X, right_X;
    vector<int> left_y, right_y;

    for (int i = 0; i < X.size(); i++) {
        if (X[i][best_feature] < best_threshold) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        } else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
        }
    }

    node->left = build(left_X, left_y, depth+1);
    node->right = build(right_X, right_y, depth+1);

    return node;
}

void DecisionTree::train(const vector<vector<double>>& X,
                         const vector<int>& y)
{
    root = build(X, y, 0);
}

int DecisionTree::predict_one(const vector<double>& x, Node* node) {
    if (node->is_leaf)
        return node->predicted_class;

    if (x[node->feature_index] < node->threshold)
        return predict_one(x, node->left);
    else
        return predict_one(x, node->right);
}

int DecisionTree::predict(const vector<double>& x) {
    return predict_one(x, root);
}
#include <fstream>
#include <iostream>

// Save entire tree
void DecisionTree::save(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: cannot open file for writing: " << filename << "\n";
        return;
    }
    saveNode(out, root);
}

// Recursive save
void DecisionTree::saveNode(std::ofstream& out, Node* node) {
    if (!node) {
        out << "NULL\n";
        return;
    }

    out << (node->is_leaf ? "LEAF " : "NODE ")
        << node->predicted_class << " "
        << node->feature_index << " "
        << node->threshold << "\n";

    saveNode(out, node->left);
    saveNode(out, node->right);
}



// Load entire tree
void DecisionTree::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: cannot open file for reading: " << filename << "\n";
        return;
    }
    root = loadNode(in);
}

// Recursive load
Node* DecisionTree::loadNode(std::ifstream& in) {
    std::string type;
    in >> type;

    if (type == "NULL") {
        return nullptr;
    }

    Node* node = new Node();
    in >> node->predicted_class
       >> node->feature_index
       >> node->threshold;

    node->is_leaf = (type == "LEAF");

    if (!node->is_leaf) {
        node->left = loadNode(in);
        node->right = loadNode(in);
    }

    return node;
}
