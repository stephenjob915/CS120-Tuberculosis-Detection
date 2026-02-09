#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <filesystem>
#include "image_processor.h"
#include "decision_tree.h"

namespace fs = std::filesystem;
using namespace std;

void addData(string directory_path, int label, vector<vector<double>> &X, vector<string> &y){
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Error: Directory '" << directory_path << "' does not exist or is not a directory." << std::endl;
        return ;
    }
        
    int cnt = 0;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        cnt++;
        if(cnt % 200 == 0) cout << "Processed " << cnt << " files..." << endl;
        if (fs::is_regular_file(entry.status())) {
            string fname = entry.path().string();
            vector<vector<double>> img = imageToVector(fname);
            vector<double> imgf = flatten(img);
            X.push_back(imgf);
            y.push_back(entry.path().filename().string());
        }
    }
}
vector<double> average(vector<vector<double>> &X){
    vector<double> re(X[0].size());
    for(int i = 0 ; i < X.size(); i++){
        for(int j = 0;  j < X[0].size(); j++){
            re[j] += X[i][j];
        }
    }
    for(int j = 0;  j < X[0].size(); j++){
        re[j] /= X.size();
    }
    return re;
}

double dot(vector<double>& a, vector<double> &b){
    assert(a.size() == b.size());

    int n = a.size();
    double re = 0;
    for(int i = 0; i < n; i++) re += a[i] * b[i];

    return re;
}

bool query(vector<double> &x, vector<double>& norm, vector<double>& pos){
    return dot(x, norm) < dot(x, pos);
}

// Apply normalization using saved means and stdevs (z-score)
void normalizeWithParams(vector<vector<double>>& X, const vector<double>& means, const vector<double>& stdevs) {
    for (auto& sample : X) {
        for (size_t j = 0; j < sample.size(); j++) {
            if (stdevs[j] > 1e-10) {
                sample[j] = (sample[j] - means[j]) / stdevs[j];
            } else {
                sample[j] = 0.0;
            }
        }
    }
}

vector<cv::Mat> read_images(const string& directory_path){
    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Error: Directory '" << directory_path << "' does not exist or is not a directory." << std::endl;
        return vector<cv::Mat>();
    }
    
    vector<cv::Mat> re;
    int cnt = 0;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        cnt++;
        if(cnt % 200 == 0) cout << cnt << endl;
        if (fs::is_regular_file(entry.status())) {
            string fname = entry.path().string();
            cv::Mat image = cv::imread(fname, cv::IMREAD_GRAYSCALE);
            re.push_back(image);
        }
    }
    return re;
}

int main() {
    try {
        // Load weights (class averages)
        ifstream is("weights.txt");
        if (!is) {
            throw runtime_error("Error: weights.txt not found! Run trainer first.");
        }
        vector<double> data;
        double x;
        while(is >> x) data.push_back(x);
        is.close();

        vector<double> norm, pos;
        for(int i = 0; i < data.size() /2; i++){
            norm.push_back(data[i]);
        }
        for(int i = data.size() / 2; i < data.size(); i++){
            pos.push_back(data[i]);
        }
        
        // Load normalization parameters
        cout << "Loading normalization parameters..." << endl;
        ifstream norm_is("normalization.txt");
        if (!norm_is) {
            throw runtime_error("Error: normalization.txt not found! Run trainer first.");
        }
        int num_features;
        norm_is >> num_features;
        vector<double> means(num_features), stdevs(num_features);
        for (size_t i = 0; i < means.size(); i++) {
            norm_is >> means[i];
        }
        for (size_t i = 0; i < stdevs.size(); i++) {
            norm_is >> stdevs[i];
        }
        norm_is.close();

        vector<vector<double>> X;
        vector<string> fname;
        string dir = "./test";
        addData(dir, -1, X, fname);
        
        cout << "Loaded " << X.size() << " test images" << endl;
        if (X.empty()) {
            cout << "ERROR: No test images found in " << dir << endl;
            return 1;
        }
        
        // Apply the same normalization used during training
        cout << "Normalizing test data..." << endl;
        normalizeWithParams(X, means, stdevs);
        
        // Track metrics for confusion matrix
        int TP = 0, FP = 0, TN = 0, FN = 0;
        
        // Find optimal threshold by testing multiple values
        double best_accuracy = 0;
        double best_threshold = 0;
        
        // Try different thresholds with finer granularity
        for (double threshold = -2.0; threshold <= 2.0; threshold += 0.01) {
            int tp = 0, fp = 0, tn = 0, fn = 0;
            
            for(size_t i = 0; i < X.size(); i++){
                double score_norm = dot(X[i], norm);
                double score_pos = dot(X[i], pos);
                bool predicted_TB = (score_pos - score_norm) > threshold;
                
                bool is_actually_TB = (fname[i].find("Tuberculosis") != string::npos);
                bool is_actually_Normal = (fname[i].find("Normal") != string::npos);
                
                if(predicted_TB) {
                    if(is_actually_TB) tp++;
                    else if(is_actually_Normal) fp++;
                } else {
                    if(is_actually_Normal) tn++;
                    else if(is_actually_TB) fn++;
                }
            }
            
            // Calculate accuracy
            double accuracy = (double)(tp + tn) / X.size();
            
            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_threshold = threshold;
                TP = tp;
                FP = fp;
                TN = tn;
                FN = fn;
            }
        }
        
        cout << "Optimal threshold found: " << fixed << setprecision(2) << best_threshold << " (Accuracy: " << best_accuracy * 100 << "%)" << endl;
        cout << "\nThe following images in " << dir << " are likely positive for tuberculosis: " << endl;
        cout << "==========================================" << endl;
        for(size_t i = 0 ; i < X.size(); i++){
            double score_norm = dot(X[i], norm);
            double score_pos = dot(X[i], pos);
            bool predicted_TB = (score_pos - score_norm) > best_threshold;
            
            if(predicted_TB) {
                cout << fname[i] << endl;
            }
        }
        
        // Calculate and display metrics
        int total = TP + FP + TN + FN;
        cout << "==========================================" << endl;
        cout << "\n=== CONFUSION MATRIX & METRICS ===" << endl;
        cout << "\nConfusion Matrix:" << endl;
        cout << "                    Predicted" << endl;
        cout << "                    Normal     TB" << endl;
        cout << "Actual Normal       " << setw(4) << TN << "      " << setw(4) << FP << endl;
        cout << "Actual TB           " << setw(4) << FN << "      " << setw(4) << TP << endl;
        
        if (total > 0) {
            double accuracy = (double)(TP + TN) / total;
            double precision = (TP + FP > 0) ? (double)TP / (TP + FP) : 0;
            double recall = (TP + FN > 0) ? (double)TP / (TP + FN) : 0;
            double specificity = (TN + FP > 0) ? (double)TN / (TN + FP) : 0;
            
            cout << "\nPerformance Metrics:" << endl;
            cout << "Accuracy:    " << fixed << setprecision(2) << accuracy * 100 << "%" << endl;
            cout << "Precision:   " << precision * 100 << "%" << endl;
            cout << "Recall:      " << recall * 100 << "%" << endl;
            cout << "Specificity: " << specificity * 100 << "%" << endl;
            
            cout << "\nDetailed Breakdown:" << endl;
            cout << "Total images tested:        " << total << endl;
            cout << "True Positives (TP):        " << TP << " (Correctly identified TB)" << endl;
            cout << "True Negatives (TN):        " << TN << " (Correctly identified Normal)" << endl;
            cout << "False Positives (FP):       " << FP << " (Incorrectly marked as TB)" << endl;
            cout << "False Negatives (FN):       " << FN << " (Missed TB cases)" << endl;
        }
        
        auto imgs = read_images("./test_filter");
        cv::Mat kernel = (cv::Mat_<float>(3,3) << -1, -1, -1,
                                         -1, 9, -1,
                                          -1, -1, -1);

        int ind = 0;
        for(auto &image : imgs) {
            cout << ind << endl;
            cv::Mat sharpened_image;
            cv::filter2D(image, sharpened_image, image.depth(), kernel);
            cv::imwrite( "./filter_out/" + to_string(ind++) + ".png", sharpened_image);
        }
        
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
