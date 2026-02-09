#include <iostream>
#include <iomanip>
#include <fstream>
#include "image_processor.h"
#include "decision_tree.h"
#include <filesystem>
namespace fs = std::filesystem;

using namespace std;
void addData(string directory_path, int label, vector<vector<double>> &X, vector<int> &y){
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
            y.push_back(label);
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

// Normalize features using z-score (mean=0, std=1)
void normalizeFeatures(vector<vector<double>>& X, vector<double>& means, vector<double>& stdevs) {
    if (X.empty()) return;
    
    // Calculate means
    means.resize(X[0].size(), 0);
    for (const auto& sample : X) {
        for (size_t j = 0; j < sample.size(); j++) {
            means[j] += sample[j];
        }
    }
    for (size_t j = 0; j < means.size(); j++) {
        means[j] /= X.size();
    }
    
    // Calculate standard deviations
    stdevs.resize(X[0].size(), 0);
    for (const auto& sample : X) {
        for (size_t j = 0; j < sample.size(); j++) {
            double diff = sample[j] - means[j];
            stdevs[j] += diff * diff;
        }
    }
    for (size_t j = 0; j < stdevs.size(); j++) {
        stdevs[j] = sqrt(stdevs[j] / X.size());
    }
    
    // Normalize all samples
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
using namespace std;
int main() {
    try {
        vector<vector<double>> X;
        vector<int> y;
        vector<double> means, stdevs;
        
        cout << "Loading Normal images..." << endl;
        addData("./TB_Chest_Radiography_Database/Normal", 0, X, y);
        int normal_count = X.size();

        cout << "Loading Tuberculosis images..." << endl;
        addData("./TB_Chest_Radiography_Database/Tuberculosis", 1, X, y);
        int tb_count = X.size() - normal_count;
        
        cout << "Loaded " << normal_count << " Normal images and " << tb_count << " TB images" << endl;
        cout << "Normalizing features (z-score)..." << endl;
        normalizeFeatures(X, means, stdevs);
        
        // Separate data after normalization
        vector<vector<double>> normal_X(X.begin(), X.begin() + normal_count);
        vector<vector<double>> tb_X(X.begin() + normal_count, X.end());
        
        vector<double> normal_avg = average(normal_X);
        vector<double> positive_avg = average(tb_X);
        
        cout << "Saving normalized weights..." << endl;
        std::ofstream os("weights.txt");
        for(size_t i = 0 ; i < positive_avg.size(); i++){
            os << normal_avg[i] << " \n"[i == positive_avg.size() - 1];
        }
        
        for(size_t i = 0 ; i < positive_avg.size(); i++){
            os << positive_avg[i] << " \n"[i == positive_avg.size() - 1];
        }
        os.close();
        
        // Save normalization parameters (means and stdevs for z-score)
        cout << "Saving normalization parameters..." << endl;
        std::ofstream norm_os("normalization.txt");
        norm_os << means.size() << "\n";
        for (double val : means) {
            norm_os << val << " ";
        }
        norm_os << "\n";
        for (double val : stdevs) {
            norm_os << val << " ";
        }
        norm_os << "\n";
        norm_os.close();
        
        cout << "Training complete! Weights and normalization parameters saved." << endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
