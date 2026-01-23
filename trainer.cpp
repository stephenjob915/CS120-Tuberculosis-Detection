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
        if(cnt % 200 == 0) cout << cnt << endl;
            // Check if the entry is a regular file
        if (fs::is_regular_file(entry.status())) {
            // cout << entry.path().filename().string() << endl;
            string fname = entry.path().string();
            // vector<double> g = extractFeatures(fname);
            // X.push_back(g);
            // y.push_back(label);
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
using namespace std;
int main() {
    try {
        vector<vector<double>> X;
        vector<int> y;
        
        addData("./TB_Chest_Radiography_Database/Normal", 0, X, y);

        vector<double> normal_avg = average(X);

        X.clear();
        y.clear();
        addData("./TB_Chest_Radiography_Database/Tuberculosis", 1, X, y);
        
        vector<double> positive_avg = average(X);
        
        std::ofstream os("weights.txt");
        for(int i = 0 ; i < positive_avg.size(); i++){
            os << normal_avg[i] << " \n"[i == positive_avg.size() - 1];
        }
        
        for(int i = 0 ; i < positive_avg.size(); i++){
            os << positive_avg[i] << " \n"[i == positive_avg.size() - 1];
        }

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
