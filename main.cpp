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
            y.push_back(fname);
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
        ifstream is("weights.txt");
        vector<double> data;
        double x;
        while(is >> x) data.push_back(x);

        vector<double> norm, pos;
        for(int i = 0; i < data.size() /2; i++){
            norm.push_back(data[i]);
        }
        for(int i = data.size() / 2; i < data.size(); i++){
            pos.push_back(data[i]);
        }

        vector<vector<double>> X;
        vector<string> fname;
        string dir = "./test";
        addData(dir, -1, X, fname);
        cout << "The following images in " << dir << " are likely positive for tuberculosis: " << endl;
        for(int i = 0 ; i < X.size(); i++){
            if(query(X[i], norm, pos)) cout << fname[i] << endl;
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
