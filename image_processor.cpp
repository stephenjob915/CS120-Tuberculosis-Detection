#include "image_processor.h"
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

void computeGLCMFeatures(const Mat& gray, double& contrast, double& energy,
                         double& homogeneity, double& entropy)
{
    // Quantize to reduce matrix size (0–255 → 0–31)
    Mat quant = gray.clone();
    quant.convertTo(quant, CV_8U, 1.0 / 8.0);  // proper scaling 0–255 → 0–31

    const int levels = 32;
    Mat glcm = Mat::zeros(levels, levels, CV_64F);

    // GLCM: offset (dx = 1, dy = 0) → horizontal relationship
    for (int i = 0; i < quant.rows; i++) {
        for (int j = 0; j < quant.cols - 1; j++) {
            int a = quant.at<uchar>(i, j);
            int b = quant.at<uchar>(i, j + 1);
            glcm.at<double>(a, b)++;
        }
    }

    // Normalize
    glcm /= sum(glcm)[0];

    contrast = energy = homogeneity = entropy = 0.0;

    for (int i = 0; i < levels; i++) {
        for (int j = 0; j < levels; j++) {
            double p = glcm.at<double>(i,j);
            if (p <= 0) continue;

            contrast += (i - j) * (i - j) * p;
            energy   += p * p;
            homogeneity += p / (1.0 + abs(i - j));
            entropy  += -p * log2(p);
        }
    }
}

vector<float> computeHOG(const Mat& gray)
{
    vector<float> descriptors;
    
    try {
        HOGDescriptor hog(
            Size(64,128), // window
            Size(16,16),  // block
            Size(8,8),    // stride
            Size(8,8),    // cell
            9             // bins
        );

        Mat resized;
        resize(gray, resized, Size(64,128));

        hog.compute(resized, descriptors);
    } catch (const std::exception& e) {
        cout << "HOG computation failed: " << e.what() << endl;
        // Return empty vector on failure
        descriptors.clear();
    }
    
    return descriptors;
}

vector<double> extractFeatures(const string& filename) {
    vector<double> re;
    
    try {
        Mat img = imread(filename, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cout << "Warning: Could not load image: " << filename << endl;
            return re;  // Return empty vector
        }

        // ---- GLCM features ----
        double contrast, energy, homogeneity, entropy;
        computeGLCMFeatures(img, contrast, energy, homogeneity, entropy);

        // ---- HOG ----
        vector<float> hogFeatures = computeHOG(img);

        for(float i : hogFeatures) re.push_back(i);
        re.push_back(contrast);
        re.push_back(energy);
        re.push_back(homogeneity);
        re.push_back(entropy);
    } catch (const std::exception& e) {
        cout << "Error in extractFeatures for " << filename << ": " << e.what() << endl;
    }
    
    return re;
}

vector<double> flatten(const std::vector<std::vector<double>> &image){
    vector<double> re;
    for(int i = 0; i < image.size(); i++){
        for(int j = 0; j < image[0].size(); j++){
            re.push_back(image[i][j]);
        }
    }

    return re;
}

std::vector<std::vector<double>> imageToVector(const std::string& filename) {
    // Step 1: Load the image
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    
    // Check if image was loaded successfully
    if (image.empty()) {
        throw std::runtime_error("Error: Could not open or find the image: " + filename);
    }
    
    // Step 1.5: Enhance contrast with histogram equalization
    cv::Mat enhanced;
    cv::equalizeHist(image, enhanced);
    
    // Step 2: Resize to 48x48 (good balance of detail and speed)
    cv::Mat resized;
    cv::resize(enhanced, resized, cv::Size(48, 48), 0, 0, cv::INTER_CUBIC);
    
    // Step 3: Normalize with z-score
    cv::Mat normalized;
    resized.convertTo(normalized, CV_64F, 1.0 / 255.0);
    cv::Scalar mean, stddev;
    cv::meanStdDev(normalized, mean, stddev);
    if (stddev[0] > 1e-10) {
        normalized = (normalized - mean[0]) / stddev[0];
    }
    
    std::vector<std::vector<double>> result;
    result.reserve(normalized.rows);
    
    for (int i = 0; i < normalized.rows; i++) {
        std::vector<double> row;
        row.reserve(normalized.cols);
        
        for (int j = 0; j < normalized.cols; j++) {
            row.push_back(normalized.at<double>(i, j));
        }
        
        result.push_back(row);
    }
    
    return result;
}
