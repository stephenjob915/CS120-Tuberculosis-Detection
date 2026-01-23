#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdexcept>

// Function to load image, resize to 64x64, and convert to vector<vector<double>>
std::vector<std::vector<double>> imageToVector(const std::string& filename);
std::vector<double> extractFeatures(const std::string& filename);
std::vector<double> flatten(const std::vector<std::vector<double>> &image);


#endif // IMAGE_PROCESSOR_H
