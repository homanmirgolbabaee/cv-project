// occupancy_classifier.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include "parking_space.hpp"

class OccupancyClassifier {
public:
    OccupancyClassifier();
    
    // Initialize with empty lot reference
    void setReference(const cv::Mat& emptyLot);
    
    // Check if a space is occupied
    bool isOccupied(const cv::Mat& frame, const ParkingSpace::SpaceInfo& space);
    
    // Process all spaces in frame
    void processFrame(const cv::Mat& frame, std::vector<ParkingSpace::SpaceInfo>& spaces);

private:
    cv::Mat reference;
    cv::Mat preprocessImage(const cv::Mat& input);
    cv::Mat extractROI(const cv::Mat& frame, const ParkingSpace::SpaceInfo& space);
    double compareROI(const cv::Mat& roi1, const cv::Mat& roi2);
    
    // Parameters
    const double OCCUPANCY_THRESHOLD = 0.3;
    const int BLUR_SIZE = 5;
};