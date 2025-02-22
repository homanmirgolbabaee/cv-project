// car_segmenter.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include "parking_space.hpp"

class CarSegmenter {
public:
    struct CarDetection {
        cv::Mat mask;
        bool misparked;
    };

    CarSegmenter();
    
    // Main detection function
    std::vector<CarDetection> detectCars(const cv::Mat& frame, 
                                       const std::vector<ParkingSpace::SpaceInfo>& spaces);

private:
    cv::Mat preprocessFrame(const cv::Mat& frame);
    cv::Mat detectVehicles(const cv::Mat& frame);
    bool isMisparked(const cv::Mat& carMask, const std::vector<ParkingSpace::SpaceInfo>& spaces);
    
    // Parameters
    const int BLUR_SIZE = 5;
    const double CAR_AREA_MIN = 1000;  // Minimum area to consider as a car
};