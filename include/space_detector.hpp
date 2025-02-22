// space_detector.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include "parking_space.hpp"

class SpaceDetector {
public:
    SpaceDetector();
    std::vector<ParkingSpace::SpaceInfo> detectSpaces(const cv::Mat& emptyLot);
    void drawSpaces(cv::Mat& frame, const std::vector<ParkingSpace::SpaceInfo>& spaces);
private:
    cv::Mat preprocessImage(const cv::Mat& input);
};