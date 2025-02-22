// space_detector.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include "parking_space.hpp"

class SpaceDetector {
public:
    SpaceDetector();
    
    // Main detection function
    std::vector<ParkingSpace::SpaceInfo> detectSpaces(const cv::Mat& emptyLot);
    
    // Visualization helper
    void drawSpaces(cv::Mat& frame, const std::vector<ParkingSpace::SpaceInfo>& spaces);
    
    // Parameters struct for tuning
    struct Parameters {
        int blurSize = 5;
        double cannyLowThresh = 50;
        double cannyHighThresh = 150;
        double houghRho = 1;
        double houghTheta = CV_PI/180;
        int houghThreshold = 50;
        double minLineLength = 50;
        double maxLineGap = 10;
        double parallelAngleThresh = 10;  // degrees
        double perpAngleThresh = 20;      // degrees
        double minSpaceArea = 1000;
        double maxSpaceArea = 20000;
        double minAspectRatio = 1.5;
        double maxAspectRatio = 4.0;
    };

private:
    Parameters params;

    // Processing pipeline
    cv::Mat preprocessImage(const cv::Mat& input);
    cv::Mat enhanceLines(const cv::Mat& input);
    std::vector<cv::Vec4i> detectLines(const cv::Mat& edges);
    std::vector<std::vector<cv::Vec4i>> groupLines(const std::vector<cv::Vec4i>& lines);
    std::vector<std::vector<cv::Point>> findIntersections(
        const std::vector<std::vector<cv::Vec4i>>& lineGroups);
    std::vector<ParkingSpace::SpaceInfo> createSpaceCandidates(
        const std::vector<std::vector<cv::Point>>& intersections);
    std::vector<ParkingSpace::SpaceInfo> filterSpaces(
        const std::vector<ParkingSpace::SpaceInfo>& candidates);

    // Helper functions
    double getLineAngle(const cv::Vec4i& line);
    bool areLinesParallel(const cv::Vec4i& line1, const cv::Vec4i& line2);
    bool areLinesPerpendicular(const cv::Vec4i& line1, const cv::Vec4i& line2);
    cv::Point2f findIntersectionPoint(const cv::Vec4i& line1, const cv::Vec4i& line2);
    bool isValidParkingSpace(const cv::RotatedRect& rect);
};