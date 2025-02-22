// visualizer.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include "parking_space.hpp"

class Visualizer {
public:
    Visualizer(const cv::Size& frameSize);
    
    // Color schemes for visualization
    struct Colors {
        static const cv::Scalar EMPTY_SPACE;      // Blue
        static const cv::Scalar OCCUPIED_SPACE;    // Red
        static const cv::Scalar CAR_CORRECT;       // Green
        static const cv::Scalar CAR_MISPARKED;     // Yellow
    };

    // Main visualization functions
    void drawSpaces(cv::Mat& frame, const std::vector<ParkingSpace::SpaceInfo>& spaces);
    void drawCarSegmentation(cv::Mat& frame, const cv::Mat& carMask, bool misparked);
    
    // 2D top-view map generation
    cv::Mat create2DMap(const std::vector<ParkingSpace::SpaceInfo>& spaces);

private:
    cv::Size frameSize;
    cv::Size mapSize;
    cv::Mat homographyMatrix;  // For perspective transform
    
    // Helper functions
    void initializeHomography(const std::vector<ParkingSpace::SpaceInfo>& spaces);
    cv::Point2f transformPoint(const cv::Point2f& point);
    void drawSpace2D(cv::Mat& map, const ParkingSpace::SpaceInfo& space);
};