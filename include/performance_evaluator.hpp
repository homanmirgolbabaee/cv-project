// performance_evaluator.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include "parking_space.hpp"

class PerformanceEvaluator {
public:
    struct Metrics {
        double mAP;       // mean Average Precision
        double mIoU;      // mean Intersection over Union
        int totalSpaces;
        int correctDetections;
        int falsePositives;
        int falseNegatives;
    };
    
    PerformanceEvaluator();
    
    // Evaluate parking space detection
    Metrics evaluateSpaceDetection(
        const std::vector<ParkingSpace::SpaceInfo>& detected,
        const std::vector<ParkingSpace::SpaceInfo>& groundTruth);
    
    // Evaluate car segmentation
    Metrics evaluateSegmentation(
        const cv::Mat& segmentation,
        const cv::Mat& groundTruthMask);
    
    // Generate report
    void generateReport(const std::string& outputPath,
                       const std::vector<Metrics>& sequenceMetrics);

private:
    double calculateIOU(const cv::RotatedRect& rect1,
                       const cv::RotatedRect& rect2);
    double calculatePixelIOU(const cv::Mat& mask1,
                           const cv::Mat& mask2);
    
    const double IOU_THRESHOLD = 0.5;
};