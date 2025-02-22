// performance_evaluator.cpp
#include "performance_evaluator.hpp"
#include <fstream>
#include <numeric>

PerformanceEvaluator::PerformanceEvaluator() {}

PerformanceEvaluator::Metrics PerformanceEvaluator::evaluateSpaceDetection(
    const std::vector<ParkingSpace::SpaceInfo>& detected,
    const std::vector<ParkingSpace::SpaceInfo>& groundTruth) {
    
    Metrics metrics;
    metrics.totalSpaces = groundTruth.size();
    metrics.correctDetections = 0;
    metrics.falsePositives = 0;
    metrics.falseNegatives = 0;
    
    std::vector<bool> matchedGT(groundTruth.size(), false);
    std::vector<bool> matchedDetected(detected.size(), false);
    
    // Calculate IoU for each detected-GT pair
    for(size_t i = 0; i < detected.size(); i++) {
        double maxIoU = 0.0;
        size_t bestMatch = 0;
        
        for(size_t j = 0; j < groundTruth.size(); j++) {
            if(matchedGT[j]) continue;
            
            double iou = calculateIOU(detected[i].rect, groundTruth[j].rect);
            if(iou > maxIoU) {
                maxIoU = iou;
                bestMatch = j;
            }
        }
        
        if(maxIoU >= IOU_THRESHOLD) {
            matchedDetected[i] = true;
            matchedGT[bestMatch] = true;
            metrics.correctDetections++;
        }
    }
    
    metrics.falsePositives = detected.size() - metrics.correctDetections;
    metrics.falseNegatives = groundTruth.size() - metrics.correctDetections;
    
    // Calculate mAP
    if(detected.size() > 0) {
        metrics.mAP = static_cast<double>(metrics.correctDetections) / 
                     (metrics.correctDetections + metrics.falsePositives);
    } else {
        metrics.mAP = 0.0;
    }
    
    return metrics;
}

PerformanceEvaluator::Metrics PerformanceEvaluator::evaluateSegmentation(
    const cv::Mat& segmentation,
    const cv::Mat& groundTruthMask) {
    
    Metrics metrics;
    
    // Calculate IoU for each class
    std::vector<double> classIoUs;
    for(int classId = 0; classId <= 2; classId++) {  // Background, Parked, Misparked
        cv::Mat predMask = (segmentation == classId);
        cv::Mat gtMask = (groundTruthMask == classId);
        
        double iou = calculatePixelIOU(predMask, gtMask);
        classIoUs.push_back(iou);
    }
    
    // Calculate mean IoU
    metrics.mIoU = std::accumulate(classIoUs.begin(), classIoUs.end(), 0.0) / classIoUs.size();
    
    return metrics;
}

double PerformanceEvaluator::calculateIOU(
    const cv::RotatedRect& rect1,
    const cv::RotatedRect& rect2) {
    
    // Create masks for both rectangles
    cv::Mat mask1 = cv::Mat::zeros(1000, 1000, CV_8UC1);
    cv::Mat mask2 = cv::Mat::zeros(1000, 1000, CV_8UC1);
    
    // Draw rectangles
    cv::Point2f vertices1[4], vertices2[4];
    rect1.points(vertices1);
    rect2.points(vertices2);
    
    std::vector<cv::Point> contour1(vertices1, vertices1 + 4);
    std::vector<cv::Point> contour2(vertices2, vertices2 + 4);
    
    cv::fillPoly(mask1, std::vector<std::vector<cv::Point>>{contour1}, 255);
    cv::fillPoly(mask2, std::vector<std::vector<cv::Point>>{contour2}, 255);
    
    return calculatePixelIOU(mask1, mask2);
}

double PerformanceEvaluator::calculatePixelIOU(
    const cv::Mat& mask1,
    const cv::Mat& mask2) {
    
    cv::Mat intersection, union_;
    
    cv::bitwise_and(mask1, mask2, intersection);
    cv::bitwise_or(mask1, mask2, union_);
    
    double intersectionArea = cv::countNonZero(intersection);
    double unionArea = cv::countNonZero(union_);
    
    if(unionArea == 0) return 0.0;
    return intersectionArea / unionArea;
}

void PerformanceEvaluator::generateReport(
    const std::string& outputPath,
    const std::vector<Metrics>& sequenceMetrics) {
    
    std::ofstream report(outputPath);
    if (!report.is_open()) {
        throw std::runtime_error("Failed to open report file: " + outputPath);
    }

    // Write header
    report << "Parking Lot Analysis Report\n";
    report << "==========================\n\n";

    // Overall statistics
    double avgMAP = 0.0, avgMIOU = 0.0;
    int totalCorrectDetections = 0;
    int totalFalsePositives = 0;
    int totalFalseNegatives = 0;

    // Process each frame's metrics
    for (size_t i = 0; i < sequenceMetrics.size(); i++) {
        const auto& metrics = sequenceMetrics[i];
        
        report << "Frame " << (i + 1) << ":\n";
        report << "  Space Detection:\n";
        report << "    mAP: " << metrics.mAP << "\n";
        report << "    Correct Detections: " << metrics.correctDetections << "\n";
        report << "    False Positives: " << metrics.falsePositives << "\n";
        report << "    False Negatives: " << metrics.falseNegatives << "\n";
        report << "  Segmentation:\n";
        report << "    mIoU: " << metrics.mIoU << "\n\n";

        // Accumulate statistics
        avgMAP += metrics.mAP;
        avgMIOU += metrics.mIoU;
        totalCorrectDetections += metrics.correctDetections;
        totalFalsePositives += metrics.falsePositives;
        totalFalseNegatives += metrics.falseNegatives;
    }

    // Calculate averages
    if (!sequenceMetrics.empty()) {
        avgMAP /= sequenceMetrics.size();
        avgMIOU /= sequenceMetrics.size();
    }

    // Write summary
    report << "\nSummary Statistics\n";
    report << "=================\n";
    report << "Average mAP: " << avgMAP << "\n";
    report << "Average mIoU: " << avgMIOU << "\n";
    report << "Total Correct Detections: " << totalCorrectDetections << "\n";
    report << "Total False Positives: " << totalFalsePositives << "\n";
    report << "Total False Negatives: " << totalFalseNegatives << "\n";

    report.close();
}