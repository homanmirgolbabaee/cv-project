// car_segmenter.cpp
#include "car_segmenter.hpp"

CarSegmenter::CarSegmenter() {}

cv::Mat CarSegmenter::preprocessFrame(const cv::Mat& frame) {
    cv::Mat processed;
    
    // Convert to grayscale
    cv::cvtColor(frame, processed, cv::COLOR_BGR2GRAY);
    
    // Apply Gaussian blur
    cv::GaussianBlur(processed, processed, cv::Size(BLUR_SIZE, BLUR_SIZE), 0);
    
    // Apply adaptive threshold
    cv::adaptiveThreshold(processed, processed, 255,
                         cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY_INV, 11, 2);
    
    return processed;
}

cv::Mat CarSegmenter::detectVehicles(const cv::Mat& frame) {
    cv::Mat processed = preprocessFrame(frame);
    
    // Morphological operations to remove noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(processed, processed, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(processed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Create mask for cars
    cv::Mat carMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    
    for(const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if(area > CAR_AREA_MIN) {
            cv::drawContours(carMask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), -1);
        }
    }
    
    return carMask;
}

bool CarSegmenter::isMisparked(const cv::Mat& carMask, 
                              const std::vector<ParkingSpace::SpaceInfo>& spaces) {
    // Check if car mask overlaps with any parking space
    for(const auto& space : spaces) {
        cv::Mat spaceMask = cv::Mat::zeros(carMask.size(), CV_8UC1);
        std::vector<cv::Point> contour = space.contour;
        std::vector<std::vector<cv::Point>> contours = {contour};
        cv::fillPoly(spaceMask, contours, cv::Scalar(255));
        
        // Calculate overlap
        cv::Mat intersection;
        cv::bitwise_and(carMask, spaceMask, intersection);
        
        if(cv::countNonZero(intersection) > 0) {
            return false;  // Car is in a parking space
        }
    }
    
    return true;  // Car is not in any parking space
}

std::vector<CarSegmenter::CarDetection> CarSegmenter::detectCars(
    const cv::Mat& frame,
    const std::vector<ParkingSpace::SpaceInfo>& spaces) {

    lastSegmentation = detectVehicles(frame);  // Store for metrics
    
    cv::Mat carMask = detectVehicles(frame);
    std::vector<CarDetection> detections;
    
    // Find connected components in car mask
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(carMask, labels, stats, centroids);
    
    for(int i = 1; i < numLabels; i++) {  // Skip background (label 0)
        cv::Mat currentMask = (labels == i);
        
        CarDetection detection;
        detection.mask = currentMask;
        detection.misparked = isMisparked(currentMask, spaces);
        
        detections.push_back(detection);
    }
    
    return detections;
}