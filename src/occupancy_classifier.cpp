// occupancy_classifier.cpp
#include "occupancy_classifier.hpp"

OccupancyClassifier::OccupancyClassifier() {}

void OccupancyClassifier::setReference(const cv::Mat& emptyLot) {
    reference = preprocessImage(emptyLot);
}

cv::Mat OccupancyClassifier::preprocessImage(const cv::Mat& input) {
    cv::Mat processed;
    if(input.channels() == 1) {
        processed = input.clone();
    } else {
        cv::cvtColor(input, processed, cv::COLOR_BGR2GRAY);
    }
    cv::GaussianBlur(processed, processed, cv::Size(BLUR_SIZE, BLUR_SIZE), 0);
    return processed;
}

cv::Mat OccupancyClassifier::extractROI(const cv::Mat& frame, const ParkingSpace::SpaceInfo& space) {
    // Get rotated rectangle points
    cv::Point2f vertices[4];
    space.rect.points(vertices);
    
    // Create mask
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    std::vector<cv::Point> contour;
    for(int i = 0; i < 4; i++) {
        contour.push_back(cv::Point(vertices[i].x, vertices[i].y));
    }
    std::vector<std::vector<cv::Point>> contours = {contour};
    cv::fillPoly(mask, contours, cv::Scalar(255));
    
    // Extract ROI
    cv::Mat roi;
    frame.copyTo(roi, mask);
    
    // Get min ROI
    cv::Rect boundRect = space.rect.boundingRect();
    return roi(boundRect);
}

double OccupancyClassifier::compareROI(const cv::Mat& roi1, const cv::Mat& roi2) {
    cv::Mat diff;
    cv::absdiff(roi1, roi2, diff);
    cv::threshold(diff, diff, 30, 255, cv::THRESH_BINARY);
    return cv::countNonZero(diff) / (double)(diff.rows * diff.cols);
}

bool OccupancyClassifier::isOccupied(const cv::Mat& frame, const ParkingSpace::SpaceInfo& space) {
    cv::Mat currentROI = extractROI(preprocessImage(frame), space);
    cv::Mat referenceROI = extractROI(reference, space);
    
    double diff = compareROI(currentROI, referenceROI);
    return diff > OCCUPANCY_THRESHOLD;
}

void OccupancyClassifier::processFrame(const cv::Mat& frame, std::vector<ParkingSpace::SpaceInfo>& spaces) {
    cv::Mat processed = preprocessImage(frame);
    
    for(auto& space : spaces) {
        space.occupied = isOccupied(processed, space);
    }
}