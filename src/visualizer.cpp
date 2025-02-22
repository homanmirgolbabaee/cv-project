// visualizer.cpp
#include "visualizer.hpp"

// Define static color constants
const cv::Scalar Visualizer::Colors::EMPTY_SPACE = cv::Scalar(255, 0, 0);      // Blue
const cv::Scalar Visualizer::Colors::OCCUPIED_SPACE = cv::Scalar(0, 0, 255);   // Red
const cv::Scalar Visualizer::Colors::CAR_CORRECT = cv::Scalar(0, 255, 0);      // Green
const cv::Scalar Visualizer::Colors::CAR_MISPARKED = cv::Scalar(0, 255, 255);  // Yellow

Visualizer::Visualizer(const cv::Size& size) : frameSize(size) {
    // Initialize 2D map size (adjust as needed)
    mapSize = cv::Size(400, 300);
}

void Visualizer::drawSpaces(cv::Mat& frame, const std::vector<ParkingSpace::SpaceInfo>& spaces) {
    for(const auto& space : spaces) {
        // Get rotated rectangle points
        cv::Point2f vertices[4];
        space.rect.points(vertices);
        
        // Choose color based on occupancy
        cv::Scalar color = space.occupied ? Colors::OCCUPIED_SPACE : Colors::EMPTY_SPACE;
        
        // Draw the rotated rectangle
        for(int i = 0; i < 4; i++) {
            cv::line(frame, vertices[i], vertices[(i+1)%4], color, 2);
        }
        
        // Draw space ID
        cv::putText(frame, std::to_string(space.id), 
                   cv::Point(vertices[0].x, vertices[0].y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}

void Visualizer::drawCarSegmentation(cv::Mat& frame, const cv::Mat& carMask, bool misparked) {
    cv::Mat overlay;
    frame.copyTo(overlay);
    
    // Color based on parking status
    cv::Scalar color = misparked ? Colors::CAR_MISPARKED : Colors::CAR_CORRECT;
    
    // Apply color to segmented areas
    overlay.setTo(color, carMask);
    
    // Blend with original frame
    cv::addWeighted(overlay, 0.3, frame, 0.7, 0, frame);
}

void Visualizer::initializeHomography(const std::vector<ParkingSpace::SpaceInfo>& spaces) {
    // Source points from actual parking lot
    std::vector<cv::Point2f> srcPoints;
    
    // Destination points for 2D map
    std::vector<cv::Point2f> dstPoints;
    
    // Use corners of parking spaces to compute homography
    for(const auto& space : spaces) {
        cv::Point2f vertices[4];
        space.rect.points(vertices);
        
        // Add corner points
        srcPoints.push_back(vertices[0]);
        srcPoints.push_back(vertices[2]);
        
        // Map to destination points in top-view
        float x = space.rect.center.x / frameSize.width * mapSize.width;
        float y = space.rect.center.y / frameSize.height * mapSize.height;
        
        dstPoints.push_back(cv::Point2f(x - 10, y - 10));
        dstPoints.push_back(cv::Point2f(x + 10, y + 10));
    }
    
    homographyMatrix = cv::findHomography(srcPoints, dstPoints);
}

cv::Point2f Visualizer::transformPoint(const cv::Point2f& point) {
    std::vector<cv::Point2f> pts = {point};
    std::vector<cv::Point2f> transformed;
    cv::perspectiveTransform(pts, transformed, homographyMatrix);
    return transformed[0];
}

void Visualizer::drawSpace2D(cv::Mat& map, const ParkingSpace::SpaceInfo& space) {
    cv::Point2f center = transformPoint(space.rect.center);
    
    // Draw rectangle representing parking space
    cv::Scalar color = space.occupied ? Colors::OCCUPIED_SPACE : Colors::EMPTY_SPACE;
    cv::rectangle(map, 
                 cv::Point(center.x - 10, center.y - 10),
                 cv::Point(center.x + 10, center.y + 10),
                 color, -1);
    
    // Add space ID
    cv::putText(map, std::to_string(space.id),
               cv::Point(center.x - 5, center.y + 5),
               cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1);
}

cv::Mat Visualizer::create2DMap(const std::vector<ParkingSpace::SpaceInfo>& spaces) {
    // Initialize homography if not done
    if(homographyMatrix.empty()) {
        initializeHomography(spaces);
    }
    
    // Create blank map
    cv::Mat map = cv::Mat::zeros(mapSize, CV_8UC3);
    
    // Draw each space
    for(const auto& space : spaces) {
        drawSpace2D(map, space);
    }
    
    return map;
}