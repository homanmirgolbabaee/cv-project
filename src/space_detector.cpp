// space_detector.cpp
#include "space_detector.hpp"

SpaceDetector::SpaceDetector() {}

std::vector<ParkingSpace::SpaceInfo> SpaceDetector::detectSpaces(const cv::Mat& emptyLot) {
    // 1. Preprocess image
    cv::Mat processed = preprocessImage(emptyLot);
    
    // 2. Enhance lines
    cv::Mat enhanced = enhanceLines(processed);
    
    // 3. Detect lines
    std::vector<cv::Vec4i> lines = detectLines(enhanced);
    
    // 4. Group lines
    auto lineGroups = groupLines(lines);
    
    // 5. Find intersections
    auto intersections = findIntersections(lineGroups);
    
    // 6. Create space candidates
    auto candidates = createSpaceCandidates(intersections);
    
    // 7. Filter and validate spaces
    auto validSpaces = filterSpaces(candidates);
    
    return validSpaces;
}

cv::Mat SpaceDetector::preprocessImage(const cv::Mat& input) {
    cv::Mat processed;
    
    // Convert to grayscale
    if(input.channels() == 3) {
        cv::cvtColor(input, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = input.clone();
    }
    
    // Apply adaptive thresholding for robustness to lighting
    cv::adaptiveThreshold(processed, processed,
                         255,
                         cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY,
                         11, 2);
    
    // Remove noise while preserving edges
    cv::GaussianBlur(processed, processed, 
                     cv::Size(params.blurSize, params.blurSize), 0);
    
    return processed;
}

cv::Mat SpaceDetector::enhanceLines(const cv::Mat& input) {
    cv::Mat enhanced;
    
    // Enhance edges
    cv::Canny(input, enhanced, 
              params.cannyLowThresh, 
              params.cannyHighThresh);
    
    // Dilate to connect broken lines
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate(enhanced, enhanced, kernel);
    
    return enhanced;
}

std::vector<cv::Vec4i> SpaceDetector::detectLines(const cv::Mat& edges) {
    std::vector<cv::Vec4i> lines;
    
    cv::HoughLinesP(edges, lines,
                    params.houghRho,
                    params.houghTheta,
                    params.houghThreshold,
                    params.minLineLength,
                    params.maxLineGap);
    
    return lines;
}

std::vector<std::vector<cv::Vec4i>> SpaceDetector::groupLines(
    const std::vector<cv::Vec4i>& lines) {
    
    std::vector<std::vector<cv::Vec4i>> groups;
    std::vector<double> angles;
    
    for(const auto& line : lines) {
        double angle = getLineAngle(line);
        
        bool found = false;
        // Check existing groups
        for(size_t i = 0; i < groups.size(); i++) {
            if(areLinesParallel(line, groups[i][0])) {
                groups[i].push_back(line);
                found = true;
                break;
            }
        }
        
        if(!found) {
            groups.push_back({line});
        }
    }
    
    return groups;
}

std::vector<std::vector<cv::Point>> SpaceDetector::findIntersections(
    const std::vector<std::vector<cv::Vec4i>>& lineGroups) {
    
    std::vector<std::vector<cv::Point>> intersections;
    
    for(size_t i = 0; i < lineGroups.size(); i++) {
        for(size_t j = i + 1; j < lineGroups.size(); j++) {
            if(areLinesPerpendicular(lineGroups[i][0], lineGroups[j][0])) {
                std::vector<cv::Point> corners;
                
                for(const auto& line1 : lineGroups[i]) {
                    for(const auto& line2 : lineGroups[j]) {
                        cv::Point2f intersection = findIntersectionPoint(line1, line2);
                        if(intersection.x >= 0 && intersection.y >= 0) {
                            corners.push_back(intersection);
                        }
                    }
                }
                
                if(corners.size() >= 4) {
                    intersections.push_back(corners);
                }
            }
        }
    }
    
    return intersections;
}

// Helper functions implementation to follow...

// Continuing space_detector.cpp...

std::vector<ParkingSpace::SpaceInfo> SpaceDetector::createSpaceCandidates(
    const std::vector<std::vector<cv::Point>>& intersections) {
    
    std::vector<ParkingSpace::SpaceInfo> candidates;
    static int id = 0;
    
    for(const auto& corners : intersections) {
        // Find minimum area rectangle
        cv::RotatedRect rect = cv::minAreaRect(corners);
        
        // Create parking space candidate
        ParkingSpace::SpaceInfo space;
        space.id = ++id;
        space.rect = rect;
        space.contour = corners;
        space.occupied = false;
        
        candidates.push_back(space);
    }
    
    return candidates;
}

std::vector<ParkingSpace::SpaceInfo> SpaceDetector::filterSpaces(
    const std::vector<ParkingSpace::SpaceInfo>& candidates) {
    
    std::vector<ParkingSpace::SpaceInfo> validSpaces;
    
    for(const auto& space : candidates) {
        if(isValidParkingSpace(space.rect)) {
            validSpaces.push_back(space);
        }
    }
    
    // Sort spaces by location (top to bottom, left to right)
    std::sort(validSpaces.begin(), validSpaces.end(),
              [](const ParkingSpace::SpaceInfo& a, const ParkingSpace::SpaceInfo& b) {
                  if(std::abs(a.rect.center.y - b.rect.center.y) > 50) {
                      return a.rect.center.y < b.rect.center.y;
                  }
                  return a.rect.center.x < b.rect.center.x;
              });
    
    // Reassign IDs in sorted order
    for(size_t i = 0; i < validSpaces.size(); i++) {
        validSpaces[i].id = i + 1;
    }
    
    return validSpaces;
}

double SpaceDetector::getLineAngle(const cv::Vec4i& line) {
    double angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;
    // Normalize angle to [0, 180)
    if(angle < 0) angle += 180;
    return angle;
}

bool SpaceDetector::areLinesParallel(const cv::Vec4i& line1, const cv::Vec4i& line2) {
    double angle1 = getLineAngle(line1);
    double angle2 = getLineAngle(line2);
    double angleDiff = std::abs(angle1 - angle2);
    return angleDiff < params.parallelAngleThresh;
}

bool SpaceDetector::areLinesPerpendicular(const cv::Vec4i& line1, const cv::Vec4i& line2) {
    double angle1 = getLineAngle(line1);
    double angle2 = getLineAngle(line2);
    double angleDiff = std::abs(angle1 - angle2);
    return std::abs(angleDiff - 90) < params.perpAngleThresh;
}

cv::Point2f SpaceDetector::findIntersectionPoint(const cv::Vec4i& line1, const cv::Vec4i& line2) {
    float x1 = line1[0], y1 = line1[1];
    float x2 = line1[2], y2 = line1[3];
    float x3 = line2[0], y3 = line2[1];
    float x4 = line2[2], y4 = line2[3];
    
    float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if(std::abs(denom) < 1e-6) {
        return cv::Point2f(-1, -1);  // Lines are parallel
    }
    
    float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    
    return cv::Point2f(x1 + t * (x2 - x1), y1 + t * (y2 - y1));
}

bool SpaceDetector::isValidParkingSpace(const cv::RotatedRect& rect) {
    float width = rect.size.width;
    float height = rect.size.height;
    
    // Ensure consistent width/height ratio
    if(width > height) {
        std::swap(width, height);
    }
    
    float area = width * height;
    float ratio = height / width;
    
    // Check constraints
    return (area >= params.minSpaceArea && 
            area <= params.maxSpaceArea &&
            ratio >= params.minAspectRatio && 
            ratio <= params.maxAspectRatio);
}

void SpaceDetector::drawSpaces(cv::Mat& frame, const std::vector<ParkingSpace::SpaceInfo>& spaces) {
    for(const auto& space : spaces) {
        // Draw rotated rectangle
        cv::Point2f vertices[4];
        space.rect.points(vertices);
        
        cv::Scalar color = space.occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        
        for(int i = 0; i < 4; i++) {
            cv::line(frame, vertices[i], vertices[(i+1)%4], color, 2);
        }
        
        // Draw ID
        cv::putText(frame, std::to_string(space.id), 
                   cv::Point(space.rect.center.x - 10, space.rect.center.y + 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}