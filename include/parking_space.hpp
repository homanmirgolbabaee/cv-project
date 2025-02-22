// parking_space.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <pugixml.hpp> // We'll need this for XML parsing

class ParkingSpace {
public:
    struct SpaceInfo {
        int id;
        cv::RotatedRect rect;
        std::vector<cv::Point> contour;
        bool occupied;
        
        // Constructor for convenience
        SpaceInfo(int _id = 0) : id(_id), occupied(false) {}
    };
    
    ParkingSpace(const std::string& xmlPath);
    
    // Main function to load spaces from XML
    std::vector<SpaceInfo> loadSpacesFromXML();
    
    // Helper functions
    static cv::RotatedRect parseRotatedRect(const pugi::xml_node& node);
    static std::vector<cv::Point> parseContour(const pugi::xml_node& node);

private:
    std::string xmlPath;
};