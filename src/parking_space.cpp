// parking_space.cpp
#include "parking_space.hpp"
#include <pugixml.hpp>

ParkingSpace::ParkingSpace(const std::string& path) : xmlPath(path) {}

std::vector<ParkingSpace::SpaceInfo> ParkingSpace::loadSpacesFromXML() {
    std::vector<SpaceInfo> spaces;
    pugi::xml_document doc;
    
    if (!doc.load_file(xmlPath.c_str())) {
        throw std::runtime_error("Failed to load XML file");
    }
    
    // Root node should be 'parking'
    pugi::xml_node parking = doc.child("parking");
    
    // Iterate through all space nodes
    for (pugi::xml_node space = parking.child("space"); space; space = space.next_sibling("space")) {
        SpaceInfo info;
        
        // Parse basic attributes
        info.id = space.attribute("id").as_int();
        info.occupied = space.attribute("occupied").as_bool();
        
        // Parse rotated rectangle
        info.rect = parseRotatedRect(space.child("rotatedRect"));
        
        // Parse contour points
        info.contour = parseContour(space.child("contour"));
        
        spaces.push_back(info);
    }
    
    return spaces;
}

cv::RotatedRect ParkingSpace::parseRotatedRect(const pugi::xml_node& node) {
    auto center = node.child("center");
    auto size = node.child("size");
    float angle = node.child("angle").attribute("d").as_float();
    
    return cv::RotatedRect(
        cv::Point2f(center.attribute("x").as_float(), center.attribute("y").as_float()),
        cv::Size2f(size.attribute("w").as_float(), size.attribute("h").as_float()),
        angle
    );
}

std::vector<cv::Point> ParkingSpace::parseContour(const pugi::xml_node& node) {
    std::vector<cv::Point> contour;
    
    for (pugi::xml_node point = node.child("point"); point; point = point.next_sibling("point")) {
        contour.push_back(cv::Point(
            point.attribute("x").as_int(),
            point.attribute("y").as_int()
        ));
    }
    
    return contour;
}