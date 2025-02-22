// main.cpp
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <algorithm>  // for std::count_if
#include <memory>    // for std::unique_ptr
#include "parking_space.hpp"
#include "space_detector.hpp"
#include "occupancy_classifier.hpp"
#include "car_segmenter.hpp"  // Make sure this exists
#include "visualizer.hpp"

namespace fs = std::filesystem;

class ParkingAnalyzer {
public:
    ParkingAnalyzer(const std::string& sequence0Path, const std::string& sequence1Path) {
        // Get executable path and calculate relative data path
        std::string exePath = fs::current_path().string();
        std::string baseDir = exePath + "/../data/";  // Go up one level and into data
        
        std::string seq0Path = baseDir + "sequence0";
        std::string seq1Path = baseDir + "sequence1";
        
        // Initialize components
        initializeFromEmptyLot(seq0Path);
        processSequence(seq1Path);
    }

private:
    OccupancyClassifier occupancyClassifier;
    CarSegmenter carSegmenter;
    std::unique_ptr<Visualizer> visualizer;
    std::vector<ParkingSpace::SpaceInfo> parkingSpaces;

    void initializeFromEmptyLot(const std::string& path) {
        // Load empty lot image and XML
        std::string xmlPath = path + "/bounding_boxes/2013-02-24_10_05_04.xml";
        std::string imagePath = path + "/frames/2013-02-24_10_05_04.jpg";

        cv::Mat emptyLot = cv::imread(imagePath);
        if (emptyLot.empty()) {
            throw std::runtime_error("Failed to load empty lot image: " + imagePath);
        }

        // Initialize parking spaces from XML
        ParkingSpace spaceLoader(xmlPath);
        parkingSpaces = spaceLoader.loadSpacesFromXML();

        // Initialize components
        occupancyClassifier.setReference(emptyLot);
        visualizer = std::make_unique<Visualizer>(emptyLot.size());

        // Show empty lot visualization
        cv::Mat visualization = emptyLot.clone();
        visualizer->drawSpaces(visualization, parkingSpaces);
        
        cv::Mat map2D = visualizer->create2DMap(parkingSpaces);

        // Display initialization results
        cv::namedWindow("Empty Lot", cv::WINDOW_NORMAL);
        cv::namedWindow("2D Map", cv::WINDOW_NORMAL);
        cv::imshow("Empty Lot", visualization);
        cv::imshow("2D Map", map2D);
        cv::waitKey(100);  // Brief display
    }

    void processSequence(const std::string& sequencePath) {
        // Create output windows with trackbars
        cv::namedWindow("Control Panel", cv::WINDOW_NORMAL);
        cv::namedWindow("Current Frame", cv::WINDOW_NORMAL);
        cv::namedWindow("2D Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Car Segmentation", cv::WINDOW_NORMAL);
        cv::namedWindow("Statistics", cv::WINDOW_NORMAL);

        // Add control trackbar
        int delay = 500; // 500ms default delay
        cv::createTrackbar("Delay (ms)", "Control Panel", &delay, 2000);
        
        bool paused = false;
        cv::createTrackbar("Pause (0/1)", "Control Panel", (int*)&paused, 1);

        std::cout << "\nControls:" << std::endl;
        std::cout << "- Press 'q' to quit" << std::endl;
        std::cout << "- Press 'p' to pause/unpause" << std::endl;
        std::cout << "- Press 's' to step when paused" << std::endl;
        std::cout << "- Use trackbar to adjust speed" << std::endl;

        for (const auto& entry : fs::directory_iterator(sequencePath + "/frames")) {
            if(paused) {
                char key = cv::waitKey(0);
                if(key == 'q') break;
                if(key == 'p') paused = false;
                if(key != 's') continue;
            }

            // Load and process frame
            cv::Mat frame = cv::imread(entry.path().string());
            if (frame.empty()) {
                std::cerr << "Failed to load frame: " << entry.path() << std::endl;
                continue;
            }

            // Load XML
            std::string xmlPath = sequencePath + "/bounding_boxes/" + 
                                entry.path().filename().stem().string() + ".xml";
            
            std::cout << "Processing frame: " << entry.path().filename() << std::endl;

            try {
                ParkingSpace frameSpaces(xmlPath);
                auto currentSpaces = frameSpaces.loadSpacesFromXML();
                processFrame(frame, currentSpaces);
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing frame: " << e.what() << std::endl;
            }

            char key = cv::waitKey(delay);
            if(key == 'q') break;
            if(key == 'p') paused = !paused;
        }
    }

    void processFrame(cv::Mat& frame, std::vector<ParkingSpace::SpaceInfo>& spaces) {
        // Detect occupancy
        occupancyClassifier.processFrame(frame, spaces);

        // Detect cars and their parking status
        auto carDetections = carSegmenter.detectCars(frame, spaces);

        // Create visualizations
        cv::Mat visualization = frame.clone();
        visualizer->drawSpaces(visualization, spaces);

        // Draw car segmentation
        cv::Mat segmentation = frame.clone();
        for (const auto& detection : carDetections) {
            visualizer->drawCarSegmentation(segmentation, detection.mask, detection.misparked);
        }

        // Create 2D map
        cv::Mat map2D = visualizer->create2DMap(spaces);

        // Show results
        cv::imshow("Current Frame", visualization);
        cv::imshow("Car Segmentation", segmentation);
        cv::imshow("2D Map", map2D);

        // Calculate and display statistics
        displayStatistics(spaces, carDetections);
    }

    void displayStatistics(const std::vector<ParkingSpace::SpaceInfo>& spaces,
                         const std::vector<CarSegmenter::CarDetection>& detections) {
        // Count statistics
        int totalSpaces = spaces.size();
        int occupiedSpaces = std::count_if(spaces.begin(), spaces.end(),
                                         [](const auto& space) { return space.occupied; });
        int misparkedCars = std::count_if(detections.begin(), detections.end(),
                                        [](const auto& det) { return det.misparked; });

        // Create statistics image
        cv::Mat stats(200, 400, CV_8UC3, cv::Scalar(255, 255, 255));
        
        // Display text
        cv::putText(stats, "Parking Lot Statistics:", cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        
        cv::putText(stats, "Total Spaces: " + std::to_string(totalSpaces),
                   cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        
        cv::putText(stats, "Occupied Spaces: " + std::to_string(occupiedSpaces),
                   cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
        
        cv::putText(stats, "Available Spaces: " + std::to_string(totalSpaces - occupiedSpaces),
                   cv::Point(10, 130), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        
        cv::putText(stats, "Misparked Cars: " + std::to_string(misparkedCars),
                   cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);

        cv::imshow("Statistics", stats);
    }
};

int main(int argc, char** argv) {
    try {
        ParkingAnalyzer analyzer("sequence0", "sequence1");
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}