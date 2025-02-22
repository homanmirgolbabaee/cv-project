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
#include "performance_evaluator.hpp"

namespace fs = std::filesystem;

class ParkingAnalyzer {
public:


    // Constructor for empty lot initialization
    ParkingAnalyzer(const std::string& sequence0Path) {
        std::string exePath = fs::current_path().string();
        std::string baseDir = exePath + "/../data/";
        std::string seq0Path = baseDir + sequence0Path;
        
        initializeFromEmptyLot(seq0Path);
    }


    // Constructor for processing sequence with cars
    ParkingAnalyzer(const std::string& sequence0Path, const std::string& sequencePath) {
        // Get executable path and calculate relative data path
        std::string exePath = fs::current_path().string();
        std::string baseDir = exePath + "/../data/";  // Go up one level and into data
        
        std::string seq0Path = baseDir + sequence0Path;
        std::string seq1Path = baseDir + sequencePath;
        
        // Initialize components
        initializeFromEmptyLot(seq0Path);
        processSequence(seq1Path);
    }

private:

    PerformanceEvaluator performanceEvaluator;
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
        std::vector<PerformanceEvaluator::Metrics> sequenceMetrics;
        
        for(const auto& entry : fs::directory_iterator(sequencePath + "/frames")) {
            // Load current frame
            cv::Mat frame = cv::imread(entry.path().string());
            if(frame.empty()) continue;
            
            // Load ground truth
            std::string gtXmlPath = sequencePath + "/bounding_boxes/" + 
                                entry.path().stem().string() + ".xml";
            std::string gtMaskPath = sequencePath + "/masks/" + 
                                    entry.path().stem().string() + ".png";
            
            ParkingSpace groundTruth(gtXmlPath);
            auto gtSpaces = groundTruth.loadSpacesFromXML();
            cv::Mat gtMask = cv::imread(gtMaskPath, cv::IMREAD_GRAYSCALE);
            
            // Fix: Pass parkingSpaces to processFrame
            processFrame(frame, parkingSpaces);
            
            // Evaluate performance
            auto spaceMetrics = performanceEvaluator.evaluateSpaceDetection(parkingSpaces, gtSpaces);
            auto segMetrics = performanceEvaluator.evaluateSegmentation(
                carSegmenter.getLastSegmentation(), gtMask);
            
            // Store metrics
            sequenceMetrics.push_back(spaceMetrics);
            
            // Display results
            displayResults(frame, spaceMetrics, segMetrics);
            
            if(cv::waitKey(1) == 'q') break;
        }
        
        // Generate final report
        performanceEvaluator.generateReport("evaluation_report.txt", sequenceMetrics);
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



    void displayResults(const cv::Mat& frame, 
                      const PerformanceEvaluator::Metrics& spaceMetrics,
                      const PerformanceEvaluator::Metrics& segMetrics) {
        // Display frame with results
        cv::Mat display = frame.clone();
        
        // Draw metrics on frame
        std::string spaceText = "Space mAP: " + std::to_string(spaceMetrics.mAP);
        std::string segText = "Seg mIoU: " + std::to_string(segMetrics.mIoU);
        
        cv::putText(display, spaceText, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::putText(display, segText, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
                   
        cv::imshow("Results", display);
    }





};

// main.cpp updated
int main(int argc, char** argv) {
    try {
        // Create windows first
        cv::namedWindow("Empty Lot", cv::WINDOW_NORMAL);
        cv::namedWindow("Current Frame", cv::WINDOW_NORMAL);
        cv::namedWindow("Car Segmentation", cv::WINDOW_NORMAL);
        cv::namedWindow("2D Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Statistics", cv::WINDOW_NORMAL);
        cv::namedWindow("Controls", cv::WINDOW_NORMAL);
        cv::namedWindow("Results", cv::WINDOW_NORMAL);

        // Add control trackbar
        int delay = 500; // 500ms default delay
        cv::createTrackbar("Delay (ms)", "Controls", &delay, 2000);
        
        bool paused = false;
        cv::createTrackbar("Pause (0/1)", "Controls", (int*)&paused, 1);

        std::cout << "Starting Parking Lot Analysis..." << std::endl;
        std::cout << "\nControls:" << std::endl;
        std::cout << "- Press 'q' to quit current sequence" << std::endl;
        std::cout << "- Press 'p' to pause/unpause" << std::endl;
        std::cout << "- Press 's' to step when paused" << std::endl;
        std::cout << "- Use trackbar to adjust speed" << std::endl;

        // Initialize with sequence0
        std::cout << "\nInitializing from empty lot (sequence0)..." << std::endl;
        ParkingAnalyzer analyzer("sequence0");
        
        // Process each sequence
        for(int seq = 1; seq <= 5; seq++) {
            std::string seqPath = "sequence" + std::to_string(seq);
            std::cout << "\nProcessing " << seqPath << "..." << std::endl;
            
            // Create new analyzer for each sequence
            ParkingAnalyzer seqAnalyzer("sequence0", seqPath);
            
            while(true) {
                char key = cv::waitKey(delay);
                if(key == 'q') break;
                if(key == 'p') paused = !paused;
                if(paused && key != 's') continue;
            }
            
            std::cout << "Sequence " << seq << " complete." << std::endl;
            std::cout << "Press any key to continue to next sequence..." << std::endl;
            cv::waitKey(0);
        }

        std::cout << "\nProcessing complete. Results saved in evaluation_report.txt" << std::endl;
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
