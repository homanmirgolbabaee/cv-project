cmake_minimum_required(VERSION 3.10)
project(ParkingAnalyzer)

find_package(OpenCV 4.8.1 REQUIRED)
find_package(pugixml REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(parking_analyzer 
    src/main.cpp
    src/parking_space.cpp
    src/space_detector.cpp
    src/occupancy_classifier.cpp
    src/car_segmenter.cpp
    src/visualizer.cpp
    src/performance_evaluator.cpp  # Add this
)

target_link_libraries(parking_analyzer 
    ${OpenCV_LIBS}
    pugixml
    stdc++fs  # For filesystem
)

