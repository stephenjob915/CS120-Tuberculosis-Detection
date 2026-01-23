cmake_minimum_required(VERSION 3.10)
project(ImageProcessor)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(trainer  
    trainer.cpp
    image_processor.cpp
    decision_tree.cpp
)

add_executable(image_processor
    main.cpp 
    image_processor.cpp
    decision_tree.cpp
)

target_link_libraries(image_processor ${OpenCV_LIBS})
target_link_libraries(trainer ${OpenCV_LIBS})
