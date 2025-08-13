cmake_minimum_required(VERSION 3.8)
project(video_reader)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(video_reader_node 
                src/video_reader.cpp # main file
                src/ImageAcquisition.cpp
              )

ament_target_dependencies(video_reader_node rclcpp std_msgs Eigen3 OpenCV sensor_msgs cv_bridge)

target_include_directories(video_reader_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(video_reader_node Eigen3::Eigen)

# Install executables and libraries
install(TARGETS
  video_reader_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(
  DIRECTORY include/
  DESTINATION include/
)

file(GLOB SOURCES "src/*.cpp")

# Install launch files and src files
install(
  DIRECTORY src launch
  DESTINATION share/${PROJECT_NAME}
)


message(=============================================================)
message("-- ROS Found. ROS Support is turned ON.")
message(=============================================================)




ament_package()
