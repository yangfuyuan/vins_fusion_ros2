cmake_minimum_required(VERSION 3.5)
project(vins_fusion_ros2)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wextra -Wpedantic)

set(CMAKE_BUILD_TYPE Release)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(vins)
add_executable(vins_fusion_ros2_node src/vins_estimator.cpp src/vins_estimator_node.cpp src/visualization.cpp)
ament_target_dependencies(vins_fusion_ros2_node rclcpp rcpputils message_filters std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge image_transport)
target_link_libraries(vins_fusion_ros2_node vins_lib) 

# Install launch files
install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Install nodes
install(
  TARGETS vins_fusion_ros2_node
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
