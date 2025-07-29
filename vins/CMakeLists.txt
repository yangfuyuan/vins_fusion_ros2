cmake_minimum_required(VERSION 3.5)
project(vins)
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wextra -Wpedantic)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Glog)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
find_package(Eigen3)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIRS}

)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
file(GLOB_RECURSE HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

add_library(vins_lib ${SRC_FILES} ${HEADER_FILES})
target_link_libraries(vins_lib  ${OpenCV_LIBS} ${CERES_LIBRARIES} ${GLOG_LIBRARIES})

if(OpenMP_CXX_FOUND)
   target_link_libraries(vins_lib OpenMP::OpenMP_CXX)
endif()

target_include_directories(vins_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CERES_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CERES_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${GLOG_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)

# Install nodes
install(
  TARGETS vins_lib
  DESTINATION lib/${PROJECT_NAME}
)
