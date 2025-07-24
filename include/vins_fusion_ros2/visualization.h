#ifndef VISUALIZATION_H
#define VISUALIZATION_H
#include <geometry_msgs/msg/point_stamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <vins/estimator/estimator.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker.hpp>

template <typename T, typename NodeT>
T readParam(NodeT node, const std::string& name, T default_value = T()) {
  T value;
  node->template declare_parameter<T>(name, default_value);
  if (!node->template get_parameter<T>(name, value)) {
    throw std::runtime_error("Parameter '" + name + "' not found.");
  }
  return value;
}

double fromMsg(const builtin_interfaces::msg::Time& msg);
Eigen::Vector3d fromMsg(const geometry_msgs::msg::Vector3& msg);
Eigen::Quaterniond fromMsg(const geometry_msgs::msg::Quaternion& msg);
IMUData fromMsg(const sensor_msgs::msg::Imu& msg);

cv::Mat fromMsg(const sensor_msgs::msg::Image& img_msg);
FeatureFrame fromMsg(const sensor_msgs::msg::PointCloud& msg);

nav_msgs::msg::Odometry toMsg(const OdomData& msg);
builtin_interfaces::msg::Time toMsg(double msg);
sensor_msgs::msg::Image toMsg(const cv::Mat& msg);
sensor_msgs::msg::PointCloud toMsg(const PointCloudData& msg);

#endif
