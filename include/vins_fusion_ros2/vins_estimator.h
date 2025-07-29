#ifndef VINS_ESTIMATOR_H
#define VINS_ESTIMATOR_H
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <vins/estimator/estimator.h>
#include <vins_fusion_ros2/visualization.h>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>

using sensor_msgs::msg::Image;
using namespace message_filters;

class VinsEstimator : public rclcpp::Node {
 public:
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<Image, Image>;

  VinsEstimator();
  ~VinsEstimator();

  void initialize();
  void initializeParamters();
  void initializeSubscribers();
  void initializerPublishers();

 private:
  void timeCallback();
  void stereoCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img0,
                      const sensor_msgs::msg::Image::ConstSharedPtr& img1);

  void publishPointCloud();
  void publishImuData();
  void publishOdometry();
  void publishImage();
  void publishKeyFrameData();

 private:
  std::shared_ptr<Estimator> estimator_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
  std::shared_ptr<Subscriber<Image>> sub_img0_filter_;
  std::shared_ptr<Subscriber<Image>> sub_img1_filter_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_img_;
  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
  rclcpp::CallbackGroup::SharedPtr image_callback_group_;
  rclcpp::CallbackGroup::SharedPtr feature_callback_group_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  //
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_track;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry,
      pub_latest_odometry, pub_keyframe_pose;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud,
      pub_margin_cloud, pub_keyframe_point;

  //
  std::string world_frame_id;
  std::string body_frame_id;
  std::string camera_frame_id;

  nav_msgs::msg::Path path;
  std::shared_ptr<VINSOptions> options;
};

#endif  // VINS_ESTIMATOR_H
