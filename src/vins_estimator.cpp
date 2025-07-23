#include <cv_bridge/cv_bridge.h>
#include <vins_fusion_ros2/vins_estimator.h>

VinsEstimator::VinsEstimator() : rclcpp::Node("vins_estimator") {
  estimator_ = std::make_shared<Estimator>();
  initialize();
}

VinsEstimator::~VinsEstimator() {}

void VinsEstimator::initialize() {
  initializeParamters();
  initializeSubscribers();
  initializerPublishers();
}
void VinsEstimator::initializeParamters() {
  auto config_file = readParam<std::string>(this, "config_file");
  world_frame_id = readParam<std::string>(this, "world_frame_id", "world");
  body_frame_id = readParam<std::string>(this, "body_frame_id", "body");

  readParameters(config_file);
  estimator_->setParameter();
}
void VinsEstimator::initializeSubscribers() {
  imu_callback_group_ =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  image_callback_group_ =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  feature_callback_group_ =
      this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions sub_opt_imu;
  sub_opt_imu.callback_group = imu_callback_group_;

  rclcpp::SubscriptionOptions sub_opt_image;
  sub_opt_image.callback_group = image_callback_group_;

  rclcpp::SubscriptionOptions sub_opt_feature;
  sub_opt_feature.callback_group = feature_callback_group_;

  if (USE_IMU) {
    auto imu = this->create_subscription<sensor_msgs::msg::Imu>(
        IMU_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)),
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
          auto imu_msg = fromMsg(*msg);
          estimator_->inputIMU(imu_msg);
        },
        sub_opt_imu);
    subs_.push_back(imu);
  }

  if (STEREO) {
    sub_img0_filter_ = std::make_shared<Subscriber<Image>>(
        this, IMAGE0_TOPIC, rmw_qos_profile_sensor_data, sub_opt_image);

    sub_img1_filter_ = std::make_shared<Subscriber<Image>>(
        this, IMAGE1_TOPIC, rmw_qos_profile_sensor_data, sub_opt_image);

    sync_img_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), *sub_img0_filter_, *sub_img1_filter_);

    sync_img_->registerCallback(std::bind(&VinsEstimator::stereoCallback, this,
                                          std::placeholders::_1,
                                          std::placeholders::_2));
  } else {
    auto sub_img0 = this->create_subscription<sensor_msgs::msg::Image>(
        IMAGE0_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)),
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
          ImageData image;
          image.image0 = fromMsg(*msg);
          image.timestamp = fromMsg(msg->header.stamp);
          estimator_->inputImage(image);
        },
        sub_opt_image);
    subs_.push_back(sub_img0);
  }

  auto sub_feature = this->create_subscription<sensor_msgs::msg::PointCloud>(
      "/feature_tracker/feature", rclcpp::QoS(rclcpp::KeepLast(100)),
      [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
        estimator_->inputFeature(fromMsg(msg->header.stamp), fromMsg(*msg));
      },
      sub_opt_feature);
  subs_.push_back(sub_feature);

  publish_timer_ =
      this->create_wall_timer(std::chrono::milliseconds(20),
                              std::bind(&VinsEstimator::timeCallback, this));
}
void VinsEstimator::initializerPublishers() {
  pub_latest_odometry =
      this->create_publisher<nav_msgs::msg::Odometry>("imu_propagate", 1);
  pub_path = this->create_publisher<nav_msgs::msg::Path>("path", 1);
  pub_odometry = this->create_publisher<nav_msgs::msg::Odometry>("odometry", 1);
  pub_image_track =
      this->create_publisher<sensor_msgs::msg::Image>("image_track", 1);
}

void VinsEstimator::stereoCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& img0,
    const sensor_msgs::msg::Image::ConstSharedPtr& img1) {
  ImageData image;
  image.image0 = fromMsg(*img0);
  image.image1 = fromMsg(*img1);
  image.timestamp = fromMsg(img0->header.stamp);
  estimator_->inputImage(image);
}

void VinsEstimator::timeCallback() {
  OdomData imu_odom;
  if (estimator_->getIntegratedImuOdom(imu_odom)) {
    nav_msgs::msg::Odometry odometry = toMsg(imu_odom);
    odometry.header.frame_id = world_frame_id;
    odometry.child_frame_id = body_frame_id;
    pub_latest_odometry->publish(odometry);
  }

  OdomData vio_odom;
  if (estimator_->getVisualInertialOdom(vio_odom)) {
    nav_msgs::msg::Odometry odometry = toMsg(vio_odom);
    odometry.header.frame_id = world_frame_id;
    odometry.child_frame_id = body_frame_id;

    pub_odometry->publish(odometry);
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = odometry.header;
    pose_stamped.header.frame_id = world_frame_id;
    pose_stamped.pose = odometry.pose.pose;
    path.header = odometry.header;
    path.header.frame_id = world_frame_id;
    path.poses.push_back(pose_stamped);
    pub_path->publish(path);
  }

  ImageData image;
  if (estimator_->getTrackImage(image)) {
    auto img = toMsg(image.image0);
    img.header.frame_id = world_frame_id;
    img.header.stamp = toMsg(image.timestamp);
    pub_image_track->publish(img);
  }
}
