
#ifndef SENSOR_DATA_TYPE_H
#define SENSOR_DATA_TYPE_H
#include <vins/common/data_type.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <mutex>
#include <opencv2/opencv.hpp>

template <typename T>
class SafeClass {
 public:
  SafeClass() = default;
  ~SafeClass() = default;

  void set(const T &value) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_ = value;
    has_value = true;
  }

  bool get(T &value, bool force = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_value && !force) return false;
    value = data_;
    has_value = false;
    return true;
  }

  bool check() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return has_value;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    has_value = false;
  }

 private:
  mutable std::mutex mutex_;
  bool has_value = false;
  T data_;
};

struct SensorDataBase {
  double timestamp = 0u;
};

struct OdomData : SensorDataBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct PoseData : SensorDataBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct IMUData : SensorDataBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct ImageData : SensorDataBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  cv::Mat image0 = cv::Mat();
  cv::Mat image1 = cv::Mat();
  ImageData(const ImageData &other) {
    timestamp = other.timestamp;
    image0 = other.image0.clone();
    image1 = other.image1.clone();
  }

  ImageData &operator=(const ImageData &other) {
    if (this != &other) {
      timestamp = other.timestamp;
      image0 = other.image0.clone();
      image1 = other.image1.clone();
    }
    return *this;
  }

  ImageData(ImageData &&other) noexcept {
    timestamp = other.timestamp;
    image0 = std::move(other.image0);
    image1 = std::move(other.image1);
  }

  ImageData &operator=(ImageData &&other) noexcept {
    if (this != &other) {
      timestamp = other.timestamp;
      image0 = std::move(other.image0);
      image1 = std::move(other.image1);
    }
    return *this;
  }

  ImageData() = default;
  ~ImageData() = default;
};

struct PoseSequenceData : SensorDataBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<Eigen::Vector3d> poses;
};

struct ChannelFloat {
  std::string name;
  std::vector<float> values;
};

struct PointCloudData : SensorDataBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<Eigen::Vector3d> points;
  std::vector<ChannelFloat> channels;
};

struct StateData : SensorDataBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();

  void clear() {
    position = Eigen::Vector3d::Zero();
    velocity = Eigen::Vector3d::Zero();
    rotation = Eigen::Matrix3d::Identity();
    accel_bias = Eigen::Vector3d::Zero();
    gyro_bias = Eigen::Vector3d::Zero();
  }

  void swap(StateData &other) {
    position.swap(other.position);
    velocity.swap(other.velocity);
    rotation.swap(other.rotation);
    accel_bias.swap(other.accel_bias);
    gyro_bias.swap(other.gyro_bias);
  }

  void toPoseArray(double *pose) const {
    pose[0] = position.x();
    pose[1] = position.y();
    pose[2] = position.z();
    Eigen::Quaterniond q(rotation);
    pose[3] = q.x();
    pose[4] = q.y();
    pose[5] = q.z();
    pose[6] = q.w();
  }

  void fromPoseArray(const double *pose) {
    position = Eigen::Vector3d(pose[0], pose[1], pose[2]);
    Eigen::Quaterniond q(pose[6], pose[3], pose[4], pose[5]);  // w, x, y, z
    rotation = q.normalized().toRotationMatrix();
  }

  void toSpeedBiasArray(double *sb) const {
    sb[0] = velocity.x();
    sb[1] = velocity.y();
    sb[2] = velocity.z();
    sb[3] = accel_bias.x();
    sb[4] = accel_bias.y();
    sb[5] = accel_bias.z();
    sb[6] = gyro_bias.x();
    sb[7] = gyro_bias.y();
    sb[8] = gyro_bias.z();
  }

  void fromSpeedBiasArray(const double *sb) {
    velocity = Eigen::Vector3d(sb[0], sb[1], sb[2]);
    accel_bias = Eigen::Vector3d(sb[3], sb[4], sb[5]);
    gyro_bias = Eigen::Vector3d(sb[6], sb[7], sb[8]);
  }
};

#endif  //
