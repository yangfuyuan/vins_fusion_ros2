
#ifndef SENSOR_DATA_TYPE_H
#define SENSOR_DATA_TYPE_H
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <atomic>
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

  bool get(T &value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_value) return false;
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

#endif  //
