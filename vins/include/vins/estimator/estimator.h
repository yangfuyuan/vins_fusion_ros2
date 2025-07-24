/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <vins/common/sensor_data_type.h>
#include <vins/estimator/feature_manager.h>
#include <vins/estimator/parameters.h>
#include <vins/factor/imu_factor.h>
#include <vins/factor/marginalization_factor.h>
#include <vins/factor/pose_local_parameterization.h>
#include <vins/factor/projectionOneFrameTwoCamFactor.h>
#include <vins/factor/projectionTwoFrameOneCamFactor.h>
#include <vins/factor/projectionTwoFrameTwoCamFactor.h>
#include <vins/featureTracker/feature_tracker.h>
#include <vins/initial/initial_alignment.h>
#include <vins/initial/initial_ex_rotation.h>
#include <vins/initial/initial_sfm.h>
#include <vins/initial/solve_5pts.h>
#include <vins/logger/logger.h>
#include <vins/utility/tic_toc.h>
#include <vins/utility/utility.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <condition_variable>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <queue>
#include <thread>

using namespace std;

class Estimator {
 public:
  Estimator();
  ~Estimator();
  //
  void setParameter();
  void changeSensorType(int use_imu, int use_stereo);

  // interface
  void inputIMU(const IMUData &imu);
  void inputFeature(Timestamp timestamp, const FeatureFrame &featureFrame);
  void inputImage(const ImageData &image);
  //
  void processMeasurements();
  //
  void resetState();
  void setFirstPose(const Eigen::Vector3d &position,
                    const Eigen::Matrix3d &rotation);

  void getPoseInWorldFrame(Eigen::Matrix4d &T);
  void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);

  bool getIntegratedImuOdom(OdomData &data);
  bool getVisualInertialOdom(OdomData &data);
  bool getKeyPoses(std::vector<Eigen::Vector3d> &poses);
  bool getTrackImage(ImageData &image);

 private:
  // 内部处理函数
  void processIMU(const IMUData &data, double deltaTime);
  void processImage(const FeatureFrame &features, Timestamp timestamp);
  void fastPredictIMU(const IMUData &data);

  // internal
  bool initialStructure();
  bool visualInitialAlign();
  bool computeRelativePose(Matrix3d &relative_R, Vector3d &relative_T,
                           int &referenceFrame);
  void slideWindow();
  void slideWindowNew();
  void slideWindowOld();
  void optimization();
  void vector2double();
  void double2vector();
  bool failureDetection();
  bool getIMUInterval(double t0, double t1, vector<IMUData> &data);
  void predictPtsInNextFrame();
  void outliersRejection(set<int> &removeIndex);
  double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici,
                           Vector3d &tici, Matrix3d &Rj, Vector3d &Pj,
                           Matrix3d &ricj, Vector3d &ticj, double depth,
                           Vector3d &uvi, Vector3d &uvj);
  void updateLatestStates();
  bool IMUAvailable(double t);
  void initFirstIMUPose(const vector<IMUData> &data);

 private:
  void printStatistics(Timestamp timestamp);
  template <typename Container>
  void clearBuffer(Container &container) {
    container.clear();
  }
  template <typename T>
  void clearBuffer(std::queue<T> &queue) {
    std::queue<T> empty;
    std::swap(queue, empty);
  }

  std::thread processThread;
  std::mutex processingMutex;
  std::mutex featureBufferMutex;
  std::condition_variable featureCondition;
  std::mutex propagateMutex;
  std::mutex imu_mutex;
  std::condition_variable imuCondition;
  //
  std::queue<IMUData> imuBuffer;
  queue<TimestampedFeatureFrame> featureBuffer;

  Timestamp previousTimestamp = -1;
  Timestamp currentTimestamp = 0;
  Timestamp initialTimestamp = 0;
  double timeDelay = 0.0;

  bool openExEstimation;
  SolverState solver_flag;
  MarginalizationType marginalization_flag;
  Vector3d gravity;

  Matrix3d cameraRotation[2];
  Vector3d cameraTranslation[2];

  Vector3d positions[(WINDOW_SIZE + 1)];
  Vector3d velocities[(WINDOW_SIZE + 1)];
  Matrix3d rotations[(WINDOW_SIZE + 1)];
  Vector3d accelerometerBiases[(WINDOW_SIZE + 1)];
  Vector3d gyroscopeBiases[(WINDOW_SIZE + 1)];
  Timestamp Headers[(WINDOW_SIZE + 1)];

  Matrix3d backRotation, lastRotation, lastRotation0;
  Vector3d backPosition, lastPosition, lastPosition0;

  IntegrationBase::Ptr pre_integrations[(WINDOW_SIZE + 1)];
  IMUData previousImuData;
  std::vector<IMUData> deltaTimeImuBuffer[(WINDOW_SIZE + 1)];

  // 计数变量
  int inputImageCount = 0;
  int frameCount = 0;
  int backCount = 0;
  int frontCount = 0;

  FeatureManager featureManager;
  FeatureTracker featureTracker;
  MotionEstimator m_estimator;
  InitialEXRotation initial_ex_rotation;

  bool isFirstIMUReceived;
  bool failure_occur;

  vector<Vector3d> point_cloud;
  vector<Vector3d> margin_cloud;

  //
  SafeClass<PoseSequenceData> safe_key_poses;
  PoseSequenceData key_poses;
  ImageData track_image;
  SafeClass<ImageData> safe_track_image;
  SafeClass<OdomData> safe_imu_pre_odom;
  OdomData imu_odom;
  SafeClass<OdomData> safe_vio_odom;
  OdomData vo_odom;

  double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
  double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
  double para_Feature[NUM_OF_F][SIZE_FEATURE];
  double para_Ex_Pose[2][SIZE_POSE];
  double para_Retrive_Pose[SIZE_POSE];
  double para_Td[1][1];
  double para_Tr[1][1];

  MarginalizationInfo::Ptr last_marginalization_info;
  vector<double *> last_marginalization_parameter_blocks;

  map<double, ImageFrame> all_image_frame;
  IntegrationBase::Ptr tmp_pre_integration;

  Eigen::Vector3d initialPosition;
  Eigen::Matrix3d initialRotation;
  Eigen::Quaterniond latest_Q;
  Eigen::Vector3d latestPosition;
  Eigen::Matrix3d latestRotation;
  Eigen::Vector3d latestVelocity;
  Eigen::Vector3d latestAccelBias;
  Eigen::Vector3d latestGyroBias;
  IMUData latestImuData;
  bool isFirstPoseInitialized = false;
  std::atomic<bool> isRunning{false};
};
