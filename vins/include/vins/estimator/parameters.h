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

#include <vins/logger/logger.h>
#include <vins/utility/utility.h>

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;

enum class ExtrinsicEstimationMode {
  FIXED = 0,  // 0: Do not estimate extrinsic parameters, use fixed
  APPROXIMATE =
      1,  // 1: Extrinsic parameters already roughly estimated, refine only
  INITIALIZE = 2  // 2: Estimate from scratch using motion
};

struct ImuOptions {
  double accelNoiseDensity = 0.0, accelRandomWalk = 0.0;
  double gyroNoiseDensity = 0.0, gyroRandomWalk = 0.0;
  Eigen::Vector3d gravityVector{0.0, 0.0, 9.8};
  std::string imu_topic;
  int useImu = 0;
  bool hasImu() const { return useImu; }
  std::string imuTopic() const { return imu_topic; }
  Eigen::Vector3d gravity() const { return gravityVector; }
  Eigen::Vector3d& gravity() { return gravityVector; }
};

struct VINSOptions {
  ImuOptions imu;
  double init_estimated_depth = 5.0;
  double min_parllaax_num = 0.0;

  std::vector<Eigen::Matrix3d> RIC;
  std::vector<Eigen::Vector3d> TIC;

  int USE_GPU = 0;
  int USE_GPU_ACC_FLOW = 0;
  int USE_GPU_CERES = 0;
  std::string EX_CALIB_RESULT_PATH;
  std::string VINS_RESULT_PATH;
  std::string OUTPUT_FOLDER;
  //////////////////////////////////////////////////////////////////////////////
  double solver_time = 0.0;
  int max_iterations = 0;
  ExtrinsicEstimationMode extrinsic_estimation_mode =
      ExtrinsicEstimationMode::FIXED;
  int estimate_td_mode = 0;

  double time_delay = 0.0;
  int num_of_camera = 0;
  int stereo = 0;
  std::string image0_topic, image1_topic;
  std::vector<std::string> camera_names;

  int max_feature_count = 0;
  int min_feature_distance = 0;
  double ransac_reproj_threshold = 0.0;
  int show_track = 0;
  int enable_reverse_optical_flow_check = 0;
  ///////////////////////////////////////////////////////////////////////////

  // 读取参数函数
  void readParameters(const std::string& config_file) {
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      VINS_ERROR << "Wrong path to settings";
      return;
    }

    fsSettings["image0_topic"] >> this->image0_topic;
    fsSettings["image1_topic"] >> this->image1_topic;
    this->max_feature_count = (int)fsSettings["max_cnt"];
    this->min_feature_distance = (int)fsSettings["min_dist"];
    this->ransac_reproj_threshold = (double)fsSettings["F_threshold"];
    this->show_track = (int)fsSettings["show_track"];
    this->enable_reverse_optical_flow_check = (int)fsSettings["flow_back"];

    this->USE_GPU = (int)fsSettings["use_gpu"];
    this->USE_GPU_ACC_FLOW = (int)fsSettings["use_gpu_acc_flow"];
    this->USE_GPU_CERES = (int)fsSettings["use_gpu_ceres"];

    this->imu.useImu = (int)fsSettings["imu"];
    VINS_INFO << "USE_IMU: " << this->hasImu() << std::endl;
    if (this->hasImu()) {
      fsSettings["imu_topic"] >> this->imu.imu_topic;
      this->imu.accelNoiseDensity = (double)fsSettings["acc_n"];
      this->imu.accelRandomWalk = (double)fsSettings["acc_w"];
      this->imu.gyroNoiseDensity = (double)fsSettings["gyr_n"];
      this->imu.gyroRandomWalk = (double)fsSettings["gyr_w"];
      this->imu.gravity().z() = (double)fsSettings["g_norm"];
    }

    this->solver_time = (double)fsSettings["max_solver_time"];
    this->max_iterations = (int)fsSettings["max_num_iterations"];
    this->min_parllaax_num = (double)fsSettings["keyframe_parallax"];
    this->min_parllaax_num = this->min_parllaax_num / FOCAL_LENGTH;

    fsSettings["output_path"] >> this->OUTPUT_FOLDER;
    this->VINS_RESULT_PATH = this->OUTPUT_FOLDER + "/vio.csv";
    VINS_INFO << "result path " << this->VINS_RESULT_PATH;
    std::ofstream fout(this->VINS_RESULT_PATH, std::ios::out);
    fout.close();

    this->extrinsic_estimation_mode = static_cast<ExtrinsicEstimationMode>(
        (int)fsSettings["estimate_extrinsic"]);
    if (isInitializingExtrinsic()) {
      this->RIC.push_back(Eigen::Matrix3d::Identity());
      this->TIC.push_back(Eigen::Vector3d::Zero());
      this->EX_CALIB_RESULT_PATH =
          this->OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    } else {
      if (isExtrinsicEstimationApproximate()) {
        this->EX_CALIB_RESULT_PATH =
            this->OUTPUT_FOLDER + "/extrinsic_parameter.csv";
      }
      cv::Mat cv_T;
      fsSettings["body_T_cam0"] >> cv_T;
      Eigen::Matrix4d T;
      cv::cv2eigen(cv_T, T);
      this->RIC.push_back(T.block<3, 3>(0, 0));
      this->TIC.push_back(T.block<3, 1>(0, 3));
    }

    this->num_of_camera = (int)fsSettings["num_of_cam"];
    if (this->num_of_camera != 1 && this->num_of_camera != 2) {
      VINS_ERROR << "num of cam should be 1 or 2 ";
    }

    int pn = (int)config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    this->camera_names.push_back(cam0Path);

    if (this->num_of_camera == 2) {
      this->stereo = 1;
      std::string cam1Calib;
      fsSettings["cam1_calib"] >> cam1Calib;
      std::string cam1Path = configPath + "/" + cam1Calib;
      this->camera_names.push_back(cam1Path);

      cv::Mat cv_T;
      fsSettings["body_T_cam1"] >> cv_T;
      Eigen::Matrix4d T;
      cv::cv2eigen(cv_T, T);
      this->RIC.push_back(T.block<3, 3>(0, 0));
      this->TIC.push_back(T.block<3, 1>(0, 3));
    }
    VINS_INFO << "STEREO: " << this->stereo;

    this->time_delay = (double)fsSettings["td"];
    this->estimate_td_mode = (int)fsSettings["estimate_td"];
    if (!this->hasImu()) {
      this->extrinsic_estimation_mode = ExtrinsicEstimationMode::FIXED;
      this->estimate_td_mode = 0;
    }

    fsSettings.release();
  }

  std::string imuTopic() const { return this->imu.imuTopic(); }
  std::string imageTopic() const { return this->image0_topic; }
  std::string image1Topic() const { return this->image1_topic; }
  int max_num_iterations() const { return max_iterations; }
  double max_solver_time() const { return solver_time; }
  /// @brief Get the number of cameras in the current configuration
  int getNumCameras() const { return num_of_camera; }
  /// @brief Check whether the system is configured for stereo camera input
  bool isUsingStereo() const { return stereo; }
  /// @brief Check whether the current configuration uses IMU
  bool hasImu() const { return imu.hasImu(); }
  /// @brief Check if the configuration is monocular without IMU
  bool isMonoWithoutImu() const { return !isUsingStereo() && !hasImu(); }
  /// @brief Check if the configuration is monocular with IMU
  bool isMonoWithImu() const { return !isUsingStereo() && hasImu(); }
  /// @brief Check if the configuration is stereo without IMU
  bool isStereoWithoutImu() const { return isUsingStereo() && !hasImu(); }
  /// @brief Check if the configuration is stereo with IMU
  bool isStereoWithImu() const { return isUsingStereo() && hasImu(); }
  bool shouldShowTrack() const { return show_track; }
  bool isExtrinsicEstimationApproximate() const {
    return extrinsic_estimation_mode == ExtrinsicEstimationMode::APPROXIMATE;
  }
  bool isInitializingExtrinsic() const {
    return extrinsic_estimation_mode == ExtrinsicEstimationMode::INITIALIZE;
  }
  bool shouldEstimateTD() const { return estimate_td_mode; }
};

enum SIZE_PARAMETERIZATION {
  SIZE_POSE = 7,
  SIZE_SPEEDBIAS = 9,
  SIZE_FEATURE = 1
};

enum StateOrder { O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12 };

enum NoiseOrder { O_AN = 0, O_GN = 3, O_AW = 6, O_GW = 9 };
