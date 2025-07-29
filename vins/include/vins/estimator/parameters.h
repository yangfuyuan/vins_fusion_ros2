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

struct ImuOptions {
  double ACC_N = 0.0, ACC_W = 0.0;
  double GYR_N = 0.0, GYR_W = 0.0;
  Eigen::Vector3d G{0.0, 0.0, 9.8};
  std::string IMU_TOPIC;
  int USE_IMU = 0;
  double BIAS_ACC_THRESHOLD = 0.1;
  double BIAS_GYR_THRESHOLD = 0.1;
  bool hasImu() const { return USE_IMU; }
  std::string imuTopic() const { return IMU_TOPIC; }
};

struct VINSOptions {
  // 参数
  ImuOptions imu;
  double INIT_DEPTH = 5.0;
  double MIN_PARALLAX = 0.0;

  std::vector<Eigen::Matrix3d> RIC;
  std::vector<Eigen::Vector3d> TIC;

  int USE_GPU = 0;
  int USE_GPU_ACC_FLOW = 0;
  int USE_GPU_CERES = 0;

  double SOLVER_TIME = 0.0;
  int NUM_ITERATIONS = 0;
  int ESTIMATE_EXTRINSIC = 0;
  int ESTIMATE_TD = 0;
  int ROLLING_SHUTTER = 0;

  std::string EX_CALIB_RESULT_PATH;
  std::string VINS_RESULT_PATH;
  std::string OUTPUT_FOLDER;

  int ROW = 0, COL = 0;
  double TD = 0.0;

  int NUM_OF_CAM = 0;
  int STEREO = 0;

  std::map<int, Eigen::Vector3d> pts_gt;

  std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
  std::string FISHEYE_MASK;
  std::vector<std::string> CAM_NAMES;

  int MAX_CNT = 0;
  int MIN_DIST = 0;
  double F_THRESHOLD = 0.0;
  int SHOW_TRACK = 0;
  int FLOW_BACK = 0;

  // 读取参数函数
  void readParameters(const std::string& config_file) {
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      VINS_ERROR << "Wrong path to settings";
      return;
    }

    fsSettings["image0_topic"] >> this->IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> this->IMAGE1_TOPIC;
    this->MAX_CNT = (int)fsSettings["max_cnt"];
    this->MIN_DIST = (int)fsSettings["min_dist"];
    this->F_THRESHOLD = (double)fsSettings["F_threshold"];
    this->SHOW_TRACK = (int)fsSettings["show_track"];
    this->FLOW_BACK = (int)fsSettings["flow_back"];

    this->USE_GPU = (int)fsSettings["use_gpu"];
    this->USE_GPU_ACC_FLOW = (int)fsSettings["use_gpu_acc_flow"];
    this->USE_GPU_CERES = (int)fsSettings["use_gpu_ceres"];

    this->imu.USE_IMU = (int)fsSettings["imu"];
    VINS_INFO << "USE_IMU: " << this->hasImu() << std::endl;
    if (this->hasImu()) {
      fsSettings["imu_topic"] >> this->imu.IMU_TOPIC;
      this->imu.ACC_N = (double)fsSettings["acc_n"];
      this->imu.ACC_W = (double)fsSettings["acc_w"];
      this->imu.GYR_N = (double)fsSettings["gyr_n"];
      this->imu.GYR_W = (double)fsSettings["gyr_w"];
      this->imu.G.z() = (double)fsSettings["g_norm"];
    }

    this->SOLVER_TIME = (double)fsSettings["max_solver_time"];
    this->NUM_ITERATIONS = (int)fsSettings["max_num_iterations"];
    this->MIN_PARALLAX = (double)fsSettings["keyframe_parallax"];
    const double FOCAL_LENGTH = 460.0;
    this->MIN_PARALLAX = this->MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> this->OUTPUT_FOLDER;
    this->VINS_RESULT_PATH = this->OUTPUT_FOLDER + "/vio.csv";
    VINS_INFO << "result path " << this->VINS_RESULT_PATH;
    std::ofstream fout(this->VINS_RESULT_PATH, std::ios::out);
    fout.close();

    this->ESTIMATE_EXTRINSIC = (int)fsSettings["estimate_extrinsic"];
    if (this->ESTIMATE_EXTRINSIC == 2) {
      this->RIC.push_back(Eigen::Matrix3d::Identity());
      this->TIC.push_back(Eigen::Vector3d::Zero());
      this->EX_CALIB_RESULT_PATH =
          this->OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    } else {
      if (this->ESTIMATE_EXTRINSIC == 1) {
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

    this->NUM_OF_CAM = (int)fsSettings["num_of_cam"];
    if (this->NUM_OF_CAM != 1 && this->NUM_OF_CAM != 2) {
      VINS_ERROR << "num of cam should be 1 or 2 ";
    }

    int pn = (int)config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    this->CAM_NAMES.push_back(cam0Path);

    if (this->NUM_OF_CAM == 2) {
      this->STEREO = 1;
      std::string cam1Calib;
      fsSettings["cam1_calib"] >> cam1Calib;
      std::string cam1Path = configPath + "/" + cam1Calib;
      this->CAM_NAMES.push_back(cam1Path);

      cv::Mat cv_T;
      fsSettings["body_T_cam1"] >> cv_T;
      Eigen::Matrix4d T;
      cv::cv2eigen(cv_T, T);
      this->RIC.push_back(T.block<3, 3>(0, 0));
      this->TIC.push_back(T.block<3, 1>(0, 3));
    }
    VINS_INFO << "STEREO: " << this->STEREO;

    this->TD = (double)fsSettings["td"];
    this->ESTIMATE_TD = (int)fsSettings["estimate_td"];

    this->ROW = (int)fsSettings["image_height"];
    this->COL = (int)fsSettings["image_width"];
    if (!this->hasImu()) {
      this->ESTIMATE_EXTRINSIC = 0;
      this->ESTIMATE_TD = 0;
    }

    fsSettings.release();
  }

  std::string imuTopic() const { return this->imu.imuTopic(); }
  std::string imageTopic() const { return this->IMAGE0_TOPIC; }
  std::string image1Topic() const { return this->IMAGE1_TOPIC; }

  int max_num_iterations() const { return NUM_ITERATIONS; }

  double max_solver_time() const { return SOLVER_TIME; }

  /// @brief Get the number of cameras in the current configuration
  int getNumCameras() const { return NUM_OF_CAM; }

  /// @brief Check whether the system is configured for stereo camera input
  bool isUsingStereo() const { return STEREO; }
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
};

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;

enum SIZE_PARAMETERIZATION {
  SIZE_POSE = 7,
  SIZE_SPEEDBIAS = 9,
  SIZE_FEATURE = 1
};

enum StateOrder { O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12 };

enum NoiseOrder { O_AN = 0, O_GN = 3, O_AW = 6, O_GW = 9 };
