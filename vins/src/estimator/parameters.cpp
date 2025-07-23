/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins/estimator/parameters.h>
#include <vins/logger/logger.h>

#include <fstream>

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

void readParameters(std::string config_file) {
  FILE *fh = fopen(config_file.c_str(), "r");
  if (fh == NULL) {
    VINS_FATAL << "config_file dosen't exist; wrong config_file path";
    return;
  }
  fclose(fh);

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    VINS_FATAL << " Wrong path to settings";
  }

  fsSettings["image0_topic"] >> IMAGE0_TOPIC;
  fsSettings["image1_topic"] >> IMAGE1_TOPIC;
  MAX_CNT = fsSettings["max_cnt"];
  MIN_DIST = fsSettings["min_dist"];
  F_THRESHOLD = fsSettings["F_threshold"];
  SHOW_TRACK = fsSettings["show_track"];
  FLOW_BACK = fsSettings["flow_back"];

  USE_GPU = fsSettings["use_gpu"];
  USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];
  USE_GPU_CERES = fsSettings["use_gpu_ceres"];

  USE_IMU = fsSettings["imu"];
  VINS_INFO << "USE_IMU: " << USE_IMU;
  if (USE_IMU) {
    fsSettings["imu_topic"] >> IMU_TOPIC;
    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
  }

  SOLVER_TIME = fsSettings["max_solver_time"];
  NUM_ITERATIONS = fsSettings["max_num_iterations"];
  MIN_PARALLAX = fsSettings["keyframe_parallax"];
  MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

  fsSettings["output_path"] >> OUTPUT_FOLDER;
  VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
  VINS_INFO << "result path " << VINS_RESULT_PATH;
  std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
  fout.close();

  ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
  if (ESTIMATE_EXTRINSIC == 2) {
    RIC.push_back(Eigen::Matrix3d::Identity());
    TIC.push_back(Eigen::Vector3d::Zero());
    EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
  } else {
    if (ESTIMATE_EXTRINSIC == 1) {
      EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    cv::Mat cv_T;
    fsSettings["body_T_cam0"] >> cv_T;
    Eigen::Matrix4d T;
    cv::cv2eigen(cv_T, T);
    RIC.push_back(T.block<3, 3>(0, 0));
    TIC.push_back(T.block<3, 1>(0, 3));
  }

  NUM_OF_CAM = fsSettings["num_of_cam"];
  if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2) {
    VINS_FATAL << "num of cam should be 1 or 2 ";
  }

  int pn = config_file.find_last_of('/');
  std::string configPath = config_file.substr(0, pn);

  std::string cam0Calib;
  fsSettings["cam0_calib"] >> cam0Calib;
  std::string cam0Path = configPath + "/" + cam0Calib;
  CAM_NAMES.push_back(cam0Path);

  if (NUM_OF_CAM == 2) {
    STEREO = 1;
    std::string cam1Calib;
    fsSettings["cam1_calib"] >> cam1Calib;
    std::string cam1Path = configPath + "/" + cam1Calib;
    CAM_NAMES.push_back(cam1Path);
    cv::Mat cv_T;
    fsSettings["body_T_cam1"] >> cv_T;
    Eigen::Matrix4d T;
    cv::cv2eigen(cv_T, T);
    RIC.push_back(T.block<3, 3>(0, 0));
    TIC.push_back(T.block<3, 1>(0, 3));
  }
  VINS_INFO << "STEREO: " << STEREO;

  INIT_DEPTH = 5.0;
  BIAS_ACC_THRESHOLD = 0.1;
  BIAS_GYR_THRESHOLD = 0.1;

  TD = fsSettings["td"];
  ESTIMATE_TD = fsSettings["estimate_td"];

  ROW = fsSettings["image_height"];
  COL = fsSettings["image_width"];
  if (!USE_IMU) {
    ESTIMATE_EXTRINSIC = 0;
    ESTIMATE_TD = 0;
  }

  fsSettings.release();
}
