/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once
#include <vins/common/sensor_data_type.h>
#include <vins/factor/imu_factor.h>
#include <vins/utility/utility.h>

#include <Eigen/Dense>
#include <map>
using namespace Eigen;
using namespace std;

class ImageFrame {
 public:
  ImageFrame(){};
  ImageFrame(
      const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points,
      double _t)
      : t{_t}, is_key_frame{false} {
    points = _points;
  };
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points;
  double t;
  Matrix3d R;
  Vector3d T;
  IntegrationBase::Ptr pre_integration;
  bool is_key_frame;
};
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame,
                        StateData *states);
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame,
                        StateData *states, Vector3d &g, VectorXd &x,
                        const VINSOptions &options);
