/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <vins/common/sensor_data_type.h>
#include <vins/estimator/parameters.h>
#include <vins/utility/tic_toc.h>

#include <Eigen/Dense>
#include <list>
#include <map>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>
using namespace std;
using namespace Eigen;

class FeaturePerFrame {
 public:
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td) {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    cur_td = td;
    is_stereo = false;
  }
  void rightObservation(const Eigen::Matrix<double, 7, 1> &_point) {
    pointRight.x() = _point(0);
    pointRight.y() = _point(1);
    pointRight.z() = _point(2);
    uvRight.x() = _point(3);
    uvRight.y() = _point(4);
    velocityRight.x() = _point(5);
    velocityRight.y() = _point(6);
    is_stereo = true;
  }
  double cur_td;
  Vector3d point, pointRight;
  Vector2d uv, uvRight;
  Vector2d velocity, velocityRight;
  bool is_stereo;
};

class FeaturePerId {
 public:
  enum SolveStatus { NOT_SOLVED = 0, SOLVED_SUCCESS = 1, SOLVED_FAIL = 2 };
  const int feature_id;
  int start_frame = 0;
  std::vector<FeaturePerFrame> feature_per_frame;
  int used_num = 0;
  double estimated_depth = -1.0;
  SolveStatus solve_flag = NOT_SOLVED;
  FeaturePerId(int id, int start) : feature_id(id), start_frame(start) {}

  int endFrame() const {
    return start_frame + static_cast<int>(feature_per_frame.size()) - 1;
  }
  bool isSolved() const { return solve_flag == SOLVED_SUCCESS; }

  bool isSolveFailed() const { return solve_flag == SOLVED_FAIL; }
};

class FeatureManager {
 public:
  FeatureManager();

  void setOptions(std::shared_ptr<VINSOptions> options_);
  void clearState();
  int getFeatureCount();
  bool addFeatureCheckParallax(
      int frame_count,
      const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
      double td);
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l,
                                                    int frame_count_r);
  void setDepth(const VectorXd &x);
  void removeFailures();
  void clearDepth();
  VectorXd getDepthVector();
  void triangulate(int frameCnt, StateData states[], Vector3d tic[],
                   Matrix3d ric[]);
  void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                        Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1,
                        Eigen::Vector3d &point_3d);
  void initFramePoseByPnP(int frameCnt, StateData states[], Vector3d tic[],
                          Matrix3d ric[]);
  bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial,
                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P,
                            Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier(set<int> &outlierIndex);
  list<FeaturePerId> feature;
  int last_track_num;
  double last_average_parallax;
  int new_feature_num;
  int long_track_num;

 private:
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
  Matrix3d ric[2];
  std::shared_ptr<VINSOptions> options;
};

#endif
