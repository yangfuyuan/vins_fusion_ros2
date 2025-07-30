/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins/estimator/feature_manager.h>
#include <vins/logger/logger.h>

FeatureManager::FeatureManager() {
  for (int i = 0; i < 2; i++) ric[i].setIdentity();
}

void FeatureManager::setOptions(std::shared_ptr<VINSOptions> options_) {
  options = options_;
  for (int i = 0; i < 2; i++) {
    ric[i] = options->RIC[i];
  }
}

void FeatureManager::clearState() { feature.clear(); }

int FeatureManager::getFeatureCount() {
  int cnt = 0;
  for (auto &it : feature) {
    it.used_num = it.feature_per_frame.size();
    if (it.used_num >= 4) {
      cnt++;
    }
  }
  return cnt;
}

bool FeatureManager::addFeatureCheckParallax(
    int frame_count,
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    double td) {
  double parallax_sum = 0;
  int parallax_num = 0;
  last_track_num = 0;
  last_average_parallax = 0;
  new_feature_num = 0;
  long_track_num = 0;
  for (auto &id_pts : image) {
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
    assert(id_pts.second[0].first == 0);
    if (id_pts.second.size() == 2) {
      f_per_fra.rightObservation(id_pts.second[1].second);
      assert(id_pts.second[1].first == 1);
    }

    int feature_id = id_pts.first;
    auto it = find_if(feature.begin(), feature.end(),
                      [feature_id](const FeaturePerId &it) {
                        return it.feature_id == feature_id;
                      });

    if (it == feature.end()) {
      feature.push_back(FeaturePerId(feature_id, frame_count));
      feature.back().feature_per_frame.push_back(f_per_fra);
      new_feature_num++;
    } else if (it->feature_id == feature_id) {
      it->feature_per_frame.push_back(f_per_fra);
      last_track_num++;
      if (it->feature_per_frame.size() >= 4) long_track_num++;
    }
  }
  if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 ||
      new_feature_num > 0.5 * last_track_num)
    return true;

  for (auto &it_per_id : feature) {
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >=
            frame_count - 1) {
      parallax_sum += compensatedParallax2(it_per_id, frame_count);
      parallax_num++;
    }
  }

  if (parallax_num == 0) {
    return true;
  } else {
    last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
    return parallax_sum / parallax_num >= options->min_parllaax_num;
  }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(
    int frame_count_l, int frame_count_r) {
  vector<pair<Vector3d, Vector3d>> corres;
  for (auto &it : feature) {
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
      Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;

      a = it.feature_per_frame[idx_l].point;

      b = it.feature_per_frame[idx_r].point;

      corres.push_back(make_pair(a, b));
    }
  }
  return corres;
}

void FeatureManager::setDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4) continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
    if (it_per_id.estimated_depth < 0) {
      it_per_id.solve_flag = FeaturePerId::SOLVED_FAIL;
    } else
      it_per_id.solve_flag = FeaturePerId::SOLVED_SUCCESS;
  }
}

void FeatureManager::removeFailures() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    if (it->isSolveFailed()) feature.erase(it);
  }
}

void FeatureManager::clearDepth() {
  for (auto &it_per_id : feature) it_per_id.estimated_depth = -1;
}

VectorXd FeatureManager::getDepthVector() {
  VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4) continue;
#if 1
    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
    dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
  }
  return dep_vec;
}

void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                                      Eigen::Matrix<double, 3, 4> &Pose1,
                                      Eigen::Vector2d &point0,
                                      Eigen::Vector2d &point1,
                                      Eigen::Vector3d &point_3d) {
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Eigen::Vector4d triangulated_point;
  triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P,
                                    vector<cv::Point2f> &pts2D,
                                    vector<cv::Point3f> &pts3D) {
  Eigen::Matrix3d R_initial;
  Eigen::Vector3d P_initial;

  // w_T_cam ---> cam_T_w
  R_initial = R.inverse();
  P_initial = -(R_initial * P);

  if (int(pts2D.size()) < 4) {
    VINS_WARN
        << ("feature tracking not enough, please slowly move you device!");
    return false;
  }
  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
  if (!pnp_succ) {
    VINS_ERROR << "pnp failed !";
    return false;
  }
  cv::Rodrigues(rvec, r);
  Eigen::MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  Eigen::MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);

  // cam_T_w ---> w_T_cam
  R = R_pnp.transpose();
  P = R * (-T_pnp);

  return true;
}

void FeatureManager::initFramePoseByPnP(int frameCnt, StateData states[],
                                        Vector3d tic[], Matrix3d ric[]) {
  if (frameCnt > 0) {
    vector<cv::Point2f> pts2D;
    vector<cv::Point3f> pts3D;
    for (auto &it_per_id : feature) {
      if (it_per_id.estimated_depth > 0) {
        int index = frameCnt - it_per_id.start_frame;
        if ((int)it_per_id.feature_per_frame.size() >= index + 1) {
          Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point *
                                        it_per_id.estimated_depth) +
                              tic[0];
          Vector3d ptsInWorld =
              states[it_per_id.start_frame].rotation * ptsInCam +
              states[it_per_id.start_frame].position;

          cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
          cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(),
                              it_per_id.feature_per_frame[index].point.y());
          pts3D.push_back(point3d);
          pts2D.push_back(point2d);
        }
      }
    }
    Eigen::Matrix3d RCam;
    Eigen::Vector3d PCam;
    // trans to w_T_cam
    RCam = states[frameCnt - 1].rotation * ric[0];
    PCam =
        states[frameCnt - 1].rotation * tic[0] + states[frameCnt - 1].position;

    if (solvePoseByPnP(RCam, PCam, pts2D, pts3D)) {
      // trans to w_T_imu
      states[frameCnt].rotation = RCam * ric[0].transpose();
      states[frameCnt].position = -RCam * ric[0].transpose() * tic[0] + PCam;
    }
  }
}

void FeatureManager::triangulate(int frameCnt, StateData states[],
                                 Vector3d tic[], Matrix3d ric[]) {
  for (auto &it_per_id : feature) {
    if (it_per_id.estimated_depth > 0) continue;

    if (options->isUsingStereo() && it_per_id.feature_per_frame[0].is_stereo) {
      int imu_i = it_per_id.start_frame;
      Eigen::Matrix<double, 3, 4> leftPose;
      Eigen::Vector3d t0 =
          states[imu_i].position + states[imu_i].rotation * tic[0];
      Eigen::Matrix3d R0 = states[imu_i].rotation * ric[0];
      leftPose.leftCols<3>() = R0.transpose();
      leftPose.rightCols<1>() = -R0.transpose() * t0;
      Eigen::Matrix<double, 3, 4> rightPose;
      Eigen::Vector3d t1 =
          states[imu_i].position + states[imu_i].rotation * tic[1];
      Eigen::Matrix3d R1 = states[imu_i].rotation * ric[1];
      rightPose.leftCols<3>() = R1.transpose();
      rightPose.rightCols<1>() = -R1.transpose() * t1;

      Eigen::Vector2d point0, point1;
      Eigen::Vector3d point3d;
      point0 = it_per_id.feature_per_frame[0].point.head(2);
      point1 = it_per_id.feature_per_frame[0].pointRight.head(2);

      triangulatePoint(leftPose, rightPose, point0, point1, point3d);
      Eigen::Vector3d localPoint;
      localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
      double depth = localPoint.z();
      if (depth > 0)
        it_per_id.estimated_depth = depth;
      else
        it_per_id.estimated_depth = options->init_estimated_depth;
      continue;
    } else if (it_per_id.feature_per_frame.size() > 1) {
      int imu_i = it_per_id.start_frame;
      Eigen::Matrix<double, 3, 4> leftPose;
      Eigen::Vector3d t0 =
          states[imu_i].position + states[imu_i].rotation * tic[0];
      Eigen::Matrix3d R0 = states[imu_i].rotation * ric[0];
      leftPose.leftCols<3>() = R0.transpose();
      leftPose.rightCols<1>() = -R0.transpose() * t0;

      imu_i++;
      Eigen::Matrix<double, 3, 4> rightPose;
      Eigen::Vector3d t1 =
          states[imu_i].position + states[imu_i].rotation * tic[0];
      Eigen::Matrix3d R1 = states[imu_i].rotation * ric[0];
      rightPose.leftCols<3>() = R1.transpose();
      rightPose.rightCols<1>() = -R1.transpose() * t1;

      Eigen::Vector2d point0, point1;
      Eigen::Vector3d point3d;
      point0 = it_per_id.feature_per_frame[0].point.head(2);
      point1 = it_per_id.feature_per_frame[1].point.head(2);
      triangulatePoint(leftPose, rightPose, point0, point1, point3d);
      Eigen::Vector3d localPoint;
      localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
      double depth = localPoint.z();
      if (depth > 0)
        it_per_id.estimated_depth = depth;
      else
        it_per_id.estimated_depth = options->init_estimated_depth;
      continue;
    }
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4) continue;

    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;

    Eigen::Matrix<double, 3, 4> P0;
    Eigen::Vector3d t0 =
        states[imu_i].position + states[imu_i].rotation * tic[0];
    Eigen::Matrix3d R0 = states[imu_i].rotation * ric[0];
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;

      Eigen::Vector3d t1 =
          states[imu_j].position + states[imu_j].rotation * tic[0];
      Eigen::Matrix3d R1 = states[imu_j].rotation * ric[0];
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);
      Eigen::Matrix3d R = R0.transpose() * R1;
      Eigen::Matrix<double, 3, 4> P;
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized();
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

      if (imu_i == imu_j) continue;
    }
    assert(svd_idx == svd_A.rows());
    Eigen::Vector4d svd_V =
        Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
            .matrixV()
            .rightCols<1>();
    double svd_method = svd_V[2] / svd_V[3];
    it_per_id.estimated_depth = svd_method;

    if (it_per_id.estimated_depth < 0.1) {
      it_per_id.estimated_depth = options->init_estimated_depth;
    }
  }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex) {
  std::set<int>::iterator itSet;
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    int index = it->feature_id;
    itSet = outlierIndex.find(index);
    if (itSet != outlierIndex.end()) {
      feature.erase(it);
    }
  }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R,
                                          Eigen::Vector3d marg_P,
                                          Eigen::Matrix3d new_R,
                                          Eigen::Vector3d new_P) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() < 2) {
        feature.erase(it);
        continue;
      } else {
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
        double dep_j = pts_j(2);
        if (dep_j > 0)
          it->estimated_depth = dep_j;
        else
          it->estimated_depth = options->init_estimated_depth;
      }
    }
  }
}

void FeatureManager::removeBack() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

void FeatureManager::removeFront(int frame_count) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame == frame_count) {
      it->start_frame--;
    } else {
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endFrame() < frame_count - 1) continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0) feature.erase(it);
    }
  }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id,
                                            int frame_count) {
  // check the second last frame is keyframe or not
  // parallax betwwen seconde last frame and third last frame
  const FeaturePerFrame &frame_i =
      it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
  const FeaturePerFrame &frame_j =
      it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

  double ans = 0;
  Vector3d p_j = frame_j.point;

  double u_j = p_j(0);
  double v_j = p_j(1);

  Vector3d p_i = frame_i.point;
  Vector3d p_i_comp;
  p_i_comp = p_i;
  double dep_i = p_i(2);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  double du = u_i - u_j, dv = v_i - v_j;

  double dep_i_comp = p_i_comp(2);
  double u_i_comp = p_i_comp(0) / dep_i_comp;
  double v_i_comp = p_i_comp(1) / dep_i_comp;
  double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

  ans = max(
      ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

  return ans;
}
