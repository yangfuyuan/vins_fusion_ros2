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
#include <vins/estimator/parameters.h>
#include <vins/logger/logger.h>
#include <vins/utility/utility.h>
using namespace Eigen;

class IntegrationBase : public std::enable_shared_from_this<IntegrationBase> {
 public:
  using Ptr = std::shared_ptr<IntegrationBase>;
  IntegrationBase() = delete;
  IntegrationBase(const IMUData &last, const Eigen::Vector3d &_linearized_ba,
                  const Eigen::Vector3d &_linearized_bg,
                  const ImuOptions &options)
      : last_imu(last),
        linearized_imu(last_imu),
        linearized_ba{_linearized_ba},
        linearized_bg{_linearized_bg},
        imu_options(options),
        jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
        covariance{Eigen::Matrix<double, 15, 15>::Zero()},
        sum_dt{0.0},
        delta_p{Eigen::Vector3d::Zero()},
        delta_q{Eigen::Quaterniond::Identity()},
        delta_v{Eigen::Vector3d::Zero()}

  {
    noise = Eigen::Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) =
        (imu_options.accelNoiseDensity * imu_options.accelNoiseDensity) *
        Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) =
        (imu_options.gyroNoiseDensity * imu_options.gyroNoiseDensity) *
        Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) =
        (imu_options.accelNoiseDensity * imu_options.accelNoiseDensity) *
        Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) =
        (imu_options.gyroNoiseDensity * imu_options.gyroNoiseDensity) *
        Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) =
        (imu_options.accelRandomWalk * imu_options.accelRandomWalk) *
        Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) =
        (imu_options.gyroRandomWalk * imu_options.gyroRandomWalk) *
        Eigen::Matrix3d::Identity();
  }

  ImuOptions getImuOptions() const { return imu_options; }

  void push_back(const IMUData &data) {
    imu_buffer.push_back(data);
    propagate(data);
  }

  void repropagate(const Eigen::Vector3d &_linearized_ba,
                   const Eigen::Vector3d &_linearized_bg) {
    sum_dt = 0.0;
    last_imu = linearized_imu;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    linearized_ba = _linearized_ba;
    linearized_bg = _linearized_bg;
    jacobian.setIdentity();
    covariance.setZero();
    for (int i = 0; i < static_cast<int>(imu_buffer.size()); i++)
      propagate(imu_buffer[i]);
  }

  void midPointIntegration(
      const IMUData &last, const IMUData &current,
      const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
      const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
      const Eigen::Vector3d &linearized_bg, Eigen::Vector3d &result_delta_p,
      Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
      Eigen::Vector3d &result_linearized_ba,
      Eigen::Vector3d &result_linearized_bg, bool update_jacobian) {
    double _dt = current.timestamp;
    double dt_sq = current.timestamp * current.timestamp;

    Vector3d un_acc_0 = delta_q * (last.linear_acceleration - linearized_ba);
    Vector3d un_gyr = 0.5 * (last.angular_velocity + current.angular_velocity) -
                      linearized_bg;
    result_delta_q =
        delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2,
                              un_gyr(2) * _dt / 2);
    Vector3d un_acc_1 =
        result_delta_q * (current.linear_acceleration - linearized_ba);
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * dt_sq;
    result_delta_v = delta_v + un_acc * _dt;
    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    if (update_jacobian) {
      Vector3d w_x = 0.5 * (last.angular_velocity + current.angular_velocity) -
                     linearized_bg;
      Vector3d a_0_x = last.linear_acceleration - linearized_ba;
      Vector3d a_1_x = current.linear_acceleration - linearized_ba;
      Matrix3d R_w_x, R_a_0_x, R_a_1_x;

      R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
      R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1),
          a_0_x(0), 0;
      R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1),
          a_1_x(0), 0;

      MatrixXd F = MatrixXd::Zero(15, 15);
      F.block<3, 3>(0, 0) = Matrix3d::Identity();
      F.block<3, 3>(0, 3) =
          -0.25 * delta_q.toRotationMatrix() * R_a_0_x * dt_sq +
          -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x *
              (Matrix3d::Identity() - R_w_x * _dt) * dt_sq;
      F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(0, 9) =
          -0.25 *
          (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) *
          dt_sq;
      F.block<3, 3>(0, 12) =
          -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * dt_sq * -_dt;
      F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
      F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x *
                                (Matrix3d::Identity() - R_w_x * _dt) * _dt;
      F.block<3, 3>(6, 6) = Matrix3d::Identity();
      F.block<3, 3>(6, 9) =
          -0.5 *
          (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) *
          _dt;
      F.block<3, 3>(6, 12) =
          -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
      F.block<3, 3>(9, 9) = Matrix3d::Identity();
      F.block<3, 3>(12, 12) = Matrix3d::Identity();
      // cout<<"A"<<endl<<A<<endl;

      MatrixXd V = MatrixXd::Zero(15, 18);
      V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * dt_sq;
      V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() *
                            R_a_1_x * dt_sq * 0.5 * _dt;
      V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * dt_sq;
      V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
      V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 3) =
          0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
      V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
      V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

      // step_jacobian = F;
      // step_V = V;
      jacobian = F * jacobian;
      covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }
  }

  void propagate(const IMUData &data) {
    if (data.timestamp > 1.0) {
      VINS_ERROR << "Abnormal IMU timestamp jump detected: " << data.timestamp;
      last_imu = data;
      return;
    }
    current_imu = data;
    Vector3d result_delta_p;
    Quaterniond result_delta_q;
    Vector3d result_delta_v;
    Vector3d result_linearized_ba;
    Vector3d result_linearized_bg;

    midPointIntegration(last_imu, data, delta_p, delta_q, delta_v,
                        linearized_ba, linearized_bg, result_delta_p,
                        result_delta_q, result_delta_v, result_linearized_ba,
                        result_linearized_bg, 1);

    // checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
    //                     linearized_ba, linearized_bg);
    delta_p = result_delta_p;
    delta_q = result_delta_q;
    delta_v = result_delta_v;
    linearized_ba = result_linearized_ba;
    linearized_bg = result_linearized_bg;
    delta_q.normalize();
    sum_dt += data.timestamp;
    last_imu = current_imu;
  }

  Eigen::Matrix<double, 15, 1> evaluate(
      const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
      const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
      const Eigen::Vector3d &Bgi, const Eigen::Vector3d &Pj,
      const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
      const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) {
    Eigen::Matrix<double, 15, 1> residuals;

    Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

    Eigen::Vector3d dba = Bai - linearized_ba;
    Eigen::Vector3d dbg = Bgi - linearized_bg;

    Eigen::Quaterniond corrected_delta_q =
        delta_q * Utility::deltaQ(dq_dbg * dbg);
    Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

    residuals.block<3, 1>(O_P, 0) =
        Qi.inverse() * (0.5 * imu_options.gravity() * sum_dt * sum_dt + Pj -
                        Pi - Vi * sum_dt) -
        corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) =
        2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) =
        Qi.inverse() * (imu_options.gravity() * sum_dt + Vj - Vi) -
        corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
  }

  IMUData last_imu;
  IMUData current_imu;
  const IMUData linearized_imu;

  Eigen::Vector3d linearized_ba, linearized_bg;

  Eigen::Matrix<double, 15, 15> jacobian, covariance;
  Eigen::Matrix<double, 15, 15> step_jacobian;
  Eigen::Matrix<double, 15, 18> step_V;
  Eigen::Matrix<double, 18, 18> noise;

  double sum_dt;
  Eigen::Vector3d delta_p;
  Eigen::Quaterniond delta_q;
  Eigen::Vector3d delta_v;
  std::vector<IMUData> imu_buffer;
  ImuOptions imu_options;
};
