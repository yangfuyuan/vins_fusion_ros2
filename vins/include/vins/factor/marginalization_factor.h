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
#include <pthread.h>
#include <vins/utility/tic_toc.h>
#include <vins/utility/utility.h>

#include <cstdlib>
#include <unordered_map>

const int NUM_THREADS = 4;

struct ResidualBlockInfo {
  ResidualBlockInfo(ceres::CostFunction *_cost_function,
                    ceres::LossFunction *_loss_function,
                    std::vector<double *> _parameter_blocks,
                    std::vector<int> _drop_set)
      : cost_function(_cost_function),
        loss_function(_loss_function),
        parameter_blocks(_parameter_blocks),
        drop_set(_drop_set) {}

  void Evaluate();

  ceres::CostFunction *cost_function;
  ceres::LossFunction *loss_function;
  std::vector<double *> parameter_blocks;
  std::vector<int> drop_set;

  double **raw_jacobians;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      jacobians;
  Eigen::VectorXd residuals;

  int localSize(int size) { return size == 7 ? 6 : size; }
};

struct ThreadsStruct {
  std::vector<ResidualBlockInfo *> sub_factors;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  std::unordered_map<long, int> parameter_block_size;  // global size
  std::unordered_map<long, int> parameter_block_idx;   // local size
};

class MarginalizationInfo
    : public std::enable_shared_from_this<MarginalizationInfo> {
 public:
  using Ptr = std::shared_ptr<MarginalizationInfo>;
  MarginalizationInfo() { valid = true; };
  ~MarginalizationInfo();
  int localSize(int size) const;
  int globalSize(int size) const;
  void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
  void preMarginalize();
  void marginalize();
  std::vector<double *> getParameterBlocks(
      std::unordered_map<long, double *> &addr_shift);

  std::vector<ResidualBlockInfo *> factors;
  int m, n;
  std::unordered_map<long, int> parameter_block_size;  // global size
  int sum_block_size;
  std::unordered_map<long, int> parameter_block_idx;  // local size
  std::unordered_map<long, double *> parameter_block_data;

  std::vector<int> keep_block_size;  // global size
  std::vector<int> keep_block_idx;   // local size
  std::vector<double *> keep_block_data;

  Eigen::MatrixXd linearized_jacobians;
  Eigen::VectorXd linearized_residuals;
  const double eps = 1e-8;
  bool valid;
};

class MarginalizationFactor : public ceres::CostFunction {
 public:
  MarginalizationFactor(MarginalizationInfo::Ptr _marginalization_info);
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  MarginalizationInfo::Ptr marginalization_info;
};
