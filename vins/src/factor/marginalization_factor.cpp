/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/
#include <omp.h>
#include <vins/factor/marginalization_factor.h>
#include <vins/logger/logger.h>
void ResidualBlockInfo::Evaluate() {
  residuals.resize(cost_function->num_residuals());

  std::vector<int> block_sizes = cost_function->parameter_block_sizes();
  raw_jacobians.resize(block_sizes.size());
  jacobians.resize(block_sizes.size());

  for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
    jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
    raw_jacobians[i] = jacobians[i].data();
  }
  cost_function->Evaluate(parameter_blocks.data(), residuals.data(),
                          raw_jacobians.data());

  if (loss_function) {
    double residual_scaling_, alpha_sq_norm_;

    double sq_norm, rho[3];

    sq_norm = residuals.squaredNorm();
    loss_function->Evaluate(sq_norm, rho);

    double sqrt_rho1_ = sqrt(rho[1]);

    if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
      residual_scaling_ = sqrt_rho1_;
      alpha_sq_norm_ = 0.0;
    } else {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      residual_scaling_ = sqrt_rho1_ / (1 - alpha);
      alpha_sq_norm_ = alpha / sq_norm;
    }

    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
      jacobians[i] = sqrt_rho1_ * (jacobians[i] -
                                   alpha_sq_norm_ * residuals *
                                       (residuals.transpose() * jacobians[i]));
    }

    residuals *= residual_scaling_;
  }
}

MarginalizationInfo::~MarginalizationInfo() { clear(); }

void MarginalizationInfo::clear() {
  for (auto &factor : factors) {
    if (factor) {
      factor->clear();
    }
  }
  factors.clear();
  parameter_block_data.clear();

  parameter_block_size.clear();
  parameter_block_idx.clear();
  keep_block_size.clear();
  keep_block_idx.clear();
  keep_block_data.clear();
  linearized_jacobians.resize(0, 0);
  linearized_residuals.resize(0, 0);
  m = 0;
  n = 0;
  sum_block_size = 0;
  valid = false;
}

void MarginalizationInfo::addResidualBlockInfo(
    std::shared_ptr<ResidualBlockInfo> residual_block_info) {
  factors.emplace_back(std::move(residual_block_info));

  std::vector<double *> &parameter_blocks = factors.back()->parameter_blocks;
  std::vector<int> parameter_block_sizes =
      factors.back()->cost_function->parameter_block_sizes();

  for (int i = 0; i < static_cast<int>(factors.back()->parameter_blocks.size());
       i++) {
    double *addr = parameter_blocks[i];
    int size = parameter_block_sizes[i];
    parameter_block_size[reinterpret_cast<long>(addr)] = size;
  }

  for (int i = 0; i < static_cast<int>(factors.back()->drop_set.size()); i++) {
    double *addr = parameter_blocks[factors.back()->drop_set[i]];
    parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
  }
}

void MarginalizationInfo::preMarginalize() {
  for (auto it : factors) {
    it->Evaluate();

    std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
      long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
      int size = block_sizes[i];
      if (parameter_block_data.find(addr) == parameter_block_data.end()) {
        auto data = std::make_unique<double[]>(size);
        memcpy(data.get(), it->parameter_blocks[i], sizeof(double) * size);
        parameter_block_data[addr] = std::move(data);
      }
    }
  }
}

int MarginalizationInfo::localSize(int size) const {
  return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const {
  return size == 6 ? 7 : size;
}

void MarginalizationInfo::marginalize() {
  int pos = 0;
  for (auto &it : parameter_block_idx) {
    it.second = pos;
    pos += localSize(parameter_block_size[it.first]);
  }

  m = pos;

  for (const auto &it : parameter_block_size) {
    if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) {
      parameter_block_idx[it.first] = pos;
      pos += localSize(it.second);
    }
  }

  n = pos - m;
  if (m == 0) {
    valid = false;
    VINS_WARN << "unstable tracking...";
    return;
  }

  TicToc t_summing;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(pos, pos);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(pos);

  // 为每个线程准备局部变量
  std::vector<Eigen::MatrixXd> thread_As(NUM_THREADS,
                                         Eigen::MatrixXd::Zero(pos, pos));
  std::vector<Eigen::VectorXd> thread_bs(NUM_THREADS,
                                         Eigen::VectorXd::Zero(pos));

// OpenMP 并行遍历所有 factors
#pragma omp parallel for num_threads(NUM_THREADS)
  for (int t = 0; t < static_cast<int>(factors.size()); ++t) {
    auto &it = factors[t];
    int tid = omp_get_thread_num();

    Eigen::MatrixXd &local_A = thread_As[tid];
    Eigen::VectorXd &local_b = thread_bs[tid];

    for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
      long key_i = reinterpret_cast<long>(it->parameter_blocks[i]);
      int idx_i = parameter_block_idx[key_i];
      int size_i = parameter_block_size[key_i];
      if (size_i == 7) size_i = 6;

      Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);

      for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++) {
        long key_j = reinterpret_cast<long>(it->parameter_blocks[j]);
        int idx_j = parameter_block_idx[key_j];
        int size_j = parameter_block_size[key_j];
        if (size_j == 7) size_j = 6;

        Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);

        Eigen::MatrixXd JtJ = jacobian_i.transpose() * jacobian_j;
        if (i == j) {
          local_A.block(idx_i, idx_j, size_i, size_j) += JtJ;
        } else {
          local_A.block(idx_i, idx_j, size_i, size_j) += JtJ;
          local_A.block(idx_j, idx_i, size_j, size_i) += JtJ.transpose();
        }
      }
      local_b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
    }
  }

  // 汇总所有线程结果
  for (int i = 0; i < NUM_THREADS; ++i) {
    A += thread_As[i];
    b += thread_bs[i];
  }

  // 对称化主块
  Eigen::MatrixXd Amm =
      0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

  Eigen::MatrixXd Amm_inv =
      saes.eigenvectors() *
      Eigen::VectorXd((saes.eigenvalues().array() > eps)
                          .select(saes.eigenvalues().array().inverse(), 0))
          .asDiagonal() *
      saes.eigenvectors().transpose();

  Eigen::VectorXd bmm = b.segment(0, m);
  Eigen::MatrixXd Amr = A.block(0, m, m, n);
  Eigen::MatrixXd Arm = A.block(m, 0, n, m);
  Eigen::MatrixXd Arr = A.block(m, m, n, n);
  Eigen::VectorXd brr = b.segment(m, n);

  A = Arr - Arm * Amm_inv * Amr;
  b = brr - Arm * Amm_inv * bmm;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
  Eigen::VectorXd S =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                          .select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                          .select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

  linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  linearized_residuals =
      S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(
    std::unordered_map<long, double *> &addr_shift) {
  std::vector<double *> keep_block_addr;
  keep_block_size.clear();
  keep_block_idx.clear();
  keep_block_data.clear();

  for (const auto &it : parameter_block_idx) {
    if (it.second >= m) {
      keep_block_size.push_back(parameter_block_size[it.first]);
      keep_block_idx.push_back(parameter_block_idx[it.first]);
      keep_block_data.push_back(parameter_block_data[it.first].get());
      keep_block_addr.push_back(addr_shift[it.first]);
    }
  }
  sum_block_size = std::accumulate(std::begin(keep_block_size),
                                   std::end(keep_block_size), 0);

  return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(
    MarginalizationInfo::Ptr _marginalization_info)
    : marginalization_info(_marginalization_info) {
  auto info = marginalization_info.lock();
  if (!info) return;

  int cnt = 0;
  for (auto it : info->keep_block_size) {
    mutable_parameter_block_sizes()->push_back(it);
    cnt += it;
  }
  set_num_residuals(info->n);
};

bool MarginalizationFactor::Evaluate(double const *const *parameters,
                                     double *residuals,
                                     double **jacobians) const {
  auto info = marginalization_info.lock();
  if (!info) return false;
  int n = info->n;
  int m = info->m;
  Eigen::VectorXd dx(n);
  for (int i = 0; i < static_cast<int>(info->keep_block_size.size()); i++) {
    int size = info->keep_block_size[i];
    int idx = info->keep_block_idx[i] - m;
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
    Eigen::VectorXd x0 =
        Eigen::Map<const Eigen::VectorXd>(info->keep_block_data[i], size);
    if (size != 7)
      dx.segment(idx, size) = x - x0;
    else {
      dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
      dx.segment<3>(idx + 3) =
          2.0 * Utility::positify(
                    Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
                    Eigen::Quaterniond(x(6), x(3), x(4), x(5)))
                    .vec();
      if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
             Eigen::Quaterniond(x(6), x(3), x(4), x(5)))
                .w() >= 0)) {
        dx.segment<3>(idx + 3) =
            2.0 *
            -Utility::positify(
                 Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
                 Eigen::Quaterniond(x(6), x(3), x(4), x(5)))
                 .vec();
      }
    }
  }
  Eigen::Map<Eigen::VectorXd>(residuals, n) =
      info->linearized_residuals + info->linearized_jacobians * dx;
  if (jacobians) {
    for (int i = 0; i < static_cast<int>(info->keep_block_size.size()); i++) {
      if (jacobians[i]) {
        int size = info->keep_block_size[i], local_size = info->localSize(size);
        int idx = info->keep_block_idx[i] - m;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            jacobian(jacobians[i], n, size);
        jacobian.setZero();
        jacobian.leftCols(local_size) =
            info->linearized_jacobians.middleCols(idx, local_size);
      }
    }
  }
  return true;
}
