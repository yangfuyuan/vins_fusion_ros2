#ifndef DATA_TYPE_H
#define DATA_TYPE_H
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <vector>

enum class SolverState { INITIAL, NON_LINEAR };
enum class MarginalizationType { MARGIN_OLD, MARGIN_SECOND_NEW };

using Timestamp = double;
using FeatureFrame =
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;
using TimestampedFeatureFrame = std::pair<Timestamp, FeatureFrame>;
using TimestampedVector3d = std::pair<Timestamp, Eigen::Vector3d>;

#endif  //
