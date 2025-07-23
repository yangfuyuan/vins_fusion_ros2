# vins_fusion_ros2

This repository is a ROS 2 adaptation of the original [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) project. It supports visual-inertial odometry for both monocular and stereo camera setups, with optional GNSS/IMU fusion for global localization.

## Features

- Ported to ROS 2 (Humble and later)
- Compatible with stereo or monocular cameras
- Integration with IMU
- Supports tightly coupled sensor fusion
- Output of odometry and pose 

## Prerequisites
- **System**
  - Ubuntu 22.04
  - ROS2 humble
- **Libraries**
  - OpenCV (with CUDA enabled option)
  - [Ceres Solver-2.1.0](http://ceres-solver.org/installation.html) (you can refer [here](https://github.com/zinuok/VINS-Fusion#-ceres-solver-1); just edit 1.14.0 to 2.1.0 for install.)
  - [Eigen-3.3.9](https://github.com/zinuok/VINS-Fusion#-eigen-1)
  - glog

## Build Instructions

```bash
cd $(PATH_TO_YOUR_ROS2_WS)/src
git clone git@github.com:yangfuyuan/vins_fusion_ros2.git
cd ..
colcon build --symlink-install && source ./install/setup.bash && source ./install/local_setup.bash
```

## run
```bash
# vins
ros2 launch vins_fusion_ros2 vins_fusion_ros2.launch.py
```

## 9. License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.