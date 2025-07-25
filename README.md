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

### Install Dependencies
```bash
sudo apt-get install libgoogle-glog-dev libeigen3-dev libceres-dev libopencv-dev 
```
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
## Converting ROS 1 Bag Files to ROS 2 Format

### ROS 2 cannot play ROS 1 `.bag` files directly

Unfortunately, you **cannot directly play** a `.bag` file recorded in **ROS 1** using **ROS 2** tools like `ros2 bag play`.

This is because:

- ROS 1 `.bag` files are stored as a **single binary file**.
- ROS 2 `.bag` files use a **folder-based structure** containing:
  - a `.db3` SQLite3 database,
  - metadata YAML files.

---

### Convert ROS 1 `.bag` to ROS 2 `.db3`

You can convert ROS 1 `.bag` files to ROS 2 `.db3` format using [`rosbags`](https://pypi.org/project/rosbags/).

#### 1. Install `rosbags`

```bash
pip install rosbags
```

If you're using `~/.local/bin` (default for user-level installs), you may need to add it to your shell PATH:

```bash
export PATH=$PATH:~/.local/bin
```

You may also consider putting the above line in your `.bashrc` or `.zshrc`.

---

#### 2. Convert the `.bag` file

```bash
rosbags-convert foo.bag --dst /path/to/output_folder \
  --src-typestore ros1_noetic \
  --dst-typestore ros2_humble
```

- `foo.bag`: your original ROS 1 bag file
- `/path/to/output_folder`: target folder where converted ROS 2 `.db3` bag will be saved
- `ros1_noetic` / `ros2_humble`: choose according to your actual ROS versions

You can omit `--src-typestore` and `--dst-typestore` if you want to use the default type inference.

---

#### 3. Play the converted bag file

Once conversion is complete, you can play it back using ROS 2 tools:

```bash
ros2 bag play /path/to/output_folder
```

---

### Additional Options

You can also control compression and message filtering:

```bash
rosbags-convert foo.bag --dst bar \
  --compress zstd \
  --compress-mode message \
  --include-topic /camera/image_raw /imu
```

See `rosbags-convert --help` for more.

---

## License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.