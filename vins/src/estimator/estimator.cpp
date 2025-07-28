/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins/estimator/estimator.h>

Estimator::Estimator() : featureManager{rotations} {
  resetState();
  isRunning.store(true);
  processThread = std::thread(&Estimator::processMeasurements, this);
}

Estimator::~Estimator() {
  if (isRunning.load()) {
    isRunning.store(false);
    imuCondition.notify_all();
    featureCondition.notify_all();
    if (processThread.joinable()) {
      processThread.join();
    }
  }
}

void Estimator::resetState() {
  {
    std::lock_guard<std::mutex> imu_lock(imu_mutex);
    clearBuffer(imuBuffer);
  }
  {
    std::lock_guard<std::mutex> feature_lock(featureBufferMutex);
    clearBuffer(featureBuffer);
  }

  std::lock_guard<std::mutex> lock(processingMutex);
  previousTimestamp = -1;
  currentTimestamp = 0;
  openExEstimation = 0;
  initialPosition = Eigen::Vector3d(0, 0, 0);
  initialRotation = Eigen::Matrix3d::Identity();
  inputImageCount = 0;
  isFirstPoseInitialized = false;

  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    rotations[i].setIdentity();
    positions[i].setZero();
    velocities[i].setZero();
    accelerometerBiases[i].setZero();
    gyroscopeBiases[i].setZero();
    std::vector<IMUData>().swap(deltaTimeImuBuffer[i]);
    pre_integrations[i] = nullptr;
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    cameraTranslation[i] = Vector3d::Zero();
    cameraRotation[i] = Matrix3d::Identity();
    updateCameraPose(i);
  }

  isFirstIMUReceived = false, backCount = 0;
  frontCount = 0;
  frameCount = 0;
  solver_flag = SolverState::INITIAL;
  initialTimestamp = 0;
  std::map<double, ImageFrame>().swap(all_image_frame);
  tmp_pre_integration = nullptr;
  last_marginalization_info = nullptr;
  last_marginalization_parameter_blocks.clear();

  featureManager.clearState();
  failure_occur = 0;
  VINS_INFO << "reset state successfully";
}

void Estimator::setParameter() {
  std::lock_guard<std::mutex> lock(processingMutex);
  for (int i = 0; i < NUM_OF_CAM; i++) {
    cameraTranslation[i] = TIC[i];
    cameraRotation[i] = RIC[i];
    updateCameraPose(i);
    VINS_INFO << " exitrinsic cam[" << i << "]: \n"
              << cameraRotation[i] << "\n"
              << cameraTranslation[i].transpose();
  }
  featureManager.setRic(cameraRotation);
  ProjectionTwoFrameOneCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionTwoFrameTwoCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionOneFrameTwoCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  timeDelay = TD;
  gravity = G;
  featureTracker.readIntrinsicParameter(CAM_NAMES);
}

void Estimator::changeSensorType(int use_imu, int use_stereo) {
  bool restart = false;
  processingMutex.lock();
  if (!use_imu && !use_stereo)
    VINS_ERROR << "at least use two sensors! ";
  else {
    if (USE_IMU != use_imu) {
      USE_IMU = use_imu;
      if (USE_IMU) {
        restart = true;
      } else {
        tmp_pre_integration = nullptr;
        last_marginalization_info = nullptr;
        last_marginalization_parameter_blocks.clear();
      }
    }

    STEREO = use_stereo;
    VINS_INFO << "use imu: " << USE_IMU << " use stereo: " << STEREO;
  }
  processingMutex.unlock();
  if (restart) {
    resetState();
    setParameter();
  }
}

void Estimator::inputImage(const ImageData &image) {
  inputImageCount++;
  FeatureFrame featureFrame;
  TicToc featureTrackerTime;

  if (image.image1.empty()) {
    featureFrame = featureTracker.trackImage(image.timestamp, image.image0);
  } else {
    featureFrame =
        featureTracker.trackImage(image.timestamp, image.image0, image.image1);
  }
  if (SHOW_TRACK) {
    track_image.image0 = featureTracker.getTrackImage();
    track_image.timestamp = image.timestamp;
    safe_track_image.set(track_image);
  }

  if (inputImageCount % 2 == 0 || featureBuffer.empty()) {
    {
      std::lock_guard<std::mutex> lock(featureBufferMutex);
      featureBuffer.push(make_pair(image.timestamp, featureFrame));
    }
    featureCondition.notify_one();
  }
}

void Estimator::inputIMU(const IMUData &imu) {
  {
    std::lock_guard<std::mutex> lock(imu_mutex);
    imuBuffer.push(imu);
  }
  imuCondition.notify_all();

  if (solver_flag == SolverState::NON_LINEAR) {
    fastPredictIMU(imu);
  }
}

void Estimator::inputFeature(double timestamp,
                             const FeatureFrame &featureFrame) {
  {
    std::lock_guard<std::mutex> lock(featureBufferMutex);
    featureBuffer.push(make_pair(timestamp, featureFrame));
  }
  featureCondition.notify_one();
}

bool Estimator::getIMUInterval(double startTime, double endTime,
                               vector<IMUData> &data) {
  std::lock_guard<std::mutex> lock(imu_mutex);
  if (imuBuffer.empty()) {
    VINS_ERROR << "No IMU data received";
    return false;
  }

  if (endTime <= imuBuffer.back().timestamp) {
    while (imuBuffer.front().timestamp <= startTime) {
      imuBuffer.pop();
    }

    while (imuBuffer.front().timestamp < endTime) {
      data.push_back(imuBuffer.front());
      imuBuffer.pop();
    }
    if (!imuBuffer.empty()) {
      data.push_back(imuBuffer.front());
    }
    return true;
  }

  VINS_WARN << "Waiting for IMU data";
  return false;
}

bool Estimator::IMUAvailable(double t) {
  return !imuBuffer.empty() && t <= imuBuffer.back().timestamp;
}

void Estimator::processMeasurements() {
  while (isRunning.load()) {
    TimestampedFeatureFrame feature;
    vector<IMUData> imu_datas;
    {
      std::unique_lock<std::mutex> lock(featureBufferMutex);
      featureCondition.wait(
          lock, [this] { return !featureBuffer.empty() || !isRunning.load(); });
      if (!isRunning.load() || featureBuffer.empty()) break;
      feature = featureBuffer.front();
      featureBuffer.pop();
      lock.unlock();
    }
    currentTimestamp = feature.first + timeDelay;
    {
      std::unique_lock<std::mutex> lock(imu_mutex);

      imuCondition.wait(lock, [this] {
        return IMUAvailable(currentTimestamp) || !isRunning.load();
      });
      if (!isRunning.load()) {
        break;
      }
    }

    if (USE_IMU) {
      getIMUInterval(previousTimestamp, currentTimestamp, imu_datas);
      if (!isFirstPoseInitialized) initFirstIMUPose(imu_datas);
      for (size_t i = 0; i < imu_datas.size(); i++) {
        double dt;
        if (i == 0)
          dt = imu_datas[i].timestamp - previousTimestamp;
        else if (i == imu_datas.size() - 1)
          dt = currentTimestamp - imu_datas[i - 1].timestamp;
        else
          dt = imu_datas[i].timestamp - imu_datas[i - 1].timestamp;
        processIMU(imu_datas[i], dt);
      }
    }
    {
      std::lock_guard<std::mutex> lock(processingMutex);
      processImage(feature.second, feature.first);
      printStatistics(currentTimestamp);
      collectPointCloudAll(feature.first);
      previousTimestamp = currentTimestamp;
    }
  }
}
void Estimator::updateCameraPose(int index) {
  PoseData pose;
  pose.position = cameraTranslation[index];
  pose.orientation = Quaterniond(cameraRotation[index]);
  safe_camera_pose[index].set(pose);
}

void Estimator::collectPointCloudAll(Timestamp timestamp) {
  PointCloudData main_cloud;
  PointCloudData point_cloud;
  PointCloudData margin_cloud;
  main_cloud.timestamp = timestamp;
  point_cloud.timestamp = timestamp;
  margin_cloud.timestamp = timestamp;

  for (const auto &it_per_id : featureManager.feature) {
    int used_num = it_per_id.feature_per_frame.size();
    int start_frame = it_per_id.start_frame;

    // Only use well-tracked and solved features
    if (used_num < 2 || !it_per_id.isSolved()) continue;

    int imu_i = start_frame;
    const auto &first_obs = it_per_id.feature_per_frame[0];
    Eigen::Vector3d pts_i = first_obs.point * it_per_id.estimated_depth;
    Eigen::Vector3d w_pts_i =
        rotations[imu_i] * (cameraRotation[0] * pts_i + cameraTranslation[0]) +
        positions[imu_i];

    // 1. Main cloud
    if (start_frame < WINDOW_SIZE - 2 && start_frame <= WINDOW_SIZE * 3 / 4) {
      main_cloud.points.emplace_back(w_pts_i);
    }

    // 2. Margin cloud
    if (start_frame == 0 && used_num <= 2) {
      margin_cloud.points.emplace_back(w_pts_i);
    }

    // 3. Current frame cloud
    if (solver_flag == SolverState::NON_LINEAR &&
        marginalization_flag == MarginalizationType::MARGIN_OLD) {
      if (start_frame < WINDOW_SIZE - 2 &&
          start_frame + used_num - 1 >= WINDOW_SIZE - 2) {
        int imu_j = WINDOW_SIZE - 2 - start_frame;
        const auto &cur_obs = it_per_id.feature_per_frame[imu_j];
        point_cloud.points.emplace_back(w_pts_i);

        ChannelFloat p_2d;
        p_2d.values.push_back(cur_obs.point.x());
        p_2d.values.push_back(cur_obs.point.y());
        p_2d.values.push_back(cur_obs.uv.x());
        p_2d.values.push_back(cur_obs.uv.y());
        p_2d.values.push_back(it_per_id.feature_id);
        point_cloud.channels.push_back(p_2d);
      }
    }
  }
  if (!main_cloud.points.empty()) {
    safe_main_cloud.set(main_cloud);
  }
  if (!margin_cloud.points.empty()) {
    safe_margin_cloud.set(margin_cloud);
  }
  if (!point_cloud.points.empty()) {
    safe_point_cloud.set(point_cloud);
  }
  if (solver_flag == SolverState::NON_LINEAR &&
      marginalization_flag == MarginalizationType::MARGIN_OLD) {
    int i = WINDOW_SIZE - 2;
    PoseData pose;
    pose.timestamp = Headers[i];
    pose.position = positions[i];
    pose.orientation = Quaterniond(rotations[i]);
    safe_keyframe_pose.set(pose);
  }
}

void Estimator::printStatistics(Timestamp timestamp) {
  if (solver_flag != SolverState::NON_LINEAR) return;
  VINS_DEBUG << "position: (" << positions[WINDOW_SIZE].x() << ","
             << positions[WINDOW_SIZE].y() << "," << positions[WINDOW_SIZE].z()
             << ")";
}

void Estimator::initFirstIMUPose(const vector<IMUData> &data) {
  isFirstPoseInitialized = true;
  Vector3d averageAcceleration = Vector3d::Zero();

  for (const auto &imu : data) {
    averageAcceleration += imu.linear_acceleration;
  }

  averageAcceleration /= data.size();
  VINS_INFO << "Average acceleration: " << averageAcceleration.transpose();

  Matrix3d initialRotation = Utility::g2R(averageAcceleration);
  double yaw = Utility::R2ypr(initialRotation).x();
  initialRotation = Utility::ypr2R(Vector3d{-yaw, 0, 0}) * initialRotation;
  rotations[0] = initialRotation;
  VINS_INFO << "init R0: \n" << rotations[0];
}

void Estimator::setFirstPose(const Eigen::Vector3d &position,
                             const Eigen::Matrix3d &rotation) {
  positions[0] = position;
  rotations[0] = rotation;
  initialPosition = position;
  initialRotation = rotation;
}

void Estimator::processIMU(const IMUData &data, double deltaTime) {
  if (!isFirstIMUReceived) {
    isFirstIMUReceived = true;
    previousImuData = data;
  }

  if (!pre_integrations[frameCount]) {
    pre_integrations[frameCount] = std::make_shared<IntegrationBase>(
        previousImuData, accelerometerBiases[frameCount],
        gyroscopeBiases[frameCount]);
  }
  if (frameCount != 0) {
    IMUData delta_imu = data;
    delta_imu.timestamp = deltaTime;

    pre_integrations[frameCount]->push_back(delta_imu);
    tmp_pre_integration->push_back(delta_imu);
    deltaTimeImuBuffer[frameCount].push_back(delta_imu);

    int j = frameCount;
    Vector3d un_acc_0 = rotations[j] * (previousImuData.linear_acceleration -
                                        accelerometerBiases[j]) -
                        gravity;
    Vector3d un_gyr =
        0.5 * (previousImuData.angular_velocity + data.angular_velocity) -
        gyroscopeBiases[j];
    rotations[j] *= Utility::deltaQ(un_gyr * deltaTime).toRotationMatrix();
    Vector3d un_acc_1 =
        rotations[j] * (data.linear_acceleration - accelerometerBiases[j]) -
        gravity;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    positions[j] +=
        deltaTime * velocities[j] + 0.5 * deltaTime * deltaTime * un_acc;
    velocities[j] += deltaTime * un_acc;
  }
  previousImuData = data;
}

void Estimator::processImage(const FeatureFrame &features,
                             Timestamp timestamp) {
  if (featureManager.addFeatureCheckParallax(frameCount, features, timeDelay)) {
    marginalization_flag = MarginalizationType::MARGIN_OLD;
  } else {
    marginalization_flag = MarginalizationType::MARGIN_SECOND_NEW;
  }

  Headers[frameCount] = timestamp;

  ImageFrame imageframe(features, timestamp);
  imageframe.pre_integration = std::move(tmp_pre_integration);
  all_image_frame.insert(make_pair(timestamp, imageframe));
  tmp_pre_integration = std::make_shared<IntegrationBase>(
      previousImuData, accelerometerBiases[frameCount],
      gyroscopeBiases[frameCount]);

  if (ESTIMATE_EXTRINSIC == 2) {
    if (frameCount != 0) {
      vector<pair<Vector3d, Vector3d>> corres =
          featureManager.getCorresponding(frameCount - 1, frameCount);
      Matrix3d calib_ric;
      if (initial_ex_rotation.CalibrationExRotation(
              corres, pre_integrations[frameCount]->delta_q, calib_ric)) {
        cameraRotation[0] = calib_ric;
        RIC[0] = calib_ric;
        ESTIMATE_EXTRINSIC = 1;
      }
    }
  }

  if (solver_flag == SolverState::INITIAL) {
    // monocular + IMU initilization
    if (!STEREO && USE_IMU) {
      if (frameCount == WINDOW_SIZE) {
        bool result = false;
        if (ESTIMATE_EXTRINSIC != 2 && (timestamp - initialTimestamp) > 0.1) {
          result = initialStructure();
          initialTimestamp = timestamp;
        }
        if (result) {
          optimize();
          updateLatestStates();
          solver_flag = SolverState::NON_LINEAR;
          slideWindow();
          VINS_INFO << "Initialization complete. Switching to NON_LINEAR mode.";
        } else {
          slideWindow();
        }
      }
    }

    // stereo + IMU initilization
    if (STEREO && USE_IMU) {
      featureManager.initFramePoseByPnP(frameCount, positions, rotations,
                                        cameraTranslation, cameraRotation);
      featureManager.triangulate(frameCount, positions, rotations,
                                 cameraTranslation, cameraRotation);
      if (frameCount == WINDOW_SIZE) {
        map<double, ImageFrame>::iterator frame_it;
        int i = 0;
        for (frame_it = all_image_frame.begin();
             frame_it != all_image_frame.end(); frame_it++) {
          frame_it->second.R = rotations[i];
          frame_it->second.T = positions[i];
          i++;
        }
        solveGyroscopeBias(all_image_frame, gyroscopeBiases);
        for (int i = 0; i <= WINDOW_SIZE; i++) {
          pre_integrations[i]->repropagate(Vector3d::Zero(),
                                           gyroscopeBiases[i]);
        }
        optimize();
        updateLatestStates();
        solver_flag = SolverState::NON_LINEAR;
        slideWindow();
        VINS_INFO << "Initialization complete. Switching to NON_LINEAR mode.";
      }
    }

    // stereo only initilization
    if (STEREO && !USE_IMU) {
      featureManager.initFramePoseByPnP(frameCount, positions, rotations,
                                        cameraTranslation, cameraRotation);
      featureManager.triangulate(frameCount, positions, rotations,
                                 cameraTranslation, cameraRotation);
      optimize();

      if (frameCount == WINDOW_SIZE) {
        optimize();
        updateLatestStates();
        solver_flag = SolverState::NON_LINEAR;
        slideWindow();
        VINS_INFO << "Initialization complete. Switching to NON_LINEAR mode.";
      }
    }

    if (frameCount < WINDOW_SIZE) {
      frameCount++;
      int prev_frame = frameCount - 1;
      positions[frameCount] = positions[prev_frame];
      velocities[frameCount] = velocities[prev_frame];
      rotations[frameCount] = rotations[prev_frame];
      accelerometerBiases[frameCount] = accelerometerBiases[prev_frame];
      gyroscopeBiases[frameCount] = gyroscopeBiases[prev_frame];
    }

  } else {
    if (!USE_IMU) {
      featureManager.initFramePoseByPnP(frameCount, positions, rotations,
                                        cameraTranslation, cameraRotation);
    }
    featureManager.triangulate(frameCount, positions, rotations,
                               cameraTranslation, cameraRotation);

    // optimization
    TicToc t_solve;
    optimize();
    set<int> removeIndex;
    outliersRejection(removeIndex);
    featureManager.removeOutlier(removeIndex);
    if (failureDetection()) {
      failure_occur = 1;
      resetState();
      setParameter();
      return;
    }

    slideWindow();
    featureManager.removeFailures();
    // prepare output of VINS
    {
      key_poses.timestamp = timestamp;
      key_poses.poses.clear();
      for (int i = 0; i <= WINDOW_SIZE; i++)
        key_poses.poses.push_back(positions[i]);

      safe_key_poses.set(key_poses);
    }

    lastRotation = rotations[WINDOW_SIZE];
    lastPosition = positions[WINDOW_SIZE];
    lastRotation0 = rotations[0];
    lastPosition0 = positions[0];
    updateLatestStates();
  }
  vio_odom.timestamp = timestamp;
  vio_odom.position = positions[WINDOW_SIZE];
  vio_odom.orientation = Quaterniond(rotations[WINDOW_SIZE]);
  vio_odom.velocity = velocities[WINDOW_SIZE];
  if (solver_flag == SolverState::NON_LINEAR) {
    safe_vio_odom.set(vio_odom);
  }
}

bool Estimator::initialStructure() {
  TicToc timer;
  {
    auto frameIt = all_image_frame.begin();
    Vector3d sumG = Vector3d::Zero();
    for (++frameIt; frameIt != all_image_frame.end(); ++frameIt) {
      double dt = frameIt->second.pre_integration->sum_dt;
      Vector3d tmpG = frameIt->second.pre_integration->delta_v / dt;
      sumG += tmpG;
    }

    Vector3d averageG = sumG / (all_image_frame.size() - 1);
    double variance = 0;
    for (frameIt = std::next(all_image_frame.begin());
         frameIt != all_image_frame.end(); ++frameIt) {
      double dt = frameIt->second.pre_integration->sum_dt;
      Vector3d tmpG = frameIt->second.pre_integration->delta_v / dt;
      variance += (tmpG - averageG).squaredNorm();
    }
    variance = std::sqrt(variance / (all_image_frame.size() - 1));
    if (variance < 0.25) {
      VINS_WARN << "IMU excitation not enouth";
    }
  }

  std::vector<Quaterniond> rotations(WINDOW_SIZE + 1);
  std::vector<Vector3d> positions(WINDOW_SIZE + 1);
  std::map<int, Vector3d> sfmPoints;
  std::vector<SFMFeature> sfmFeatures;

  for (auto &feature : featureManager.feature) {
    int startFrame = feature.start_frame - 1;
    SFMFeature sfmFeature;
    sfmFeature.state = false;
    sfmFeature.id = feature.feature_id;

    for (auto &frame : feature.feature_per_frame) {
      startFrame++;
      Vector3d point = frame.point;
      sfmFeature.observation.push_back({startFrame, {point.x(), point.y()}});
    }
    sfmFeatures.push_back(sfmFeature);
  }

  Matrix3d relativeRotation;
  Vector3d relativeTranslation;
  int referenceFrame;
  if (!computeRelativePose(relativeRotation, relativeTranslation,
                           referenceFrame)) {
    VINS_WARN << "Not enough features or parallax; Move device around!!!";
    return false;
  }

  GlobalSFM sfm;
  if (!sfm.construct(frameCount + 1, rotations, positions, referenceFrame,
                     relativeRotation, relativeTranslation, sfmFeatures,
                     sfmPoints)) {
    marginalization_flag = MarginalizationType::MARGIN_OLD;
    VINS_WARN << "Failed to construect sfm!!!";
    return false;
  }

  auto frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
    if (frame_it->first == Headers[i]) {
      frame_it->second.is_key_frame = true;
      frame_it->second.R = rotations[i].toRotationMatrix() * RIC[0].transpose();
      frame_it->second.T = positions[i];
      i++;
      continue;
    }
    if (frame_it->first > Headers[i]) {
      i++;
    }
    cv::Mat r, rvec, t, D, tmp_r;

    Matrix3d R_inital = (rotations[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * positions[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    frame_it->second.is_key_frame = false;
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto &id_pts : frame_it->second.points) {
      int feature_id = id_pts.first;
      for (auto &i_p : id_pts.second) {
        auto it = sfmPoints.find(feature_id);
        if (it != sfmPoints.end()) {
          Vector3d world_pts = it->second;
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6) {
      VINS_WARN << "Not enough points for solve pnp:  " << pts_3_vector.size();
      return false;
    }
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
      VINS_WARN << "Failed to solvo pnp!!!";
      return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp, tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * RIC[0].transpose();
    frame_it->second.T = T_pnp;
  }

  return visualInitialAlign();
}

bool Estimator::visualInitialAlign() {
  TicToc t_g;
  VectorXd x;
  // solve scale
  bool result =
      VisualIMUAlignment(all_image_frame, gyroscopeBiases, gravity, x);
  if (!result) {
    VINS_DEBUG << "misalign visual structure with IMU";
    return false;
  }

  // change state
  for (int i = 0; i <= frameCount; i++) {
    Matrix3d Ri = all_image_frame[Headers[i]].R;
    Vector3d Pi = all_image_frame[Headers[i]].T;
    positions[i] = Pi;
    rotations[i] = Ri;
    all_image_frame[Headers[i]].is_key_frame = true;
  }

  double s = (x.tail<1>())(0);
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    pre_integrations[i]->repropagate(Vector3d::Zero(), gyroscopeBiases[i]);
  }
  for (int i = frameCount; i >= 0; i--)
    positions[i] = s * positions[i] - rotations[i] * TIC[0] -
                   (s * positions[0] - rotations[0] * TIC[0]);
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end();
       frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      velocities[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }

  Matrix3d R0 = Utility::g2R(gravity);
  double yaw = Utility::R2ypr(R0 * rotations[0]).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  gravity = R0 * gravity;
  // Matrix3d rot_diff = R0 * rotations[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frameCount; i++) {
    positions[i] = rot_diff * positions[i];
    rotations[i] = rot_diff * rotations[i];
    velocities[i] = rot_diff * velocities[i];
  }
  featureManager.clearDepth();
  featureManager.triangulate(frameCount, positions, rotations,
                             cameraTranslation, cameraRotation);

  return true;
}

bool Estimator::computeRelativePose(Matrix3d &relative_R, Vector3d &relative_T,
                                    int &referenceFrame) {
  for (int i = 0; i < WINDOW_SIZE; i++) {
    auto correspondences = featureManager.getCorresponding(i, WINDOW_SIZE);
    if (correspondences.size() > 20) {
      double totalParallax = 0;
      for (const auto &corr : correspondences) {
        Vector2d pts0(corr.first(0), corr.first(1));
        Vector2d pts1(corr.second(0), corr.second(1));
        totalParallax += (pts0 - pts1).norm();
      }

      double avgParallax = totalParallax / correspondences.size();
      if (avgParallax * 460 > 30 &&
          m_estimator.solveRelativeRT(correspondences, relative_R,
                                      relative_T)) {
        referenceFrame = i;
        return true;
      }
    }
  }
  return false;
}

void Estimator::prepareParameters() {
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    para_Pose[i][0] = positions[i].x();
    para_Pose[i][1] = positions[i].y();
    para_Pose[i][2] = positions[i].z();
    Quaterniond q{rotations[i]};
    para_Pose[i][3] = q.x();
    para_Pose[i][4] = q.y();
    para_Pose[i][5] = q.z();
    para_Pose[i][6] = q.w();

    if (USE_IMU) {
      para_SpeedBias[i][0] = velocities[i].x();
      para_SpeedBias[i][1] = velocities[i].y();
      para_SpeedBias[i][2] = velocities[i].z();

      para_SpeedBias[i][3] = accelerometerBiases[i].x();
      para_SpeedBias[i][4] = accelerometerBiases[i].y();
      para_SpeedBias[i][5] = accelerometerBiases[i].z();

      para_SpeedBias[i][6] = gyroscopeBiases[i].x();
      para_SpeedBias[i][7] = gyroscopeBiases[i].y();
      para_SpeedBias[i][8] = gyroscopeBiases[i].z();
    }
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    para_Ex_Pose[i][0] = cameraTranslation[i].x();
    para_Ex_Pose[i][1] = cameraTranslation[i].y();
    para_Ex_Pose[i][2] = cameraTranslation[i].z();
    Quaterniond q{cameraRotation[i]};
    para_Ex_Pose[i][3] = q.x();
    para_Ex_Pose[i][4] = q.y();
    para_Ex_Pose[i][5] = q.z();
    para_Ex_Pose[i][6] = q.w();
  }

  VectorXd dep = featureManager.getDepthVector();
  for (int i = 0; i < featureManager.getFeatureCount(); i++)
    para_Feature[i][0] = dep(i);

  para_Td[0][0] = timeDelay;
}

void Estimator::updateEstimates() {
  Vector3d origin_R0 = Utility::R2ypr(rotations[0]);
  Vector3d origin_P0 = positions[0];

  if (failure_occur) {
    origin_R0 = Utility::R2ypr(lastRotation0);
    origin_P0 = lastPosition0;
    failure_occur = 0;
  }

  if (USE_IMU) {
    Vector3d origin_R00 =
        Utility::R2ypr(Quaterniond(para_Pose[0][6], para_Pose[0][3],
                                   para_Pose[0][4], para_Pose[0][5])
                           .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 ||
        abs(abs(origin_R00.y()) - 90) < 1.0) {
      rot_diff = rotations[0] * Quaterniond(para_Pose[0][6], para_Pose[0][3],
                                            para_Pose[0][4], para_Pose[0][5])
                                    .toRotationMatrix()
                                    .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {
      rotations[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3],
                                            para_Pose[i][4], para_Pose[i][5])
                                    .normalized()
                                    .toRotationMatrix();

      positions[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                         para_Pose[i][1] - para_Pose[0][1],
                                         para_Pose[i][2] - para_Pose[0][2]) +
                     origin_P0;

      velocities[i] =
          rot_diff * Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1],
                              para_SpeedBias[i][2]);

      accelerometerBiases[i] = Vector3d(
          para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);

      gyroscopeBiases[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7],
                                    para_SpeedBias[i][8]);
    }
  } else {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
      rotations[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3],
                                 para_Pose[i][4], para_Pose[i][5])
                         .normalized()
                         .toRotationMatrix();

      positions[i] =
          Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
    }
  }

  if (USE_IMU) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
      cameraTranslation[i] =
          Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
      cameraRotation[i] = Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3],
                                      para_Ex_Pose[i][4], para_Ex_Pose[i][5])
                              .normalized()
                              .toRotationMatrix();
      updateCameraPose(i);
    }
  }

  VectorXd dep = featureManager.getDepthVector();
  for (int i = 0; i < featureManager.getFeatureCount(); i++)
    dep(i) = para_Feature[i][0];
  featureManager.setDepth(dep);

  if (USE_IMU) timeDelay = para_Td[0][0];
}

bool Estimator::failureDetection() {
  return false;
  if (featureManager.last_track_num < 2) {
  }
  if (accelerometerBiases[WINDOW_SIZE].norm() > 2.5) {
    return true;
  }
  if (gyroscopeBiases[WINDOW_SIZE].norm() > 1.0) {
    return true;
  }
  Vector3d tmp_P = positions[WINDOW_SIZE];
  if ((tmp_P - lastPosition).norm() > 5) {
    // return true;
  }
  if (abs(tmp_P.z() - lastPosition.z()) > 1) {
    // return true;
  }
  Matrix3d tmp_R = rotations[WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * lastRotation;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) {
    // return true;
  }
  return false;
}

void Estimator::AddPoseParameterBlocks(ceres::Problem &problem) {
  for (int i = 0; i < frameCount + 1; i++) {
    auto *local_param = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_param);
    if (USE_IMU) problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
  }
  if (!USE_IMU) problem.SetParameterBlockConstant(para_Pose[0]);
}
void Estimator::AddExtrinsicParameterBlocks(ceres::Problem &problem) {
  for (int i = 0; i < NUM_OF_CAM; i++) {
    auto *local_param = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_param);
    if ((ESTIMATE_EXTRINSIC && frameCount == WINDOW_SIZE &&
         velocities[0].norm() > 0.2) ||
        openExEstimation) {
      openExEstimation = 1;
    } else {
      problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
  }
}
void Estimator::AddTimeDelayParameterBlock(ceres::Problem &problem) {
  problem.AddParameterBlock(para_Td[0], 1);
  if (!ESTIMATE_TD || velocities[0].norm() < 0.2)
    problem.SetParameterBlockConstant(para_Td[0]);
}
void Estimator::AddMarginalizationFactor(ceres::Problem &problem) {
  if (last_marginalization_info && last_marginalization_info->valid) {
    auto *marginalization_factor =
        new MarginalizationFactor(last_marginalization_info);
    problem.AddResidualBlock(marginalization_factor, nullptr,
                             last_marginalization_parameter_blocks);
  }
}
void Estimator::AddIMUFactors(ceres::Problem &problem) {
  if (!USE_IMU) return;

  for (int i = 0; i < frameCount; i++) {
    int j = i + 1;
    if (pre_integrations[j]->sum_dt > 10.0) continue;
    auto *imu_factor = new IMUFactor(pre_integrations[j]);
    problem.AddResidualBlock(imu_factor, nullptr, para_Pose[i],
                             para_SpeedBias[i], para_Pose[j],
                             para_SpeedBias[j]);
  }
}
void Estimator::AddFeatureFactors(ceres::Problem &problem) {
  int feature_index = -1;
  for (auto &it : featureManager.feature) {
    if (it.feature_per_frame.size() < 4) continue;
    ++feature_index;

    int imu_i = it.start_frame, imu_j = imu_i - 1;
    Vector3d pts_i = it.feature_per_frame[0].point;

    for (auto &f : it.feature_per_frame) {
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = f.point;
        auto *factor = new ProjectionTwoFrameOneCamFactor(
            pts_i, pts_j, it.feature_per_frame[0].velocity, f.velocity,
            it.feature_per_frame[0].cur_td, f.cur_td);
        problem.AddResidualBlock(factor, new ceres::HuberLoss(1.0),
                                 para_Pose[imu_i], para_Pose[imu_j],
                                 para_Ex_Pose[0], para_Feature[feature_index],
                                 para_Td[0]);
      }

      if (STEREO && f.is_stereo) {
        Vector3d pts_j_right = f.pointRight;
        if (imu_i != imu_j) {
          auto *factor = new ProjectionTwoFrameTwoCamFactor(
              pts_i, pts_j_right, it.feature_per_frame[0].velocity,
              f.velocityRight, it.feature_per_frame[0].cur_td, f.cur_td);
          problem.AddResidualBlock(factor, new ceres::HuberLoss(1.0),
                                   para_Pose[imu_i], para_Pose[imu_j],
                                   para_Ex_Pose[0], para_Ex_Pose[1],
                                   para_Feature[feature_index], para_Td[0]);
        } else {
          auto *factor = new ProjectionOneFrameTwoCamFactor(
              pts_i, pts_j_right, it.feature_per_frame[0].velocity,
              f.velocityRight, it.feature_per_frame[0].cur_td, f.cur_td);
          problem.AddResidualBlock(factor, new ceres::HuberLoss(1.0),
                                   para_Ex_Pose[0], para_Ex_Pose[1],
                                   para_Feature[feature_index], para_Td[0]);
        }
      }
    }
  }
}
void Estimator::solveOptimization() {
  ceres::Problem problem;

  AddPoseParameterBlocks(problem);
  AddExtrinsicParameterBlocks(problem);
  AddTimeDelayParameterBlock(problem);
  AddMarginalizationFactor(problem);
  AddIMUFactors(problem);
  AddFeatureFactors(problem);

  ceres::Solver::Options options;
  options.linear_solver_type =
      USE_GPU_CERES ? ceres::DENSE_QR : ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = NUM_ITERATIONS;
  options.max_solver_time_in_seconds =
      (marginalization_flag == MarginalizationType::MARGIN_OLD)
          ? SOLVER_TIME * 0.8
          : SOLVER_TIME;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}

void Estimator::processOldMarginalization() {
  auto marginalization_info = std::make_shared<MarginalizationInfo>();
  prepareParameters();

  if (last_marginalization_info && last_marginalization_info->valid) {
    vector<int> drop_set;
    for (int i = 0;
         i < static_cast<int>(last_marginalization_parameter_blocks.size());
         i++) {
      if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
          last_marginalization_parameter_blocks[i] == para_SpeedBias[0]) {
        drop_set.push_back(i);
      }
    }

    auto marginalization_factor =
        std::make_shared<MarginalizationFactor>(last_marginalization_info);
    auto residual_block_info = std::make_shared<ResidualBlockInfo>(
        marginalization_factor, nullptr, last_marginalization_parameter_blocks,
        drop_set);
    marginalization_info->addResidualBlockInfo(residual_block_info);
  }

  // IMU
  if (USE_IMU && pre_integrations[1]->sum_dt < 10.0) {
    marginalization_info->addResidualBlockInfo(createIMUResidualBlock());
  }

  // Features
  addFeatureResidualBlocks(marginalization_info);

  marginalization_info->preMarginalize();
  marginalization_info->marginalize();

  auto addr_shift = createAddrShift(true);
  auto parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
  last_marginalization_info = std::move(marginalization_info);
  last_marginalization_parameter_blocks = parameter_blocks;
}
void Estimator::processNewMarginalization() {
  if (!last_marginalization_info) return;
  if (std::count(last_marginalization_parameter_blocks.begin(),
                 last_marginalization_parameter_blocks.end(),
                 para_Pose[WINDOW_SIZE - 1])) {
    auto marginalization_info = std::make_shared<MarginalizationInfo>();
    prepareParameters();

    if (last_marginalization_info->valid) {
      vector<int> drop_set;
      for (int i = 0;
           i < static_cast<int>(last_marginalization_parameter_blocks.size());
           i++) {
        if (last_marginalization_parameter_blocks[i] ==
            para_Pose[WINDOW_SIZE - 1]) {
          drop_set.push_back(i);
        }
      }

      auto marginalization_factor =
          std::make_shared<MarginalizationFactor>(last_marginalization_info);
      auto residual_block_info = std::make_shared<ResidualBlockInfo>(
          marginalization_factor, nullptr,
          last_marginalization_parameter_blocks, drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    marginalization_info->preMarginalize();
    marginalization_info->marginalize();

    auto addr_shift = createAddrShift(false);
    auto parameter_blocks =
        marginalization_info->getParameterBlocks(addr_shift);
    last_marginalization_info = std::move(marginalization_info);
    last_marginalization_parameter_blocks = parameter_blocks;
  }
}
std::shared_ptr<ResidualBlockInfo> Estimator::createIMUResidualBlock() {
  auto imu_factor = std::make_shared<IMUFactor>(pre_integrations[1]);
  return std::make_shared<ResidualBlockInfo>(
      imu_factor, nullptr,
      std::vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1],
                            para_SpeedBias[1]},
      std::vector<int>{0, 1});
}
void Estimator::addFeatureResidualBlocks(
    std::shared_ptr<MarginalizationInfo> &marg_info) {
  int feature_index = -1;

  for (auto &it_per_id : featureManager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4) continue;

    ++feature_index;

    int imu_i = it_per_id.start_frame;
    if (imu_i != 0) continue;

    int imu_j = imu_i - 1;
    const Vector3d &pts_i = it_per_id.feature_per_frame[0].point;

    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      const Vector3d &pts_j = it_per_frame.point;

      if (imu_i != imu_j) {
        auto f_td = std::make_shared<ProjectionTwoFrameOneCamFactor>(
            pts_i, pts_j, it_per_id.feature_per_frame[0].velocity,
            it_per_frame.velocity, it_per_id.feature_per_frame[0].cur_td,
            it_per_frame.cur_td);
        auto loss = std::make_shared<ceres::HuberLoss>(1.0);
        marg_info->addResidualBlockInfo(std::make_shared<ResidualBlockInfo>(
            f_td, loss,
            vector<double *>{para_Pose[imu_i], para_Pose[imu_j],
                             para_Ex_Pose[0], para_Feature[feature_index],
                             para_Td[0]},
            vector<int>{0, 3}));
      }

      if (STEREO && it_per_frame.is_stereo) {
        const Vector3d &pts_j_right = it_per_frame.pointRight;
        auto loss = std::make_shared<ceres::HuberLoss>(1.0);

        if (imu_i != imu_j) {
          auto f = std::make_shared<ProjectionTwoFrameTwoCamFactor>(
              pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
              it_per_frame.velocityRight, it_per_id.feature_per_frame[0].cur_td,
              it_per_frame.cur_td);
          marg_info->addResidualBlockInfo(std::make_shared<ResidualBlockInfo>(
              f, loss,
              vector<double *>{para_Pose[imu_i], para_Pose[imu_j],
                               para_Ex_Pose[0], para_Ex_Pose[1],
                               para_Feature[feature_index], para_Td[0]},
              vector<int>{0, 4}));
        } else {
          auto f = std::make_shared<ProjectionOneFrameTwoCamFactor>(
              pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
              it_per_frame.velocityRight, it_per_id.feature_per_frame[0].cur_td,
              it_per_frame.cur_td);
          marg_info->addResidualBlockInfo(std::make_shared<ResidualBlockInfo>(
              f, loss,
              vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1],
                               para_Feature[feature_index], para_Td[0]},
              vector<int>{2}));
        }
      }
    }
  }
}
std::unordered_map<long, double *> Estimator::createAddrShift(bool is_old) {
  std::unordered_map<long, double *> addr_shift;

  for (int i = 0; i <= WINDOW_SIZE; i++) {
    if (!is_old && i == WINDOW_SIZE - 1) continue;

    if (i == WINDOW_SIZE && !is_old) {
      addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
      if (USE_IMU)
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
            para_SpeedBias[i - 1];
    } else {
      addr_shift[reinterpret_cast<long>(para_Pose[i])] =
          para_Pose[i - (is_old ? 1 : 0)];
      if (USE_IMU)
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
            para_SpeedBias[i - (is_old ? 1 : 0)];
    }
  }

  for (int i = 0; i < NUM_OF_CAM; i++)
    addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

  addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

  return addr_shift;
}

void Estimator::optimize() {
  prepareParameters();
  solveOptimization();
  updateEstimates();

  if (frameCount < WINDOW_SIZE) return;
  if (marginalization_flag == MarginalizationType::MARGIN_OLD) {
    processOldMarginalization();
  } else {
    processNewMarginalization();
  }
}

void Estimator::slideWindow() {
  TicToc t_margin;
  if (marginalization_flag == MarginalizationType::MARGIN_OLD) {
    double t_0 = Headers[0];
    backRotation = rotations[0];
    backPosition = positions[0];
    if (frameCount == WINDOW_SIZE) {
      for (int i = 0; i < WINDOW_SIZE; i++) {
        Headers[i] = Headers[i + 1];
        rotations[i].swap(rotations[i + 1]);
        positions[i].swap(positions[i + 1]);
        if (USE_IMU) {
          std::swap(pre_integrations[i], pre_integrations[i + 1]);
          deltaTimeImuBuffer[i].swap(deltaTimeImuBuffer[i + 1]);

          velocities[i].swap(velocities[i + 1]);
          accelerometerBiases[i].swap(accelerometerBiases[i + 1]);
          gyroscopeBiases[i].swap(gyroscopeBiases[i + 1]);
        }
      }
      Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
      positions[WINDOW_SIZE] = positions[WINDOW_SIZE - 1];
      rotations[WINDOW_SIZE] = rotations[WINDOW_SIZE - 1];

      if (USE_IMU) {
        velocities[WINDOW_SIZE] = velocities[WINDOW_SIZE - 1];
        accelerometerBiases[WINDOW_SIZE] = accelerometerBiases[WINDOW_SIZE - 1];
        gyroscopeBiases[WINDOW_SIZE] = gyroscopeBiases[WINDOW_SIZE - 1];

        pre_integrations[WINDOW_SIZE] = nullptr;
        pre_integrations[WINDOW_SIZE] = std::make_shared<IntegrationBase>(
            previousImuData, accelerometerBiases[WINDOW_SIZE],
            gyroscopeBiases[WINDOW_SIZE]);

        deltaTimeImuBuffer[WINDOW_SIZE].clear();
      }

      if (true || solver_flag == SolverState::INITIAL) {
        map<double, ImageFrame>::iterator it_0;
        it_0 = all_image_frame.find(t_0);
        all_image_frame.erase(all_image_frame.begin(), it_0);
      }
      slideWindowOld();
    }
  } else {
    if (frameCount == WINDOW_SIZE) {
      Headers[frameCount - 1] = Headers[frameCount];
      positions[frameCount - 1] = positions[frameCount];
      rotations[frameCount - 1] = rotations[frameCount];

      if (USE_IMU) {
        for (unsigned int i = 0; i < deltaTimeImuBuffer[frameCount].size();
             i++) {
          const auto &imu = deltaTimeImuBuffer[frameCount][i];
          pre_integrations[frameCount - 1]->push_back(imu);
          deltaTimeImuBuffer[frameCount - 1].push_back(imu);
        }

        velocities[frameCount - 1] = velocities[frameCount];
        accelerometerBiases[frameCount - 1] = accelerometerBiases[frameCount];
        gyroscopeBiases[frameCount - 1] = gyroscopeBiases[frameCount];

        pre_integrations[WINDOW_SIZE] = nullptr;
        pre_integrations[WINDOW_SIZE] = std::make_shared<IntegrationBase>(
            previousImuData, accelerometerBiases[WINDOW_SIZE],
            gyroscopeBiases[WINDOW_SIZE]);

        deltaTimeImuBuffer[WINDOW_SIZE].clear();
      }
      slideWindowNew();
    }
  }
}

void Estimator::slideWindowNew() {
  frontCount++;
  featureManager.removeFront(frameCount);
}

void Estimator::slideWindowOld() {
  backCount++;

  bool shift_depth = solver_flag == SolverState::NON_LINEAR ? true : false;
  if (shift_depth) {
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = backRotation * cameraRotation[0];
    R1 = rotations[0] * cameraRotation[0];
    P0 = backPosition + backRotation * cameraTranslation[0];
    P1 = positions[0] + rotations[0] * cameraTranslation[0];
    featureManager.removeBackShiftDepth(R0, P0, R1, P1);
  } else
    featureManager.removeBack();
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = rotations[frameCount];
  T.block<3, 1>(0, 3) = positions[frameCount];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = rotations[index];
  T.block<3, 1>(0, 3) = positions[index];
}

bool Estimator::getIntegratedImuOdom(OdomData &data) {
  if (!safe_imu_pre_odom.check()) {
    return false;
  }
  safe_imu_pre_odom.get(data);
  return true;
}
bool Estimator::getVisualInertialOdom(OdomData &data) {
  if (!safe_vio_odom.check()) {
    return false;
  }
  safe_vio_odom.get(data);
  return true;
}

bool Estimator::getKeyPoses(PoseSequenceData &poses) {
  if (!safe_key_poses.check()) {
    return false;
  }
  safe_key_poses.get(poses);
  if (poses.poses.size() < 1) {
    return false;
  }
  return true;
}

bool Estimator::getCameraPose(int index, PoseData &data) {
  safe_camera_pose[index].get(data, true);
  return true;
}

bool Estimator::getTrackImage(ImageData &image) {
  if (!safe_track_image.check()) {
    return false;
  }
  safe_track_image.get(image);
  return true;
}

bool Estimator::getMainCloud(PointCloudData &data) {
  if (!safe_main_cloud.check()) {
    return false;
  }
  safe_main_cloud.get(data);
  return true;
}
bool Estimator::getMarginCloud(PointCloudData &data) {
  if (!safe_margin_cloud.check()) {
    return false;
  }
  safe_margin_cloud.get(data);
  return true;
}
bool Estimator::getkeyframeCloud(PointCloudData &data) {
  if (!safe_point_cloud.check()) {
    return false;
  }
  safe_point_cloud.get(data);
  return true;
}

bool Estimator::getkeyframePose(PoseData &data) {
  if (!safe_keyframe_pose.check()) {
    return false;
  }
  safe_keyframe_pose.get(data);
  return true;
}

void Estimator::predictPtsInNextFrame() {
  if (frameCount < 2) return;
  Eigen::Matrix4d curT, prevT, nextT;
  getPoseInWorldFrame(curT);
  getPoseInWorldFrame(frameCount - 1, prevT);
  nextT = curT * (prevT.inverse() * curT);
  map<int, Eigen::Vector3d> predictPts;

  for (auto &it_per_id : featureManager.feature) {
    if (it_per_id.estimated_depth > 0) {
      int firstIndex = it_per_id.start_frame;
      int lastIndex =
          it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
      if ((int)it_per_id.feature_per_frame.size() >= 2 &&
          lastIndex == frameCount) {
        double depth = it_per_id.estimated_depth;
        Vector3d pts_j =
            cameraRotation[0] * (depth * it_per_id.feature_per_frame[0].point) +
            cameraTranslation[0];
        Vector3d pts_w = rotations[firstIndex] * pts_j + positions[firstIndex];
        Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() *
                             (pts_w - nextT.block<3, 1>(0, 3));
        Vector3d pts_cam =
            cameraRotation[0].transpose() * (pts_local - cameraTranslation[0]);
        int ptsIndex = it_per_id.feature_id;
        predictPts[ptsIndex] = pts_cam;
      }
    }
  }
  featureTracker.setPrediction(predictPts);
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici,
                                    Vector3d &tici, Matrix3d &Rj, Vector3d &Pj,
                                    Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi,
                                    Vector3d &uvj) {
  Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
  Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
  Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
  double rx = residual.x();
  double ry = residual.y();
  return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex) {
  // return;
  int feature_index = -1;
  for (auto &it_per_id : featureManager.feature) {
    double err = 0;
    int errCnt = 0;
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4) continue;
    feature_index++;
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
    double depth = it_per_id.estimated_depth;
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = it_per_frame.point;
        double tmp_error = reprojectionError(
            rotations[imu_i], positions[imu_i], cameraRotation[0],
            cameraTranslation[0], rotations[imu_j], positions[imu_j],
            cameraRotation[0], cameraTranslation[0], depth, pts_i, pts_j);
        err += tmp_error;
        errCnt++;
      }
      // need to rewrite projecton factor.........
      if (STEREO && it_per_frame.is_stereo) {
        Vector3d pts_j_right = it_per_frame.pointRight;
        if (imu_i != imu_j) {
          double tmp_error = reprojectionError(
              rotations[imu_i], positions[imu_i], cameraRotation[0],
              cameraTranslation[0], rotations[imu_j], positions[imu_j],
              cameraRotation[1], cameraTranslation[1], depth, pts_i,
              pts_j_right);
          err += tmp_error;
          errCnt++;
        } else {
          double tmp_error = reprojectionError(
              rotations[imu_i], positions[imu_i], cameraRotation[0],
              cameraTranslation[0], rotations[imu_j], positions[imu_j],
              cameraRotation[1], cameraTranslation[1], depth, pts_i,
              pts_j_right);
          err += tmp_error;
          errCnt++;
        }
      }
    }
    double ave_err = err / errCnt;
    if (ave_err * FOCAL_LENGTH > 3) removeIndex.insert(it_per_id.feature_id);
  }
}

void Estimator::fastPredictIMU(const IMUData &data) {
  std::lock_guard<std::mutex> lock(propagateMutex);
  double dt = data.timestamp - latestImuData.timestamp;
  Eigen::Vector3d un_acc_0 =
      latest_Q * (latestImuData.linear_acceleration - latestAccelBias) -
      gravity;
  Eigen::Vector3d un_gyr =
      0.5 * (latestImuData.angular_velocity + data.angular_velocity) -
      latestGyroBias;
  latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
  Eigen::Vector3d un_acc_1 =
      latest_Q * (data.linear_acceleration - latestAccelBias) - gravity;
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  latestPosition =
      latestPosition + dt * latestVelocity + 0.5 * dt * dt * un_acc;
  latestVelocity = latestVelocity + dt * un_acc;

  latestImuData = data;
  imu_odom.timestamp = data.timestamp;
  imu_odom.position = lastPosition;
  imu_odom.velocity = latestVelocity;
  imu_odom.orientation = latest_Q;
  safe_imu_pre_odom.set(imu_odom);
}

void Estimator::updateLatestStates() {
  latestPosition = positions[frameCount];
  latest_Q = rotations[frameCount];
  latestVelocity = velocities[frameCount];
  latestAccelBias = accelerometerBiases[frameCount];
  latestGyroBias = gyroscopeBiases[frameCount];
  latestImuData = previousImuData;
  latestImuData.timestamp = Headers[frameCount] + timeDelay;

  queue<IMUData> tmp_imu;
  {
    std::lock_guard<std::mutex> imu_lock(imu_mutex);
    tmp_imu = imuBuffer;
  }
  while (!tmp_imu.empty()) {
    fastPredictIMU(tmp_imu.front());
    tmp_imu.pop();
  }
}
