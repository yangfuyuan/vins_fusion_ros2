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

Estimator::Estimator() {}

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
  inputImageCount = 0;
  isFirstPoseInitialized = false;

  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    estimator_state[i].clear();
    std::vector<IMUData>().swap(deltaTimeImuBuffer[i]);
    pre_integrations[i] = nullptr;
  }

  for (int i = 0; i < options->getNumCameras(); i++) {
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

void Estimator::initialize(std::shared_ptr<VINSOptions> options_) {
  std::lock_guard<std::mutex> lock(processingMutex);
  options = options_;
  initializeCamerasFromOptions();
  isRunning.store(true);
  processThread = std::thread(&Estimator::processMeasurements, this);
}

void Estimator::initializeCamerasFromOptions() {
  for (int i = 0; i < options->getNumCameras(); i++) {
    cameraTranslation[i] = options->TIC[i];
    cameraRotation[i] = options->RIC[i];
    updateCameraPose(i);
    VINS_INFO << " exitrinsic cam[" << i << "]: \n"
              << cameraRotation[i] << "\n"
              << cameraTranslation[i].transpose();
  }
  featureManager.setOptions(options);
  featureTracker.setOptions(options);
  ProjectionTwoFrameOneCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionTwoFrameTwoCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionOneFrameTwoCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  gravity = options->imu.gravity();
  featureTracker.readIntrinsicParameter(options->camera_names);
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
  if (options->shouldShowTrack()) {
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
    currentTimestamp = feature.first + options->time_delay;
    if (options->hasImu()) {
      std::unique_lock<std::mutex> lock(imu_mutex);

      imuCondition.wait(lock, [this] {
        return IMUAvailable(currentTimestamp) || !isRunning.load();
      });
      if (!isRunning.load()) {
        break;
      }
    }

    if (options->hasImu()) {
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
        estimator_state[imu_i].rotation *
            (cameraRotation[0] * pts_i + cameraTranslation[0]) +
        estimator_state[imu_i].position;

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
    pose.timestamp = estimator_state[i].timestamp;
    pose.position = estimator_state[i].position;
    pose.orientation = Quaterniond(estimator_state[i].rotation);
    safe_keyframe_pose.set(pose);
  }
}

void Estimator::printStatistics(Timestamp timestamp) {
  if (solver_flag != SolverState::NON_LINEAR) return;
  VINS_DEBUG << "position: (" << estimator_state[WINDOW_SIZE].position.x()
             << "," << estimator_state[WINDOW_SIZE].position.y() << ","
             << estimator_state[WINDOW_SIZE].position.z() << ")";
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
  estimator_state[0].rotation = initialRotation;
  VINS_INFO << "init R0: \n" << estimator_state[0].rotation;
}

void Estimator::setFirstPose(const Eigen::Vector3d &position,
                             const Eigen::Matrix3d &rotation) {
  estimator_state[0].position = position;
  estimator_state[0].rotation = rotation;
}

void Estimator::processIMU(const IMUData &data, double deltaTime) {
  if (!isFirstIMUReceived) {
    isFirstIMUReceived = true;
    previousImuData = data;
  }

  if (!pre_integrations[frameCount]) {
    pre_integrations[frameCount] = std::make_shared<IntegrationBase>(
        previousImuData, estimator_state[frameCount].accel_bias,
        estimator_state[frameCount].gyro_bias, options->imu);
  }
  if (frameCount != 0) {
    IMUData delta_imu = data;
    delta_imu.timestamp = deltaTime;

    pre_integrations[frameCount]->push_back(delta_imu);
    tmp_pre_integration->push_back(delta_imu);
    deltaTimeImuBuffer[frameCount].push_back(delta_imu);

    updateStateWithIMU(data, deltaTime);
  }
  previousImuData = data;
}

void Estimator::updateStateWithIMU(const IMUData &data, double deltaTime) {
  int j = frameCount;
  Vector3d un_acc_0 =
      estimator_state[j].rotation * (previousImuData.linear_acceleration -
                                     estimator_state[j].accel_bias) -
      gravity;
  Vector3d un_gyr =
      0.5 * (previousImuData.angular_velocity + data.angular_velocity) -
      estimator_state[j].gyro_bias;
  estimator_state[j].rotation *=
      Utility::deltaQ(un_gyr * deltaTime).toRotationMatrix();
  Vector3d un_acc_1 =
      estimator_state[j].rotation *
          (data.linear_acceleration - estimator_state[j].accel_bias) -
      gravity;
  Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  estimator_state[j].position += deltaTime * estimator_state[j].velocity +
                                 0.5 * deltaTime * deltaTime * un_acc;
  estimator_state[j].velocity += deltaTime * un_acc;
}

void Estimator::processImage(const FeatureFrame &features,
                             Timestamp timestamp) {
  setMarginalizationFlag(features);
  insertImageFrame(features, timestamp);
  handleExtrinsicInitialization();
  if (!isNonLinearSolver()) {
    processInitialization(timestamp);
  } else {
    processNonLinearSolver(timestamp);
  }
}

void Estimator::setMarginalizationFlag(const FeatureFrame &features) {
  if (featureManager.addFeatureCheckParallax(frameCount, features,
                                             options->time_delay)) {
    marginalization_flag = MarginalizationType::MARGIN_OLD;
  } else {
    marginalization_flag = MarginalizationType::MARGIN_SECOND_NEW;
  }
}
void Estimator::insertImageFrame(const FeatureFrame &features,
                                 Timestamp timestamp) {
  estimator_state[frameCount].timestamp = timestamp;
  ImageFrame imageframe(features, timestamp);
  imageframe.pre_integration = std::move(tmp_pre_integration);
  all_image_frame.insert(make_pair(timestamp, imageframe));
  tmp_pre_integration = std::make_shared<IntegrationBase>(
      previousImuData, estimator_state[frameCount].accel_bias,
      estimator_state[frameCount].gyro_bias, options->imu);
}

void Estimator::handleExtrinsicInitialization() {
  if (!options->isInitializingExtrinsic() || frameCount == 0) {
    return;
  }
  vector<pair<Vector3d, Vector3d>> corres =
      featureManager.getCorresponding(frameCount - 1, frameCount);
  Matrix3d calib_ric;
  if (initial_ex_rotation.CalibrationExRotation(
          corres, pre_integrations[frameCount]->delta_q, calib_ric)) {
    cameraRotation[0] = calib_ric;
    options->RIC[0] = calib_ric;
    options->extrinsic_estimation_mode = ExtrinsicEstimationMode::APPROXIMATE;
  }
}

void Estimator::processInitialization(Timestamp timestamp) {
  if (options->isMonoWithImu()) {
    processMonoWithImuInitialization(timestamp);
  }
  if (options->isStereoWithImu()) {
    processStereoWithImuInitialization();
  }
  if (options->isStereoWithoutImu()) {
    processStereoWithoutImuInitialization();
  }

  if (frameCount < WINDOW_SIZE) {
    frameCount++;
    estimator_state[frameCount] = estimator_state[frameCount - 1];
  }
}
void Estimator::processNonLinearSolver(Timestamp timestamp) {
  if (!options->hasImu()) {
    featureManager.initFramePoseByPnP(frameCount, estimator_state,
                                      cameraTranslation, cameraRotation);
  }
  featureManager.triangulate(frameCount, estimator_state, cameraTranslation,
                             cameraRotation);

  // optimization
  TicToc t_solve;
  optimize();
  set<int> removeIndex;
  outliersRejection(removeIndex);
  featureManager.removeOutlier(removeIndex);
  if (failureDetection()) {
    failure_occur = 1;
    resetState();
    initializeCamerasFromOptions();
    return;
  }

  slideWindow();
  featureManager.removeFailures();
  // prepare output of VINS
  {
    key_poses.timestamp = timestamp;
    key_poses.poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
      key_poses.poses.push_back(estimator_state[i].position);

    safe_key_poses.set(key_poses);
  }

  last_state = estimator_state[WINDOW_SIZE];
  last_state0 = estimator_state[0];
  updateLatestStates();

  vio_odom.timestamp = timestamp;
  vio_odom.position = estimator_state[WINDOW_SIZE].position;
  vio_odom.orientation = Quaterniond(estimator_state[WINDOW_SIZE].rotation);
  vio_odom.velocity = estimator_state[WINDOW_SIZE].velocity;
  safe_vio_odom.set(vio_odom);
}

void Estimator::processMonoWithImuInitialization(Timestamp timestamp) {
  if (frameCount != WINDOW_SIZE) return;

  bool result = false;
  if (!options->isInitializingExtrinsic() &&
      (timestamp - initialTimestamp) > 0.1) {
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

void Estimator::processStereoWithImuInitialization() {
  featureManager.initFramePoseByPnP(frameCount, estimator_state,
                                    cameraTranslation, cameraRotation);
  featureManager.triangulate(frameCount, estimator_state, cameraTranslation,
                             cameraRotation);

  if (frameCount != WINDOW_SIZE) return;

  map<double, ImageFrame>::iterator frame_it;
  int i = 0;
  for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end();
       frame_it++) {
    frame_it->second.R = estimator_state[i].rotation;
    frame_it->second.T = estimator_state[i].position;
    i++;
  }
  solveGyroscopeBias(all_image_frame, estimator_state);
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    pre_integrations[i]->repropagate(Vector3d::Zero(),
                                     estimator_state[i].gyro_bias);
  }
  optimize();
  updateLatestStates();
  solver_flag = SolverState::NON_LINEAR;
  slideWindow();
  VINS_INFO << "Initialization complete. Switching to NON_LINEAR mode.";
}

void Estimator::processStereoWithoutImuInitialization() {
  featureManager.initFramePoseByPnP(frameCount, estimator_state,
                                    cameraTranslation, cameraRotation);
  featureManager.triangulate(frameCount, estimator_state, cameraTranslation,
                             cameraRotation);
  optimize();

  if (frameCount != WINDOW_SIZE) return;

  optimize();
  updateLatestStates();
  solver_flag = SolverState::NON_LINEAR;
  slideWindow();
  VINS_INFO << "Initialization complete. Switching to NON_LINEAR mode.";
}

bool Estimator::isNonLinearSolver() const {
  return solver_flag == SolverState::NON_LINEAR;
}

bool Estimator::isNewMarginalization() const {
  return marginalization_flag == MarginalizationType::MARGIN_SECOND_NEW;
}

bool Estimator::checkIMUExcitation() {
  Vector3d sumG = Vector3d::Zero();
  auto it = std::next(all_image_frame.begin());
  for (; it != all_image_frame.end(); ++it) {
    double dt = it->second.pre_integration->sum_dt;
    sumG += it->second.pre_integration->delta_v / dt;
  }

  Vector3d avgG = sumG / (all_image_frame.size() - 1);
  double variance = 0;
  for (it = std::next(all_image_frame.begin()); it != all_image_frame.end();
       ++it) {
    double dt = it->second.pre_integration->sum_dt;
    Vector3d tmpG = it->second.pre_integration->delta_v / dt;
    variance += (tmpG - avgG).squaredNorm();
  }

  variance = std::sqrt(variance / (all_image_frame.size() - 1));
  return variance >= 0.25;
}

std::vector<SFMFeature> Estimator::buildSFMFeatures() {
  std::vector<SFMFeature> sfm_features;

  for (auto &feature : featureManager.feature) {
    SFMFeature sf;
    sf.state = false;
    sf.id = feature.feature_id;
    int frame_id = feature.start_frame - 1;
    for (auto &f : feature.feature_per_frame) {
      ++frame_id;
      sf.observation.emplace_back(frame_id, Vector2d(f.point.x(), f.point.y()));
    }
    sfm_features.emplace_back(std::move(sf));
  }

  return sfm_features;
}

bool Estimator::solvePoseWithPnP(const std::vector<Quaterniond> &rotations,
                                 const std::vector<Vector3d> &positions,
                                 const std::map<int, Vector3d> &sfmPoints) {
  auto frame_it = all_image_frame.begin();
  int i = 0;

  while (frame_it != all_image_frame.end()) {
    if (frame_it->first == estimator_state[i].timestamp) {
      frame_it->second.is_key_frame = true;
      frame_it->second.R =
          rotations[i].toRotationMatrix() * options->RIC[0].transpose();
      frame_it->second.T = positions[i];
      ++i;
      ++frame_it;
      continue;
    }

    if (frame_it->first > estimator_state[i].timestamp) {
      ++i;
    }
    cv::Mat tmp_r;
    cv::Mat r, rvec, tvec, D;
    Matrix3d R_inital = (rotations[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * positions[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, tvec);

    frame_it->second.is_key_frame = false;
    vector<cv::Point3f> pts3;
    vector<cv::Point2f> pts2;

    for (const auto &pt : frame_it->second.points) {
      auto it = sfmPoints.find(pt.first);
      if (it == sfmPoints.end()) continue;

      for (const auto &obs : pt.second) {
        Vector3d p3d = it->second;
        Vector2d p2d = obs.second.head<2>();
        pts3.emplace_back(p3d.x(), p3d.y(), p3d.z());
        pts2.emplace_back(p2d.x(), p2d.y());
      }
    }

    if (pts3.size() < 6) {
      VINS_WARN << "Not enough points for solvePnP: " << pts3.size();
      return false;
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

    if (!cv::solvePnP(pts3, pts2, K, D, rvec, tvec, true)) {
      VINS_WARN << "Failed to solvePnP";
      return false;
    }

    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp, tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(tvec, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * options->RIC[0].transpose();
    frame_it->second.T = T_pnp;
    ++frame_it;
  }

  return true;
}

bool Estimator::initialStructure() {
  if (!checkIMUExcitation()) {
    VINS_WARN << "IMU excitation insufficient";
  }

  auto sfm_features = buildSFMFeatures();

  Matrix3d relativeRotation;
  Vector3d relativeTranslation;
  int referenceFrame;
  if (!computeRelativePose(relativeRotation, relativeTranslation,
                           referenceFrame)) {
    VINS_WARN << "Not enough features or parallax; Move device around!!!";
    return false;
  }

  std::vector<Quaterniond> rotations(WINDOW_SIZE + 1);
  std::vector<Vector3d> positions(WINDOW_SIZE + 1);
  std::map<int, Vector3d> sfmPoints;
  GlobalSFM sfm;
  if (!sfm.construct(frameCount + 1, rotations, positions, referenceFrame,
                     relativeRotation, relativeTranslation, sfm_features,
                     sfmPoints)) {
    marginalization_flag = MarginalizationType::MARGIN_OLD;
    VINS_WARN << "Failed to construect sfm!!!";
    return false;
  }

  if (!solvePoseWithPnP(rotations, positions, sfmPoints)) {
    return false;
  }

  return visualInitialAlign();
}

bool Estimator::visualInitialAlign() {
  TicToc t_g;
  VectorXd x;
  // solve scale
  bool result = VisualIMUAlignment(all_image_frame, estimator_state, gravity, x,
                                   *options);
  if (!result) {
    VINS_DEBUG << "misalign visual structure with IMU";
    return false;
  }

  // change state
  for (int i = 0; i <= frameCount; i++) {
    Matrix3d Ri = all_image_frame[estimator_state[i].timestamp].R;
    Vector3d Pi = all_image_frame[estimator_state[i].timestamp].T;
    estimator_state[i].position = Pi;
    estimator_state[i].rotation = Ri;
    all_image_frame[estimator_state[i].timestamp].is_key_frame = true;
  }

  double s = (x.tail<1>())(0);
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    pre_integrations[i]->repropagate(Vector3d::Zero(),
                                     estimator_state[i].gyro_bias);
  }
  for (int i = frameCount; i >= 0; i--)
    estimator_state[i].position =
        s * estimator_state[i].position -
        estimator_state[i].rotation * options->TIC[0] -
        (s * estimator_state[0].position -
         estimator_state[0].rotation * options->TIC[0]);
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end();
       frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      estimator_state[kv].velocity = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }

  Matrix3d R0 = Utility::g2R(gravity);
  double yaw = Utility::R2ypr(R0 * estimator_state[0].rotation).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  gravity = R0 * gravity;
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frameCount; i++) {
    estimator_state[i].position = rot_diff * estimator_state[i].position;
    estimator_state[i].rotation = rot_diff * estimator_state[i].rotation;
    estimator_state[i].velocity = rot_diff * estimator_state[i].velocity;
  }
  featureManager.clearDepth();
  featureManager.triangulate(frameCount, estimator_state, cameraTranslation,
                             cameraRotation);

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
    estimator_state[i].toPoseArray(poseArray[i]);
    estimator_state[i].toSpeedBiasArray(speedBiasArray[i]);
  }

  for (int i = 0; i < options->getNumCameras(); i++) {
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

  para_Td[0][0] = options->time_delay;
}

void Estimator::updateEstimates() {
  Vector3d origin_R0 = Utility::R2ypr(estimator_state[0].rotation);
  Vector3d origin_P0 = estimator_state[0].position;

  if (failure_occur) {
    origin_R0 = Utility::R2ypr(last_state0.rotation);
    origin_P0 = last_state0.position;
    failure_occur = 0;
  }

  if (options->hasImu()) {
    const auto &pose0Vector = Utility::toPositionVector(poseArray[0]);
    const auto &pose0Matrix =
        Utility::toQuaternion(poseArray[0]).toRotationMatrix();

    Vector3d origin_R00 = Utility::R2ypr(pose0Matrix);
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 ||
        abs(abs(origin_R00.y()) - 90) < 1.0) {
      rot_diff = estimator_state[0].rotation * pose0Matrix.transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {
      estimator_state[i].rotation =
          rot_diff * Utility::toQuaternion(poseArray[i]).toRotationMatrix();

      estimator_state[i].position =
          rot_diff * (Utility::toPositionVector(poseArray[i]) - pose0Vector) +
          origin_P0;
      estimator_state[i].velocity =
          rot_diff * Utility::toPositionVector(speedBiasArray[i]);
      estimator_state[i].accel_bias =
          Utility::toPositionVector(speedBiasArray[i] + 3);
      estimator_state[i].gyro_bias =
          Utility::toPositionVector(speedBiasArray[i] + 6);
    }
  } else {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
      estimator_state[i].rotation =
          Utility::toQuaternion(poseArray[i]).toRotationMatrix();
      estimator_state[i].position = Utility::toPositionVector(poseArray[i]);
    }
  }

  if (options->hasImu()) {
    for (int i = 0; i < options->getNumCameras(); i++) {
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

  if (options->hasImu()) options->time_delay = para_Td[0][0];
}

bool Estimator::failureDetection() {
  return false;
  if (featureManager.last_track_num < 2) {
  }
  if (estimator_state[WINDOW_SIZE].accel_bias.norm() > 2.5) {
    return true;
  }
  if (estimator_state[WINDOW_SIZE].gyro_bias.norm() > 1.0) {
    return true;
  }
  Vector3d tmp_P = estimator_state[WINDOW_SIZE].position;
  if ((tmp_P - last_state.position).norm() > 5) {
    // return true;
  }
  if (abs(tmp_P.z() - last_state.position.z()) > 1) {
    // return true;
  }
  Matrix3d tmp_R = estimator_state[WINDOW_SIZE].rotation;
  Matrix3d delta_R = tmp_R.transpose() * last_state.rotation;
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
    problem.AddParameterBlock(poseArray[i], SIZE_POSE, local_param);
    if (options->hasImu()) {
      problem.AddParameterBlock(speedBiasArray[i], SIZE_SPEEDBIAS);
    }
  }
  if (!options->hasImu()) {
    problem.SetParameterBlockConstant(poseArray[0]);
  }
}
void Estimator::AddExtrinsicParameterBlocks(ceres::Problem &problem) {
  for (int i = 0; i < options->getNumCameras(); i++) {
    auto *local_param = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_param);
    if (((options->isExtrinsicEstimationApproximate() ||
          options->isInitializingExtrinsic()) &&
         frameCount == WINDOW_SIZE &&
         estimator_state[0].velocity.norm() > 0.2) ||
        openExEstimation) {
      openExEstimation = 1;
    } else {
      problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
  }
}
void Estimator::AddTimeDelayParameterBlock(ceres::Problem &problem) {
  problem.AddParameterBlock(para_Td[0], 1);
  if (!options->shouldEstimateTD() || estimator_state[0].velocity.norm() < 0.2)
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
  if (!options->hasImu()) return;

  for (int i = 0; i < frameCount; i++) {
    int j = i + 1;
    if (pre_integrations[j]->sum_dt > 10.0) continue;
    auto *imu_factor = new IMUFactor(pre_integrations[j]);
    problem.AddResidualBlock(imu_factor, nullptr, poseArray[i],
                             speedBiasArray[i], poseArray[j],
                             speedBiasArray[j]);
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
                                 poseArray[imu_i], poseArray[imu_j],
                                 para_Ex_Pose[0], para_Feature[feature_index],
                                 para_Td[0]);
      }

      if (options->isUsingStereo() && f.is_stereo) {
        Vector3d pts_j_right = f.pointRight;
        if (imu_i != imu_j) {
          auto *factor = new ProjectionTwoFrameTwoCamFactor(
              pts_i, pts_j_right, it.feature_per_frame[0].velocity,
              f.velocityRight, it.feature_per_frame[0].cur_td, f.cur_td);
          problem.AddResidualBlock(factor, new ceres::HuberLoss(1.0),
                                   poseArray[imu_i], poseArray[imu_j],
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

  ceres::Solver::Options ceres_options;
  ceres_options.linear_solver_type =
      options->USE_GPU_CERES ? ceres::DENSE_QR : ceres::DENSE_SCHUR;
  ceres_options.trust_region_strategy_type = ceres::DOGLEG;
  ceres_options.max_num_iterations = options->max_num_iterations();
  ceres_options.max_solver_time_in_seconds =
      (!isNewMarginalization()) ? options->max_solver_time() * 0.8
                                : options->max_solver_time();

  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options, &problem, &summary);
}

void Estimator::processOldMarginalization() {
  auto marginalization_info = std::make_shared<MarginalizationInfo>();
  prepareParameters();

  if (last_marginalization_info && last_marginalization_info->valid) {
    vector<int> drop_set;
    for (int i = 0;
         i < static_cast<int>(last_marginalization_parameter_blocks.size());
         i++) {
      if (last_marginalization_parameter_blocks[i] == poseArray[0] ||
          last_marginalization_parameter_blocks[i] == speedBiasArray[0]) {
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
  if (options->hasImu() && pre_integrations[1]->sum_dt < 10.0) {
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
                 poseArray[WINDOW_SIZE - 1])) {
    auto marginalization_info = std::make_shared<MarginalizationInfo>();
    prepareParameters();

    if (last_marginalization_info->valid) {
      vector<int> drop_set;
      for (int i = 0;
           i < static_cast<int>(last_marginalization_parameter_blocks.size());
           i++) {
        if (last_marginalization_parameter_blocks[i] ==
            poseArray[WINDOW_SIZE - 1]) {
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
      std::vector<double *>{poseArray[0], speedBiasArray[0], poseArray[1],
                            speedBiasArray[1]},
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
            vector<double *>{poseArray[imu_i], poseArray[imu_j],
                             para_Ex_Pose[0], para_Feature[feature_index],
                             para_Td[0]},
            vector<int>{0, 3}));
      }

      if (options->isUsingStereo() && it_per_frame.is_stereo) {
        const Vector3d &pts_j_right = it_per_frame.pointRight;
        auto loss = std::make_shared<ceres::HuberLoss>(1.0);

        if (imu_i != imu_j) {
          auto f = std::make_shared<ProjectionTwoFrameTwoCamFactor>(
              pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
              it_per_frame.velocityRight, it_per_id.feature_per_frame[0].cur_td,
              it_per_frame.cur_td);
          marg_info->addResidualBlockInfo(std::make_shared<ResidualBlockInfo>(
              f, loss,
              vector<double *>{poseArray[imu_i], poseArray[imu_j],
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
      addr_shift[reinterpret_cast<long>(poseArray[i])] = poseArray[i - 1];
      if (options->hasImu())
        addr_shift[reinterpret_cast<long>(speedBiasArray[i])] =
            speedBiasArray[i - 1];
    } else {
      addr_shift[reinterpret_cast<long>(poseArray[i])] =
          poseArray[i - (is_old ? 1 : 0)];
      if (options->hasImu())
        addr_shift[reinterpret_cast<long>(speedBiasArray[i])] =
            speedBiasArray[i - (is_old ? 1 : 0)];
    }
  }

  for (int i = 0; i < options->getNumCameras(); i++)
    addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

  addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

  return addr_shift;
}

void Estimator::optimize() {
  prepareParameters();
  solveOptimization();
  updateEstimates();
  if (frameCount < WINDOW_SIZE) return;
  if (isNewMarginalization()) {
    processNewMarginalization();
  } else {
    processOldMarginalization();
  }
}

void Estimator::slideWindow() {
  if (!isNewMarginalization()) {
    double t_0 = estimator_state[0].timestamp;
    back_state = estimator_state[0];
    slideWindowOld();
  } else {
    slideWindowNew();
  }
}

void Estimator::slideWindowNew() {
  if (frameCount != WINDOW_SIZE) {
    return;
  }
  estimator_state[frameCount - 1] = estimator_state[frameCount];
  if (options->hasImu()) {
    for (unsigned int i = 0; i < deltaTimeImuBuffer[frameCount].size(); i++) {
      const auto &imu = deltaTimeImuBuffer[frameCount][i];
      pre_integrations[frameCount - 1]->push_back(imu);
      deltaTimeImuBuffer[frameCount - 1].push_back(imu);
    }
    pre_integrations[WINDOW_SIZE] = nullptr;
    pre_integrations[WINDOW_SIZE] = std::make_shared<IntegrationBase>(
        previousImuData, estimator_state[WINDOW_SIZE].accel_bias,
        estimator_state[WINDOW_SIZE].gyro_bias, options->imu);

    deltaTimeImuBuffer[WINDOW_SIZE].clear();
  }
  frontCount++;
  featureManager.removeFront(frameCount);
}

void Estimator::slideWindowOld() {
  if (frameCount != WINDOW_SIZE) {
    return;
  }
  for (int i = 0; i < WINDOW_SIZE; i++) {
    estimator_state[i].timestamp = estimator_state[i + 1].timestamp;
    estimator_state[i].swap(estimator_state[i + 1]);
    if (options->hasImu()) {
      std::swap(pre_integrations[i], pre_integrations[i + 1]);
      deltaTimeImuBuffer[i].swap(deltaTimeImuBuffer[i + 1]);
    }
  }
  estimator_state[WINDOW_SIZE] = estimator_state[WINDOW_SIZE - 1];
  if (options->hasImu()) {
    pre_integrations[WINDOW_SIZE] = nullptr;
    pre_integrations[WINDOW_SIZE] = std::make_shared<IntegrationBase>(
        previousImuData, estimator_state[WINDOW_SIZE].accel_bias,
        estimator_state[WINDOW_SIZE].gyro_bias, options->imu);
    deltaTimeImuBuffer[WINDOW_SIZE].clear();
  }
  auto it_0 = all_image_frame.find(estimator_state[0].timestamp);
  all_image_frame.erase(all_image_frame.begin(), it_0);

  backCount++;
  if (isNonLinearSolver()) {
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_state.rotation * cameraRotation[0];
    R1 = estimator_state[0].rotation * cameraRotation[0];
    P0 = back_state.position + back_state.rotation * cameraTranslation[0];
    P1 = estimator_state[0].position +
         estimator_state[0].rotation * cameraTranslation[0];
    featureManager.removeBackShiftDepth(R0, P0, R1, P1);
  } else {
    featureManager.removeBack();
  }
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = estimator_state[frameCount].rotation;
  T.block<3, 1>(0, 3) = estimator_state[frameCount].position;
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = estimator_state[index].rotation;
  T.block<3, 1>(0, 3) = estimator_state[index].position;
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
        Vector3d pts_w = estimator_state[firstIndex].rotation * pts_j +
                         estimator_state[firstIndex].position;
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
            estimator_state[imu_i].rotation, estimator_state[imu_i].position,
            cameraRotation[0], cameraTranslation[0],
            estimator_state[imu_j].rotation, estimator_state[imu_j].position,
            cameraRotation[0], cameraTranslation[0], depth, pts_i, pts_j);
        err += tmp_error;
        errCnt++;
      }
      // need to rewrite projecton factor.........
      if (options->isUsingStereo() && it_per_frame.is_stereo) {
        Vector3d pts_j_right = it_per_frame.pointRight;
        if (imu_i != imu_j) {
          double tmp_error = reprojectionError(
              estimator_state[imu_i].rotation, estimator_state[imu_i].position,
              cameraRotation[0], cameraTranslation[0],
              estimator_state[imu_j].rotation, estimator_state[imu_j].position,
              cameraRotation[1], cameraTranslation[1], depth, pts_i,
              pts_j_right);
          err += tmp_error;
          errCnt++;
        } else {
          double tmp_error = reprojectionError(
              estimator_state[imu_i].rotation, estimator_state[imu_i].position,
              cameraRotation[0], cameraTranslation[0],
              estimator_state[imu_j].rotation, estimator_state[imu_j].position,
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
  if (dt > 1.0) {
    VINS_ERROR << "Abnormal IMU timestamp jump detected: " << dt;
    return;
  }
  Eigen::Vector3d un_acc_0 =
      latest_Q * (latestImuData.linear_acceleration - latest_state.accel_bias) -
      gravity;
  Eigen::Vector3d un_gyr =
      0.5 * (latestImuData.angular_velocity + data.angular_velocity) -
      latest_state.gyro_bias;
  latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
  Eigen::Vector3d un_acc_1 =
      latest_Q * (data.linear_acceleration - latest_state.accel_bias) - gravity;
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  latest_state.position = latest_state.position + dt * latest_state.velocity +
                          0.5 * dt * dt * un_acc;
  latest_state.velocity = latest_state.velocity + dt * un_acc;

  latestImuData = data;
  imu_odom.timestamp = data.timestamp;
  imu_odom.position = latest_state.position;
  imu_odom.velocity = latest_state.velocity;
  imu_odom.orientation = latest_Q;
  safe_imu_pre_odom.set(imu_odom);
}

void Estimator::updateLatestStates() {
  latest_state = estimator_state[frameCount];
  latest_Q = latest_state.rotation;
  latestImuData = previousImuData;
  latestImuData.timestamp =
      estimator_state[frameCount].timestamp + options->time_delay;

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
