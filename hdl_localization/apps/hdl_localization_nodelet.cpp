// hdl localizaton ROS2 - Multi-rate sensor fusion architecture
// IMU (100-400Hz) -> UKF prediction
// LiDAR Fast GICP (10-20Hz) -> frame-to-frame odometry -> UKF correction
// NDT (1-2Hz) -> global map alignment -> UKF correction (drift reset)

#include <mutex>
#include <memory>
#include <iostream>
#include <thread>
#include <atomic>
#include <condition_variable>

#include <rclcpp/rclcpp.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <std_srvs/srv/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_gicp.hpp>

// Fast GICP for frame-to-frame odometry (already included above)

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/msg/scan_matching_status.hpp>
#include <hdl_global_localization/srv/set_global_map.hpp>
#include <hdl_global_localization/srv/query_global_localization.hpp>

using namespace std;

namespace hdl_localization {

class HdlLocalizationNodelet : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZ;

  HdlLocalizationNodelet(const rclcpp::NodeOptions& options) : Node("hdl_localization", options) {
    tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    robot_odom_frame_id = declare_parameter<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id = declare_parameter<std::string>("odom_child_frame_id", "base_link");
    send_tf_transforms = declare_parameter<bool>("send_tf_transforms", true);
    cool_time_duration = declare_parameter<double>("cool_time_duration", 0.5);
    reg_method = declare_parameter<std::string>("reg_method", "NDT_OMP");
    ndt_neighbor_search_method = declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");
    ndt_neighbor_search_radius = declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
    ndt_resolution = declare_parameter<double>("ndt_resolution", 0.5);
    enable_robot_odometry_prediction = declare_parameter<bool>("enable_robot_odometry_prediction", false);

    use_imu = declare_parameter<bool>("use_imu", true);
    invert_acc = declare_parameter<bool>("invert_acc", false);
    invert_gyro = declare_parameter<bool>("invert_gyro", false);

    // IMU noise covariance parameters
    acc_cov = declare_parameter<double>("acc_cov", 0.5);
    gyr_cov = declare_parameter<double>("gyr_cov", 0.3);
    b_acc_cov = declare_parameter<double>("b_acc_cov", 0.0001);
    b_gyr_cov = declare_parameter<double>("b_gyr_cov", 0.0001);

    // NDT rate control (Hz) - default 0.2Hz (every 5 seconds)
    ndt_rate = declare_parameter<double>("ndt_rate", 0.2);

    // Local map parameters for efficient NDT matching
    use_local_map = declare_parameter<bool>("use_local_map", true);
    local_map_radius = declare_parameter<double>("local_map_radius", 50.0);
    RCLCPP_INFO(get_logger(), "Local map: %s, radius: %.1fm",
      use_local_map ? "enabled" : "disabled", local_map_radius);

    // Topic names (configurable via parameters)
    std::string points_topic = declare_parameter<std::string>("points_topic", "/velodyne_points");
    std::string imu_topic = declare_parameter<std::string>("imu_topic", "/gpsimu_driver/imu_data");

    if (use_imu) {
      RCLCPP_INFO(get_logger(), "enable imu-based prediction, subscribing to: %s", imu_topic.c_str());
      imu_sub = create_subscription<sensor_msgs::msg::Imu>(imu_topic, 256, std::bind(&HdlLocalizationNodelet::imu_callback, this, std::placeholders::_1));
    }
    points_sub = create_subscription<sensor_msgs::msg::PointCloud2>(points_topic, 5, std::bind(&HdlLocalizationNodelet::points_callback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "subscribing to points topic: %s", points_topic.c_str());

    auto latch_qos = 10;
    globalmap_sub =
      create_subscription<sensor_msgs::msg::PointCloud2>("/globalmap", latch_qos, std::bind(&HdlLocalizationNodelet::globalmap_callback, this, std::placeholders::_1));

    initialpose_sub =
      create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 8, std::bind(&HdlLocalizationNodelet::initialpose_callback, this, std::placeholders::_1));

    pose_pub = create_publisher<nav_msgs::msg::Odometry>("/odom", 5);
    aligned_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_points", 5);
    status_pub = create_publisher<msg::ScanMatchingStatus>("/status", 5);

    // global localization
    use_global_localization = declare_parameter<bool>("use_global_localization", true);
    if (use_global_localization) {
      RCLCPP_INFO_STREAM(get_logger(), "wait for global localization services");
      set_global_map_service = create_client<hdl_global_localization::srv::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = create_client<hdl_global_localization::srv::QueryGlobalLocalization>("/hdl_global_localization/query");
      while (!set_global_map_service->wait_for_service(std::chrono::milliseconds(1000))) {
        RCLCPP_WARN(get_logger(), "Waiting for SetGlobalMap service");
        if (!rclcpp::ok()) {
          return;
        }
      }
      while (!query_global_localization_service->wait_for_service(std::chrono::milliseconds(1000))) {
        RCLCPP_WARN(get_logger(), "Waiting for QueryGlobalLocalization service");
        if (!rclcpp::ok()) {
          return;
        }
      }

      relocalize_server = create_service<std_srvs::srv::Empty>("/relocalize", std::bind(&HdlLocalizationNodelet::relocalize, this, std::placeholders::_1, std::placeholders::_2));
    }

    // Initialize NDT thread control
    ndt_thread_running = true;
    ndt_has_new_scan = false;

    initialize_params();

    // Start NDT thread (runs at ndt_rate Hz)
    ndt_thread = std::thread(&HdlLocalizationNodelet::ndt_thread_func, this);
    RCLCPP_INFO(get_logger(), "NDT thread started at %.1f Hz", ndt_rate);
  }

  ~HdlLocalizationNodelet() {
    // Stop NDT thread
    ndt_thread_running = false;
    ndt_cv.notify_all();
    if (ndt_thread.joinable()) {
      ndt_thread.join();
    }
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() {
    if (reg_method == "NDT_OMP") {
      RCLCPP_INFO(get_logger(), "NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setMaximumIterations(30);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          RCLCPP_INFO(get_logger(), "search_method KDTREE is selected");
        } else {
          RCLCPP_WARN(get_logger(), "invalid search method was given");
          RCLCPP_WARN(get_logger(), "default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if (reg_method.find("NDT_CUDA") != std::string::npos) {
      RCLCPP_INFO(get_logger(), "NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if (reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        RCLCPP_INFO_STREAM(get_logger(), "search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        RCLCPP_WARN(get_logger(), "invalid search method was given");
      }
      std::shared_ptr<pcl::Registration<PointT, PointT>>(ndt.get(), [](pcl::Registration<PointT, PointT>*) {});
    }

    RCLCPP_ERROR_STREAM(get_logger(), "unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_frame_to_frame_gicp() {
    // Fast GICP parameters for frame-to-frame odometry
    // For consecutive scans at 20Hz, displacement is small (~5cm at 1m/s)
    // So correspondence distance should be small
    int gicp_num_threads = declare_parameter<int>("gicp_num_threads", 4);
    int gicp_max_iterations = declare_parameter<int>("gicp_max_iterations", 32);
    double gicp_correspondence_distance = declare_parameter<double>("gicp_correspondence_distance", 0.5);

    // Initialize Fast GICP
    auto gicp = pcl::make_shared<fast_gicp::FastGICP<PointT, PointT>>();
    gicp->setNumThreads(gicp_num_threads);
    gicp->setMaximumIterations(gicp_max_iterations);
    gicp->setMaxCorrespondenceDistance(gicp_correspondence_distance);
    gicp->setTransformationEpsilon(1e-4);  // Looser epsilon for faster convergence
    frame_to_frame_gicp = gicp;

    RCLCPP_INFO(get_logger(), "Fast GICP params - threads: %d, max_iter: %d, corr_dist: %.2f",
      gicp_num_threads, gicp_max_iterations, gicp_correspondence_distance);
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = declare_parameter<double>("downsample_resolution", 0.1);
    std::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid = std::make_shared<pcl::VoxelGrid<PointT>>();
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    RCLCPP_INFO(get_logger(), "create NDT registration for global map alignment");
    registration = create_registration();

    RCLCPP_INFO(get_logger(), "create Fast GICP registration for frame-to-frame odometry");
    initialize_frame_to_frame_gicp();

    // global localization
    RCLCPP_INFO(get_logger(), "create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // initialize pose estimator
    bool specify_init_pose = declare_parameter<bool>("specify_init_pose", true);
    if (specify_init_pose) {
      RCLCPP_INFO(get_logger(), "initialize pose estimator with specified parameters!!");
      RCLCPP_INFO(get_logger(), "IMU covariance - acc: %.4f, gyr: %.4f, b_acc: %.6f, b_gyr: %.6f", acc_cov, gyr_cov, b_acc_cov, b_gyr_cov);

      Eigen::Vector3f init_pos(
        declare_parameter<double>("init_pos_x", 0.0),
        declare_parameter<double>("init_pos_y", 0.0),
        declare_parameter<double>("init_pos_z", 0.0));
      Eigen::Quaternionf init_quat(
        declare_parameter<double>("init_ori_w", 1.0),
        declare_parameter<double>("init_ori_x", 0.0),
        declare_parameter<double>("init_ori_y", 0.0),
        declare_parameter<double>("init_ori_z", 0.0));

      pose_estimator.reset(new hdl_localization::PoseEstimator(
        registration,
        get_clock()->now(),
        init_pos,
        init_quat,
        cool_time_duration,
        acc_cov,
        gyr_cov,
        b_acc_cov,
        b_gyr_cov));
    }
  }

private:
  // ===========================================
  // IMU Callback - UKF prediction only (no odom publish here)
  // Also accumulate gyro delta for GICP initial guess
  // ===========================================
  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg) {
    if (use_imu) {
      std::lock_guard<std::mutex> lock(pose_estimator_mutex);
      if (pose_estimator && globalmap) {
        const auto& acc = imu_msg->linear_acceleration;
        const auto& gyro = imu_msg->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;

        Eigen::Vector3f gyro_vec = gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z);

        // Run UKF prediction with IMU data (orientation update)
        pose_estimator->predict(
          imu_msg->header.stamp,
          acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z),
          gyro_vec);

        // Accumulate gyro for GICP initial guess
        // This helps GICP converge correctly when the robot is rotating
        {
          std::lock_guard<std::mutex> gyro_lock(gyro_delta_mutex);
          double dt = 0.01;  // Approximate IMU rate
          if (last_imu_stamp.nanoseconds() > 0) {
            dt = (rclcpp::Time(imu_msg->header.stamp) - last_imu_stamp).seconds();
            if (dt > 0.0 && dt < 0.1) {
              // Integrate gyro to get delta rotation
              Eigen::AngleAxisf delta_rot(gyro_vec.norm() * dt, gyro_vec.normalized());
              accumulated_gyro_delta = accumulated_gyro_delta * Eigen::Quaternionf(delta_rot);
              accumulated_gyro_delta.normalize();
            }
          }
          last_imu_stamp = imu_msg->header.stamp;
        }
      }
    }
  }

  // Helper function to remove leading slash from frame_id (TF2 requirement)
  std::string sanitize_frame_id(const std::string& frame_id) {
    if (!frame_id.empty() && frame_id[0] == '/') {
      return frame_id.substr(1);
    }
    return frame_id;
  }

  // ===========================================
  // Points Callback - Frame-to-frame Fast GICP odometry (LiDAR rate)
  // ===========================================
  void points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg) {
    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if (!pose_estimator) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "waiting for initial pose input!!");
      return;
    }

    if (!globalmap) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "globalmap has not been received!!");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    // Sanitize frame_id
    pcl_cloud->header.frame_id = sanitize_frame_id(pcl_cloud->header.frame_id);

    if (pcl_cloud->empty()) {
      RCLCPP_ERROR(get_logger(), "cloud is empty!!");
      return;
    }

    // transform pointcloud into odom_child_frame_id
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, *tf_buffer)) {
      RCLCPP_ERROR(get_logger(), "point cloud cannot be transformed into target frame!!");
      return;
    }

    auto filtered = downsample(cloud);

    if (relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    // ===========================================
    // Frame-to-frame Fast GICP odometry (LiDAR rate ~20Hz)
    // Compute delta motion and feed to UKF via predict_odom
    // ===========================================
    Eigen::Vector3f linear_vel = Eigen::Vector3f::Zero();
    Eigen::Vector3f angular_vel = Eigen::Vector3f::Zero();

    // Calculate dt for prediction
    rclcpp::Time curr_stamp(stamp);
    double dt = 0.05;
    if (prev_scan_stamp.nanoseconds() > 0) {
      dt = (curr_stamp - prev_scan_stamp).seconds();
      if (dt < 0.001 || dt > 1.0) dt = 0.05;  // Default to 20Hz
    }

    // Frame-to-frame ICP using Fast GICP
    if (prev_scan && prev_scan->size() > 100 && filtered->size() > 100) {
      // Build initial guess from gyro
      Eigen::Quaternionf gyro_delta_copy;
      {
        std::lock_guard<std::mutex> gyro_lock(gyro_delta_mutex);
        gyro_delta_copy = accumulated_gyro_delta;
        accumulated_gyro_delta = Eigen::Quaternionf::Identity();
      }

      Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
      initial_guess.block<3, 3>(0, 0) = gyro_delta_copy.toRotationMatrix();

      // Set up Fast GICP for frame-to-frame matching
      frame_to_frame_gicp->setInputSource(filtered);
      frame_to_frame_gicp->setInputTarget(prev_scan);

      // Measure ICP computation time
      auto icp_start = std::chrono::high_resolution_clock::now();

      pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
      frame_to_frame_gicp->align(*aligned, initial_guess);

      auto icp_end = std::chrono::high_resolution_clock::now();
      double icp_time_ms = std::chrono::duration<double, std::milli>(icp_end - icp_start).count();

      if (frame_to_frame_gicp->hasConverged()) {
        Eigen::Matrix4f delta = frame_to_frame_gicp->getFinalTransformation();
        Eigen::Vector3f delta_trans = delta.block<3, 1>(0, 3);
        Eigen::Matrix3f delta_rot = delta.block<3, 3>(0, 0);
        double trans_norm = delta_trans.norm();

        // Sanity check: reject unreasonably large deltas
        double max_delta = 1.0;  // 1 meter max per frame
        double fitness = frame_to_frame_gicp->getFitnessScore();

        if (trans_norm < max_delta && delta_trans.allFinite()) {
          // Compute world-frame velocity for odom message
          Eigen::Quaternionf current_quat = pose_estimator->quat();
          Eigen::Vector3f world_delta = current_quat.toRotationMatrix() * delta_trans;
          if (dt > 0.001 && dt < 1.0) {
            linear_vel = world_delta / dt;

            // Extract rotation and compute angular velocity
            Eigen::AngleAxisf angle_axis(delta_rot);
            angular_vel = (angle_axis.axis() * angle_axis.angle()) / dt;
          }

          // Tightly-coupled fusion: GICP measurement corrects IMU-predicted state
          // Convert frame-to-frame delta to absolute pose
          Eigen::Matrix4f gicp_pose = pose_estimator->matrix() * delta;

          // UKF correction with GICP measurement
          // Kalman gain optimally weights IMU prediction vs GICP measurement
          pose_estimator->correct_gicp(gicp_pose, fitness);

          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 200,
            "GICP-UKF: delta=(%.4f, %.4f, %.4f), fitness=%.4f, src=%zu, tgt=%zu",
            delta_trans.x(), delta_trans.y(), delta_trans.z(), fitness,
            filtered->size(), prev_scan->size());
        } else {
          RCLCPP_WARN(get_logger(), "Fast-GICP: %.2fms, REJECTED (large delta: %.4f)",
            icp_time_ms, trans_norm);
        }
      } else {
        RCLCPP_WARN(get_logger(), "Fast-GICP: %.2fms, NOT CONVERGED", icp_time_ms);
      }
    }

    // Store current scan for next iteration and NDT
    prev_scan = filtered;
    prev_scan_stamp = stamp;
    last_scan = filtered;
    last_scan_stamp = stamp;

    // ===========================================
    // Publish odometry at LiDAR rate (~20Hz)
    // ===========================================
    publish_odometry(stamp, pose_estimator->matrix(), linear_vel, angular_vel);

    // ===========================================
    // Queue scan for NDT thread (low frequency global alignment)
    // ===========================================
    {
      std::lock_guard<std::mutex> lock(ndt_mutex);
      ndt_scan_queue = filtered;
      ndt_scan_stamp = stamp;
      ndt_has_new_scan = true;
    }
    ndt_cv.notify_one();

  }

  // ===========================================
  // NDT Thread - Global map alignment (1-2Hz)
  // ===========================================
  void ndt_thread_func() {
    double ndt_period_ms = 1000.0 / ndt_rate;
    auto last_ndt_time = std::chrono::steady_clock::now();

    while (ndt_thread_running && rclcpp::ok()) {
      pcl::PointCloud<PointT>::ConstPtr scan;
      rclcpp::Time scan_stamp;

      // Wait for new scan or timeout
      {
        std::unique_lock<std::mutex> lock(ndt_mutex);
        ndt_cv.wait_for(lock, std::chrono::milliseconds((int)ndt_period_ms),
          [this]{ return ndt_has_new_scan || !ndt_thread_running; });

        if (!ndt_thread_running) break;
        if (!ndt_has_new_scan) continue;

        // Rate limiting
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_ndt_time).count();
        if (elapsed < ndt_period_ms * 0.8) {
          continue;  // Skip if too soon
        }

        scan = ndt_scan_queue;
        scan_stamp = ndt_scan_stamp;
        ndt_has_new_scan = false;
        last_ndt_time = now;
      }

      if (!scan || scan->empty()) continue;
      if (!globalmap) continue;
      if (!registration) continue;

      // ===========================================
      // Perform NDT alignment to global map
      // ===========================================
      std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
      if (!pose_estimator) continue;

      try {

      // Use current UKF estimate as initial guess
      Eigen::Matrix4f init_guess = pose_estimator->matrix();
      Eigen::Vector3f current_pos = pose_estimator->pos();

      // Extract local map around current position for efficient NDT matching
      size_t local_map_size = 0;
      if (use_local_map && globalmap_kdtree) {
        pcl::PointCloud<PointT>::Ptr local_map = extractLocalMap(current_pos, local_map_radius);
        if (local_map && local_map->size() > 100) {  // Minimum points threshold
          registration->setInputTarget(local_map);
          local_map_size = local_map->size();
        }
      }

      // NDT alignment
      pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
      registration->setInputSource(scan);

      // Measure NDT computation time
      auto ndt_start = std::chrono::high_resolution_clock::now();
      registration->align(*aligned, init_guess);
      auto ndt_end = std::chrono::high_resolution_clock::now();
      double ndt_time_ms = std::chrono::duration<double, std::milli>(ndt_end - ndt_start).count();

      if (registration->hasConverged()) {
        Eigen::Matrix4f ndt_result = registration->getFinalTransformation();
        Eigen::Vector3f ndt_pos = ndt_result.block<3, 1>(0, 3);
        Eigen::Quaternionf ndt_quat(ndt_result.block<3, 3>(0, 0));

        // Calculate prediction error
        Eigen::Vector3f pred_pos = pose_estimator->pos();
        Eigen::Vector3f pred_error = ndt_pos - pred_pos;

        // Apply NDT correction to UKF (reset drift)
        // Use correct_ndt to directly set UKF state to NDT result
        double ndt_score = registration->getFitnessScore();
        pose_estimator->correct_ndt(ndt_result, ndt_score);

        RCLCPP_INFO(get_logger(),
          "NDT: %.2fms, pos: (%.3f, %.3f, %.3f), score: %.4f, pred_err: (%.3f, %.3f, %.3f), scan: %zu, localmap: %zu",
          ndt_time_ms, ndt_pos.x(), ndt_pos.y(), ndt_pos.z(),
          registration->getFitnessScore(),
          pred_error.x(), pred_error.y(), pred_error.z(),
          scan->size(), local_map_size);

        // Publish aligned points
        if (aligned_pub->get_subscription_count()) {
          aligned->header.frame_id = "map";
          aligned->header.stamp = scan->header.stamp;
          sensor_msgs::msg::PointCloud2 aligned_msg;
          pcl::toROSMsg(*aligned, aligned_msg);
          aligned_pub->publish(aligned_msg);
        }
      } else {
        RCLCPP_WARN(get_logger(), "NDT: %.2fms, NOT CONVERGED", ndt_time_ms);
      }

      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "NDT thread exception: %s", e.what());
      } catch (...) {
        RCLCPP_ERROR(get_logger(), "NDT thread unknown exception");
      }
    }
  }

  /**
   * @brief callback for globalmap input
   */
  void globalmap_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg) {
    RCLCPP_INFO(get_logger(), "globalmap received!");

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    // Calculate bounding box of global map
    PointT min_pt, max_pt;
    pcl::getMinMax3D(*globalmap, min_pt, max_pt);
    RCLCPP_INFO(get_logger(), "globalmap bbox: min(%.2f, %.2f, %.2f) max(%.2f, %.2f, %.2f)",
      min_pt.x, min_pt.y, min_pt.z, max_pt.x, max_pt.y, max_pt.z);

    // Build KD-Tree for local map extraction
    globalmap_kdtree.reset(new pcl::KdTreeFLANN<PointT>());
    globalmap_kdtree->setInputCloud(globalmap);
    RCLCPP_INFO(get_logger(), "globalmap KD-Tree built with %zu points", globalmap->size());

    // Set full globalmap as initial target (will be replaced by local map during operation)
    registration->setInputTarget(globalmap);

    if (use_global_localization) {
      RCLCPP_INFO(get_logger(), "set globalmap for global localization!");
      auto req = std::make_shared<hdl_global_localization::srv::SetGlobalMap::Request>();
      pcl::toROSMsg(*globalmap, req->global_map);
      auto result = set_global_map_service->async_send_request(req);
      // Note: Not blocking here to avoid deadlock
    }
  }

  /**
   * @brief Extract local map around current position for efficient NDT matching
   * @param center  Current position (center of local map)
   * @param radius  Radius of local map extraction
   * @return Local map point cloud (downsampled to match scan density)
   */
  pcl::PointCloud<PointT>::Ptr extractLocalMap(const Eigen::Vector3f& center, double radius) {
    if (!globalmap || !globalmap_kdtree) {
      return nullptr;
    }

    pcl::PointCloud<PointT>::Ptr local_map(new pcl::PointCloud<PointT>());

    // Use radius search to extract nearby points
    PointT search_point;
    search_point.x = center.x();
    search_point.y = center.y();
    search_point.z = center.z();

    std::vector<int> indices;
    std::vector<float> distances;

    if (globalmap_kdtree->radiusSearch(search_point, radius, indices, distances) > 0) {
      local_map->reserve(indices.size());
      for (int idx : indices) {
        local_map->push_back(globalmap->points[idx]);
      }
    }

    // Downsample local map to match scan density for better NDT matching
    // Using larger voxel size (0.25m) to reduce point density and improve score
    if (local_map->size() > 10000) {
      pcl::VoxelGrid<PointT> local_voxel;
      local_voxel.setLeafSize(0.25f, 0.25f, 0.25f);
      local_voxel.setInputCloud(local_map);
      pcl::PointCloud<PointT>::Ptr filtered_local(new pcl::PointCloud<PointT>());
      local_voxel.filter(*filtered_local);
      local_map = filtered_local;
    }

    local_map->header = globalmap->header;
    local_map->width = local_map->size();
    local_map->height = 1;
    local_map->is_dense = true;

    return local_map;
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   */
  bool relocalize(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res) {
    if (last_scan == nullptr) {
      RCLCPP_INFO_STREAM(get_logger(), "no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    auto query_req = std::make_shared<hdl_global_localization::srv::QueryGlobalLocalization::Request>();
    pcl::toROSMsg(*scan, query_req->cloud);
    query_req->max_num_candidates = 1;

    auto query_result_future = query_global_localization_service->async_send_request(query_req);
    if (rclcpp::spin_until_future_complete(rclcpp::Node::SharedPtr(this), query_result_future) != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Failed to call QueryGlobalLocalization service");
      return false;
    }
    auto query_result = query_result_future.get();

    if (query_result->poses.empty()) {
      RCLCPP_ERROR(get_logger(), "QueryGlobalLocalization returned empty poses array");
      return false;
    }

    const auto& result = query_result->poses[0];

    RCLCPP_INFO_STREAM(get_logger(), "--- Global localization result ---");
    RCLCPP_INFO_STREAM(get_logger(), "Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    RCLCPP_INFO_STREAM(get_logger(), "Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    RCLCPP_INFO_STREAM(get_logger(), "Error :" << query_result->errors[0]);
    RCLCPP_INFO_STREAM(get_logger(), "Inlier:" << query_result->inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(
      new hdl_localization::PoseEstimator(registration, get_clock()->now(), Eigen::Vector3f(pose.translation()), Eigen::Quaternionf(pose.linear()), cool_time_duration,
        acc_cov, gyr_cov, b_acc_cov, b_gyr_cov));

    relocalizing = false;

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   */
  void initialpose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_msg) {
    RCLCPP_INFO(get_logger(), "initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(
      new hdl_localization::PoseEstimator(registration, get_clock()->now(), Eigen::Vector3f(p.x, p.y, p.z), Eigen::Quaternionf(q.w, q.x, q.y, q.z), cool_time_duration,
        acc_cov, gyr_cov, b_acc_cov, b_gyr_cov));

    // Reset previous scan for GICP
    prev_scan = nullptr;
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  void publish_odometry(const rclcpp::Time& stamp, const Eigen::Matrix4f& pose, const Eigen::Vector3f& linear_vel, const Eigen::Vector3f& angular_vel) {
    // broadcast the transform over tf
    if (send_tf_transforms) {
      if (tf_buffer->canTransform(robot_odom_frame_id, odom_child_frame_id, rclcpp::Time((int64_t)0, get_clock()->get_clock_type()))) {
        geometry_msgs::msg::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
        map_wrt_frame.header.stamp = stamp;
        map_wrt_frame.header.frame_id = odom_child_frame_id;
        map_wrt_frame.child_frame_id = "map";

        geometry_msgs::msg::TransformStamped frame_wrt_odom = tf_buffer->lookupTransform(
          robot_odom_frame_id,
          odom_child_frame_id,
          rclcpp::Time((int64_t)0, get_clock()->get_clock_type()),
          rclcpp::Duration(std::chrono::milliseconds(100)));
        Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

        geometry_msgs::msg::TransformStamped map_wrt_odom;
        tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

        tf2::Transform odom_wrt_map;
        tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
        odom_wrt_map = odom_wrt_map.inverse();

        geometry_msgs::msg::TransformStamped odom_trans;
        odom_trans.transform = tf2::toMsg(odom_wrt_map);
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = robot_odom_frame_id;

        tf_broadcaster->sendTransform(odom_trans);
      } else {
        geometry_msgs::msg::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = odom_child_frame_id;
        tf_broadcaster->sendTransform(odom_trans);
      }
    }

    // publish the transform
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";

    odom.pose.pose = tf2::toMsg(Eigen::Isometry3d(pose.cast<double>()));
    odom.child_frame_id = odom_child_frame_id;
    odom.twist.twist.linear.x = linear_vel.x();
    odom.twist.twist.linear.y = linear_vel.y();
    odom.twist.twist.linear.z = linear_vel.z();
    odom.twist.twist.angular.x = angular_vel.x();
    odom.twist.twist.angular.y = angular_vel.y();
    odom.twist.twist.angular.z = angular_vel.z();

    pose_pub->publish(odom);
  }

private:
  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;
  bool send_tf_transforms;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub;
  rclcpp::Publisher<hdl_localization::msg::ScanMatchingStatus>::SharedPtr status_pub;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  // globalmap and registration methods
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::KdTreeFLANN<PointT>::Ptr globalmap_kdtree;  // KD-Tree for local map extraction
  bool use_local_map;
  double local_map_radius;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;       // NDT for global map alignment

  // Fast GICP for frame-to-frame odometry
  pcl::Registration<PointT, PointT>::Ptr frame_to_frame_gicp;

  // Frame-to-frame scan storage
  pcl::PointCloud<PointT>::ConstPtr prev_scan;
  rclcpp::Time prev_scan_stamp;

  // IMU gyro accumulation for Fast GICP initial guess
  std::mutex gyro_delta_mutex;
  Eigen::Quaternionf accumulated_gyro_delta = Eigen::Quaternionf::Identity();
  rclcpp::Time last_imu_stamp;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // NDT thread for low-frequency global alignment
  std::thread ndt_thread;
  std::atomic<bool> ndt_thread_running;
  std::mutex ndt_mutex;
  std::condition_variable ndt_cv;
  pcl::PointCloud<PointT>::ConstPtr ndt_scan_queue;
  rclcpp::Time ndt_scan_stamp;
  std::atomic<bool> ndt_has_new_scan;
  double ndt_rate;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  rclcpp::Time last_scan_stamp;
  rclcpp::Client<hdl_global_localization::srv::SetGlobalMap>::SharedPtr set_global_map_service;
  rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedPtr query_global_localization_service;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr relocalize_server;

  // Parameters
  double cool_time_duration;
  std::string reg_method;
  std::string ndt_neighbor_search_method;
  double ndt_neighbor_search_radius;
  double ndt_resolution;
  bool enable_robot_odometry_prediction;

  // IMU noise covariance parameters
  double acc_cov;
  double gyr_cov;
  double b_acc_cov;
  double b_gyr_cov;
};
}  // namespace hdl_localization

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_localization::HdlLocalizationNodelet)
