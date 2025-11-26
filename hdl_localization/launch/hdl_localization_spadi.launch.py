from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration

import launch_ros.actions
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    """
    HDL Localization launch file for SPADI (MLX LiDAR + IMU)

    Usage:
        ros2 launch hdl_localization hdl_localization_spadi.launch.py globalmap_pcd:=/path/to/map.pcd

    Sensor Configuration:
        - LiDAR: MLX (20Hz), topic: /spadi/pointcloud, frame: /spadi/lidar
        - IMU: 100Hz, topic: /spadi/imu, frame: imu
        - LiDAR-IMU extrinsic: T=[-0.052, -0.083, -0.018], R=Identity
        - IMU covariance: acc_cov=0.5, gyr_cov=0.3
    """

    # Required arguments
    globalmap_pcd = DeclareLaunchArgument(
        "globalmap_pcd",
        default_value="/root/ros2_ws/src/map.pcd",
        description="Path to the global map PCD file",
    )

    # LiDAR configuration
    points_topic = LaunchConfiguration("points_topic", default="/spadi/pointcloud")
    odom_child_frame_id = LaunchConfiguration(
        "odom_child_frame_id", default="spadi/lidar"  # LiDAR frame (without leading /)
    )

    # IMU configuration
    imu_topic = LaunchConfiguration("imu_topic", default="/spadi/imu")

    # Simulation time (set to false for real robot)
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    # IMU settings
    use_imu = LaunchConfiguration("use_imu", default="true")
    invert_imu_acc = LaunchConfiguration("invert_imu_acc", default="false")
    invert_imu_gyro = LaunchConfiguration("invert_imu_gyro", default="false")

    # Localization settings
    use_global_localization = LaunchConfiguration(
        "use_global_localization", default="false"
    )
    enable_robot_odometry_prediction = LaunchConfiguration(
        "enable_robot_odometry_prediction", default="false"
    )
    robot_odom_frame_id = LaunchConfiguration("robot_odom_frame_id", default="odom")

    # Static TF: imu -> spadi/lidar (LiDAR-IMU extrinsic calibration)
    # Translation: [-0.052, -0.083, -0.018] (x, y, z)
    # Rotation: Identity (quaternion: 0, 0, 0, 1)
    lidar_imu_tf = Node(
        name="lidar_imu_tf",
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "-0.052", "-0.083", "-0.018",  # x, y, z translation
            "0", "0", "0", "1",             # quaternion (x, y, z, w)
            "imu", "spadi/lidar"            # parent_frame, child_frame
        ],
    )

    # Static TF: map -> odom (identity, since no odometry source)
    map_odom_tf = Node(
        name="map_odom_tf",
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "0", "0", "0",
            "0", "0", "0", "1",
            "map", "odom"
        ],
    )

    container = ComposableNodeContainer(
        name="hdl_localization_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="hdl_localization",
                plugin="hdl_localization::GlobalmapServerNodelet",
                name="globalmap_server",
                parameters=[
                    {"globalmap_pcd": LaunchConfiguration("globalmap_pcd")},
                    {"convert_utm_to_local": False},
                    {"downsample_resolution": 0.1},
                ],
            ),
            ComposableNode(
                package="hdl_localization",
                plugin="hdl_localization::HdlLocalizationNodelet",
                name="hdl_localization",
                remappings=[
                    ("/velodyne_points", points_topic),
                    ("/gpsimu_driver/imu_data", imu_topic),
                ],
                parameters=[
                    {"odom_child_frame_id": odom_child_frame_id},
                    # IMU settings
                    {"use_imu": use_imu},
                    {"invert_acc": invert_imu_acc},
                    {"invert_gyro": invert_imu_gyro},
                    # Timing
                    {"cool_time_duration": 2.0},
                    # Odometry prediction
                    {"enable_robot_odometry_prediction": enable_robot_odometry_prediction},
                    {"robot_odom_frame_id": robot_odom_frame_id},
                    # NDT registration method
                    # Available: NDT_OMP, NDT_CUDA_P2D, NDT_CUDA_D2D
                    {"reg_method": "NDT_OMP"},
                    {"ndt_neighbor_search_method": "DIRECT7"},
                    {"ndt_neighbor_search_radius": 1.0},
                    {"ndt_resolution": 0.5},
                    # Downsampling
                    {"downsample_resolution": 0.1},
                    # Initial pose: (0, 0, 0) facing forward
                    {"specify_init_pose": True},
                    {"init_pos_x": 0.0},
                    {"init_pos_y": 0.0},
                    {"init_pos_z": 0.0},
                    {"init_ori_w": 1.0},  # No rotation (identity quaternion)
                    {"init_ori_x": 0.0},
                    {"init_ori_y": 0.0},
                    {"init_ori_z": 0.0},
                    # Global localization
                    {"use_global_localization": use_global_localization},
                ],
            ),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            globalmap_pcd,
            launch_ros.actions.SetParameter(name="use_sim_time", value=use_sim_time),
            lidar_imu_tf,
            map_odom_tf,
            container,
        ]
    )
