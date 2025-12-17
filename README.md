# HDL Localization ROS2

ROS2 기반 3D LiDAR 위치 추정 패키지 (NDT + Fast GICP 센서 퓨전)

## 개요

이 패키지는 IMU와 LiDAR를 융합한 멀티레이트 센서 퓨전 아키텍처를 제공합니다:
- **IMU (100-400Hz)**: UKF 예측
- **Fast GICP (10-20Hz)**: 프레임-투-프레임 오도메트리 → UKF 보정
- **NDT (1-2Hz)**: 글로벌 맵 정합 → 드리프트 보정

## 빌드 방법

### 의존성 설치

```bash
# ROS2 Humble 기준
sudo apt install ros-humble-pcl-ros ros-humble-tf2-eigen ros-humble-tf2-geometry-msgs
```

### 빌드

```bash
cd ~/colcon_ws
colcon build --packages-select hdl_localization hdl_global_localization fast_gicp ndt_omp
source install/setup.bash
```

Release 모드로 빌드 (성능 최적화):
```bash
colcon build --packages-select hdl_localization --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## 실행 방법

### 기본 실행

```bash
ros2 launch hdl_localization hdl_localization_spadi.launch.py \
  globalmap_pcd:=/path/to/your/map.pcd
```

### Rosbag과 함께 실행

```bash
# 터미널 1: HDL Localization 실행
ros2 launch hdl_localization hdl_localization_spadi.launch.py \
  globalmap_pcd:=/path/to/map.pcd

# 터미널 2: Rosbag 재생
ros2 bag play /path/to/rosbag --clock
```

## 설정 파라미터

### 토픽 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `points_topic` | `/spadi/pointcloud` | LiDAR 포인트클라우드 토픽 |
| `imu_topic` | `/spadi/imu` | IMU 토픽 |
| `odom_child_frame_id` | `spadi/lidar` | 오도메트리 child frame ID |

### IMU 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `use_imu` | `true` | IMU 사용 여부 |
| `invert_acc` | `false` | 가속도계 부호 반전 |
| `invert_gyro` | `false` | 자이로스코프 부호 반전 |
| `acc_cov` | `0.5` | 가속도계 노이즈 공분산 |
| `gyr_cov` | `0.3` | 자이로스코프 노이즈 공분산 |
| `b_acc_cov` | `0.0001` | 가속도계 바이어스 공분산 |
| `b_gyr_cov` | `0.0001` | 자이로스코프 바이어스 공분산 |

### NDT 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `reg_method` | `NDT_OMP` | 등록 방법 (`NDT_OMP`, `NDT_CUDA_P2D`, `NDT_CUDA_D2D`) |
| `ndt_resolution` | `1.0` | NDT 해상도 (m) |
| `ndt_rate` | `1.0` | NDT 실행 주파수 (Hz) |
| `ndt_neighbor_search_method` | `DIRECT7` | 이웃 탐색 방법 (`DIRECT1`, `DIRECT7`, `KDTREE`) |
| `ndt_neighbor_search_radius` | `2.0` | 이웃 탐색 반경 (m) |

### Fast GICP 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `gicp_correspondence_distance` | `0.3` | 대응점 거리 임계값 (m) |
| `gicp_num_threads` | `4` | GICP 스레드 수 |
| `gicp_max_iterations` | `32` | 최대 반복 횟수 |
| `downsample_resolution` | `0.1` | 다운샘플링 해상도 (m) |

### 초기 위치 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `specify_init_pose` | `true` | 초기 위치 지정 여부 |
| `init_pos_x` | `0.0` | 초기 X 좌표 (m) |
| `init_pos_y` | `0.0` | 초기 Y 좌표 (m) |
| `init_pos_z` | `0.0` | 초기 Z 좌표 (m) |
| `init_ori_w` | `1.0` | 초기 방향 쿼터니언 w |
| `init_ori_x` | `0.0` | 초기 방향 쿼터니언 x |
| `init_ori_y` | `0.0` | 초기 방향 쿼터니언 y |
| `init_ori_z` | `0.0` | 초기 방향 쿼터니언 z |

### 로깅 설정

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `log_dir` | `hdl_localization/Log` | 로그 저장 디렉토리 |

## 출력 파일

프로그램 종료 시 자동으로 `Log/` 폴더에 다음 파일들이 저장됩니다:

### `traj_lidar.txt`
추정된 궤적 파일 (TUM 형식):
```
timestamp tx ty tz qx qy qz qw
```

### `profiling_stats.txt`
연산 시간 프로파일링 통계:

**Localization 스레드 (20Hz)**:
- `localization`: 전체 콜백 시간
- `tf_transform`: TF 변환 시간
- `downsample`: 다운샘플링 시간
- `gicp`: Fast GICP 시간
- `ukf_gicp`: UKF 보정 시간

**Global Optimization 스레드 (1Hz)**:
- `global_optimization`: 전체 NDT 처리 시간
- `local_map_extract`: 로컬 맵 추출 시간
- `ndt`: NDT 정합 시간
- `ukf_ndt`: UKF 보정 시간

## 토픽

### 구독 토픽
- `/points_topic` (sensor_msgs/PointCloud2): LiDAR 포인트클라우드
- `/imu_topic` (sensor_msgs/Imu): IMU 데이터
- `/globalmap` (sensor_msgs/PointCloud2): 글로벌 맵
- `/initialpose` (geometry_msgs/PoseWithCovarianceStamped): 초기 위치 (RViz에서 설정 가능)

### 발행 토픽
- `/odom` (nav_msgs/Odometry): 추정된 오도메트리
- `/aligned_points` (sensor_msgs/PointCloud2): 정렬된 포인트클라우드

## 프로파일링 결과 예시

### Localization (20Hz)
| 항목 | 평균 | p95 | 최대 |
|------|------|-----|------|
| total | 20.22ms | 29.33ms | 104.89ms |
| GICP | 14.30ms | 21.87ms | 45.00ms |
| downsample | 0.34ms | 0.47ms | 0.59ms |

### Global Optimization (1Hz)
| 항목 | 평균 | p95 | 최대 |
|------|------|-----|------|
| total | 76.74ms | 81.25ms | 85.07ms |
| local_map_extract | 39.56ms | 42.05ms | 43.70ms |
| NDT | 22.42ms | 24.10ms | 27.44ms |

## Docker 사용 (선택사항)

Docker 환경에서 실행하려면:

```bash
# Docker 이미지 빌드
cd docker
docker build -t hdl-localization-ros2:latest .

# 컨테이너 실행
xhost +local:docker
nvidia-docker run --privileged -it \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --volume=/path/to/hdl-localization-ROS2:/root/workspace/src \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
  --net=host \
  --ipc=host \
  --name=hdl-localization-ros2 \
  --env="DISPLAY=$DISPLAY" \
  hdl-localization-ros2:latest /bin/bash
```

## Acknowledgement

이 저장소는 다음 패키지들을 기반으로 합니다:

- [DataspeedInc/hdl_localization](https://github.com/DataspeedInc/hdl_localization/tree/ros2)
- [DataspeedInc/hdl_global_localization](https://github.com/DataspeedInc/hdl_global_localization/tree/ros2)
- [DataspeedInc/fast_gicp](https://github.com/DataspeedInc/fast_gicp/tree/ros2)
- [tier4/ndt_omp](https://github.com/tier4/ndt_omp)
