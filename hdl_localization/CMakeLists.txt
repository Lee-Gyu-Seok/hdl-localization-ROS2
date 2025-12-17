cmake_minimum_required(VERSION 3.5)
project(hdl_localization)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

cmake_policy(SET CMP0074 NEW)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
# find_package(pclomp REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(fast_gicp REQUIRED)
find_package(hdl_global_localization REQUIRED)

# KISS-ICP dependencies
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(Sophus REQUIRED)

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# KISS-ICP library
set(KISS_ICP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../add/kiss-icp/cpp/kiss_icp)
set(KISS_ICP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../add/kiss-icp/cpp)
include_directories(${KISS_ICP_INCLUDE_DIR})

# Use system-installed robin-map
find_package(tsl-robin-map REQUIRED)

# Build KISS-ICP core library
add_library(kiss_icp_core STATIC
  ${KISS_ICP_DIR}/core/Registration.cpp
  ${KISS_ICP_DIR}/core/VoxelHashMap.cpp
  ${KISS_ICP_DIR}/core/VoxelUtils.cpp
  ${KISS_ICP_DIR}/core/Preprocessing.cpp
  ${KISS_ICP_DIR}/core/Threshold.cpp
)
target_include_directories(kiss_icp_core PUBLIC
  ${KISS_ICP_INCLUDE_DIR}
)
target_link_libraries(kiss_icp_core PUBLIC
  Eigen3::Eigen
  TBB::tbb
  Sophus::Sophus
  tsl::robin_map
)
set_target_properties(kiss_icp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

########################
## message generation ##
########################
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/ScanMatchingStatus.msg
DEPENDENCIES
  std_msgs
  geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)

###########
## Build ##
###########

# nodelets
add_library(${PROJECT_NAME}_lib SHARED
  src/hdl_localization/pose_estimator.cpp
  apps/hdl_localization_nodelet.cpp
  apps/globalmap_server_nodelet.cpp
)
ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  pcl_ros
  pcl_conversions
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_srvs
  ndt_omp
  fast_gicp
  hdl_global_localization
)
rosidl_target_interfaces(${PROJECT_NAME}_lib
  ${PROJECT_NAME}
  rosidl_typesupport_cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${KISS_ICP_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME}_lib
  kiss_icp_core
  Sophus::Sophus
)
rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "${PROJECT_NAME}::HdlLocalizationNodelet"
  EXECUTABLE ${PROJECT_NAME}_node
)
rclcpp_components_register_node(${PROJECT_NAME}_lib
  PLUGIN "${PROJECT_NAME}::GlobalmapServerNodelet"
  EXECUTABLE ${PROJECT_NAME}_map_server
)

install(
  TARGETS
    ${PROJECT_NAME}_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY
    launch
    rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
