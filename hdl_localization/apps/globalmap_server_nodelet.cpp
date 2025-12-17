#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>

////***************************start pyc modified***************************/
#include <std_msgs/msg/string.hpp>
using namespace std;
/***************************end pyc modified***************************/  ///

namespace hdl_localization {

// rclcpp::Node 얘를 상속받은 GlobalmapServerNodelet class
class GlobalmapServerNodelet : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZ;
  using PointRGB = pcl::PointXYZRGB;

  GlobalmapServerNodelet(const rclcpp::NodeOptions& options)
  // GlobalmapServerNodelet 생성자. 기본적으로 map_server 라는 이름을 갖는 인스턴스를 만듦
  : Node("map_server", options), last_subscriber_count_(0) {
    initialize_params();

    // Latched QoS for globalmap (transient_local keeps last message for late subscribers)
    auto latch_qos = rclcpp::QoS(1).transient_local().reliable();
    globalmap_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/globalmap", latch_qos);

    // Timer to monitor subscriber count and republish when new subscriber joins
    // This handles RViz checkbox toggle (off->on creates new subscription)
    globalmap_pub_timer = create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        if (!globalmap_msg) return;

        size_t current_count = globalmap_pub->get_subscription_count();

        // Publish when new subscriber detected or on first run
        if (current_count > last_subscriber_count_ || (current_count > 0 && !initial_published_)) {
          globalmap_pub->publish(*globalmap_msg);
          initial_published_ = true;
          RCLCPP_INFO(get_logger(), "Globalmap published (subscribers: %zu)", current_count);
        }

        last_subscriber_count_ = current_count;
      });

    ////***************************start pyc modified***************************/
    map_update_sub =
      create_subscription<std_msgs::msg::String>("/map_request/pcd", latch_qos, std::bind(&GlobalmapServerNodelet::map_update_callback, this, std::placeholders::_1));
    /***************************end pyc modified***************************/  ///
  }

private:
  // Helper function to check if PCD has RGB field
  bool hasRgbField(const std::string& pcd_path) {
    pcl::PCLPointCloud2 cloud2;
    if (pcl::io::loadPCDFile(pcd_path, cloud2) < 0) {
      return false;
    }
    for (const auto& field : cloud2.fields) {
      if (field.name == "rgb" || field.name == "rgba") {
        return true;
      }
    }
    return false;
  }

  // Load PCD and convert to PointCloud2 message (auto-detect point type)
  void loadPcdToMsg(const std::string& pcd_path, double downsample_resolution) {
    bool has_rgb = hasRgbField(pcd_path);
    RCLCPP_INFO(get_logger(), "PCD point type: %s", has_rgb ? "PointXYZRGB" : "PointXYZ");

    if (has_rgb) {
      // Load as PointXYZRGB
      pcl::PointCloud<PointRGB>::Ptr cloud_rgb(new pcl::PointCloud<PointRGB>());
      pcl::io::loadPCDFile(pcd_path, *cloud_rgb);
      cloud_rgb->header.frame_id = "map";

      // Apply UTM offset if available
      bool convert_utm_to_local = declare_parameter<bool>("convert_utm_to_local", true);
      std::ifstream utm_file(pcd_path + ".utm");
      if (utm_file.is_open() && convert_utm_to_local) {
        double utm_easting, utm_northing, altitude;
        utm_file >> utm_easting >> utm_northing >> altitude;
        for (auto& pt : cloud_rgb->points) {
          pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utm_northing, altitude);
        }
        RCLCPP_INFO_STREAM(get_logger(),
          "Global map offset by UTM reference coordinates (x = " << utm_easting << ", y = " << utm_northing << ") and altitude (z = " << altitude << ")");
      }

      // Downsample
      boost::shared_ptr<pcl::VoxelGrid<PointRGB>> voxelgrid(new pcl::VoxelGrid<PointRGB>());
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      voxelgrid->setInputCloud(cloud_rgb);
      pcl::PointCloud<PointRGB>::Ptr filtered_rgb(new pcl::PointCloud<PointRGB>());
      voxelgrid->filter(*filtered_rgb);

      // Convert to ROS message
      globalmap_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
      pcl::toROSMsg(*filtered_rgb, *globalmap_msg);
      globalmap_msg->header.frame_id = "map";

      // Also store as PointXYZ for internal use
      globalmap.reset(new pcl::PointCloud<PointT>());
      pcl::copyPointCloud(*filtered_rgb, *globalmap);

      RCLCPP_INFO(get_logger(), "Loaded PointXYZRGB map with %zu points", filtered_rgb->size());
    } else {
      // Load as PointXYZ
      globalmap.reset(new pcl::PointCloud<PointT>());
      pcl::io::loadPCDFile(pcd_path, *globalmap);
      globalmap->header.frame_id = "map";

      // Apply UTM offset if available
      bool convert_utm_to_local = declare_parameter<bool>("convert_utm_to_local", true);
      std::ifstream utm_file(pcd_path + ".utm");
      if (utm_file.is_open() && convert_utm_to_local) {
        double utm_easting, utm_northing, altitude;
        utm_file >> utm_easting >> utm_northing >> altitude;
        for (auto& pt : globalmap->points) {
          pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utm_northing, altitude);
        }
        RCLCPP_INFO_STREAM(get_logger(),
          "Global map offset by UTM reference coordinates (x = " << utm_easting << ", y = " << utm_northing << ") and altitude (z = " << altitude << ")");
      }

      // Downsample
      boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      voxelgrid->setInputCloud(globalmap);
      pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
      voxelgrid->filter(*filtered);
      globalmap = filtered;

      // Convert to ROS message
      globalmap_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
      pcl::toROSMsg(*globalmap, *globalmap_msg);
      globalmap_msg->header.frame_id = "map";

      RCLCPP_INFO(get_logger(), "Loaded PointXYZ map with %zu points", globalmap->size());
    }
  }

  ////***************************start pyc modified***************************/
  void map_update_callback(const std_msgs::msg::String::SharedPtr msg) {
    cout << "in map update callback" << endl;
    RCLCPP_INFO(get_logger(), "Received map request, map path : %s", msg->data.c_str());
    double downsample_resolution = declare_parameter<double>("downsample_resolution", 0.1);
    loadPcdToMsg(msg->data, downsample_resolution);
  }
  /***************************end pyc modified***************************/  ///

  void initialize_params() {
    ////***************************start pyc modified***************************/
    cout << endl << "initialize_params" << endl << endl;
    /***************************end pyc modified***************************/  ///

    // read globalmap from a pcd file
    std::string globalmap_pcd = declare_parameter<std::string>("globalmap_pcd", std::string(""));

    ////***************************start pyc modified***************************/
    cout << endl << globalmap_pcd << endl << endl;
    /***************************end pyc modified***************************/  ///

    // downsample globalmap
    double downsample_resolution = declare_parameter<double>("downsample_resolution", 0.1);
    loadPcdToMsg(globalmap_pcd, downsample_resolution);
  }

private:
  // ROS
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_pub;

  sensor_msgs::msg::PointCloud2::SharedPtr globalmap_msg;

  rclcpp::TimerBase::SharedPtr globalmap_pub_timer;
  pcl::PointCloud<PointT>::Ptr globalmap;

  ////***************************start pyc modified***************************/
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr map_update_sub;
  /***************************end pyc modified***************************/  ///

  // Subscriber monitoring for auto-republish
  size_t last_subscriber_count_;
  bool initial_published_ = false;
};

}  // namespace hdl_localization

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_localization::GlobalmapServerNodelet)
