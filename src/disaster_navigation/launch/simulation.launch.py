from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    disaster_world_pkg = get_package_share_directory('disaster_world')
    disaster_robot_pkg = get_package_share_directory('disaster_robot')
    disaster_nav_pkg   = get_package_share_directory('disaster_navigation')
    gazebo_ros_pkg     = get_package_share_directory('gazebo_ros')

    world_file       = os.path.join(disaster_world_pkg, 'worlds', 'disaster_collapsed_building.world')
    urdf_file        = os.path.join(disaster_robot_pkg, 'urdf',   'disaster_robot.urdf.xacro')
    nav2_params_file = os.path.join(disaster_nav_pkg,   'config', 'nav2_params.yaml')
    ekf_params_file  = os.path.join(disaster_nav_pkg,   'config', 'ekf_params.yaml')
    slam_params_file = os.path.join(disaster_nav_pkg,   'config', 'slam_toolbox_params.yaml')
    rviz_config_file = os.path.join(disaster_nav_pkg,   'rviz',   'disaster_full.rviz')
    model_path       = os.path.join(disaster_world_pkg, 'models')

    return LaunchDescription([
        
    ])