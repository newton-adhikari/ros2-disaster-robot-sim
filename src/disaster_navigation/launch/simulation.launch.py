from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    disaster_world_pkg = get_package_share_directory('disaster_world')
    disaster_robot_pkg = get_package_share_directory('disaster_robot')
    disaster_nav_pkg   = get_package_share_directory('disaster_navigation')
    gazebo_ros_pkg     = get_package_share_directory('gazebo_ros')

    return LaunchDescription([
        
    ])