from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import (SetEnvironmentVariable,IncludeLaunchDescription, TimerAction, LogInfo)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


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

    set_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        [model_path, ':', os.path.expanduser('~/.gazebo/models')]
    )

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file, 'verbose': 'false'}.items()
    )

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkg, 'launch', 'gzclient.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }]
    )

    # Spawn robot (delayed 3s to let Gazebo start)
    spawn_robot = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name='spawn_robot',
                output='screen',
                arguments=[
                    '-entity', 'disaster_robot',
                    '-topic',  'robot_description',
                    '-x', LaunchConfiguration('x'),
                    '-y', LaunchConfiguration('y'),
                    '-z', '1.0',   # ground plane is at z=1.0 in world — spawn AT the surface
                    '-Y', LaunchConfiguration('yaw'),
                ]
            )
        ]
    )

    ekf_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_filter_node',
                output='screen',
                parameters=[
                    ekf_params_file,
                    {'use_sim_time': True}
                ]
            )
        ]
    )

    slam_node = TimerAction(
        period=12.0,
        actions=[
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    slam_params_file,
                    {'use_sim_time': True}
                ],
                condition=IfCondition(LaunchConfiguration('slam'))
            )
        ]
    )

    return LaunchDescription([
        LogInfo(msg=''),
        LogInfo(msg='DISASTER ROBOT SIMULATOR — LAUNCH'),
        LogInfo(msg=''),
        set_model_path,
        gzserver,
        gzclient,
        robot_state_pub,
        spawn_robot,
        ekf_node,
        slam_node
    ])