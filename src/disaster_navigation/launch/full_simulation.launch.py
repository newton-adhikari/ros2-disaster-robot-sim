from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import (DeclareLaunchArgument,SetEnvironmentVariable,IncludeLaunchDescription, TimerAction, LogInfo, ExecuteProcess)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():

    # ── Package paths ─────────────────────────────────────────────────
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

    # ── Arguments ─────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('slam', default_value='true'),
        DeclareLaunchArgument('map', default_value=''),
        DeclareLaunchArgument('x',   default_value='0.0'),
        DeclareLaunchArgument('y',   default_value='-2.0'),
        DeclareLaunchArgument('yaw', default_value='1.5708'),
        # Map saver
        DeclareLaunchArgument('map_output_dir',
            default_value=os.path.expanduser('~/disaster_results')),
        DeclareLaunchArgument('map_output_prefix', default_value='map'),
        # Navigation policy: none, frontier_explorer, potential_field_navigator,
        #                    rl_navigator, rl_navigator_model
        DeclareLaunchArgument('nav_policy', default_value='none'),
        DeclareLaunchArgument('rl_model_path', default_value=''),
        # EKF monitor CSV paths (empty = don't launch ekf_monitor)
        DeclareLaunchArgument('ekf_csv', default_value=''),
        DeclareLaunchArgument('collision_csv', default_value=''),
        DeclareLaunchArgument('launch_ekf_monitor', default_value='false'),
    ]

    # ── Environment ───────────────────────────────────────────────────
    set_model_path = SetEnvironmentVariable(
        'GAZEBO_MODEL_PATH',
        [model_path, ':', os.path.expanduser('~/.gazebo/models')]
    )

    # ── 1. Gazebo ─────────────────────────────────────────────────────
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

    # ── 2a. Robot state publisher ─────────────────────────────────────
    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]), value_type=str)

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description,
                      'use_sim_time': True}])

    # ── 2b. Delete old + spawn robot ─────────────────────────────────
    delete_old_robot = TimerAction(period=38.0, actions=[
        ExecuteProcess(
            cmd=['ros2', 'service', 'call', '/delete_entity',
                 'gazebo_msgs/srv/DeleteEntity',
                 '{name: "disaster_robot"}'],
            output='log')])

    spawn_robot = TimerAction(period=44.0, actions=[
        Node(package='gazebo_ros', executable='spawn_entity.py',
             name='spawn_robot', output='screen',
             arguments=['-entity', 'disaster_robot',
                        '-topic', 'robot_description',
                        '-x', LaunchConfiguration('x'),
                        '-y', LaunchConfiguration('y'),
                        '-z', '0.0',
                        '-Y', LaunchConfiguration('yaw'),
                        '-timeout', '60'])])

    # ── 3. EKF ────────────────────────────────────────────────────────
    ekf_node = TimerAction(period=48.0, actions=[
        Node(package='robot_localization', executable='ekf_node',
             name='ekf_filter_node', output='screen',
             parameters=[ekf_params_file, {'use_sim_time': True}])])

    # ── 4. SLAM ───────────────────────────────────────────────────────
    slam_node = TimerAction(period=70.0, actions=[
        Node(package='slam_toolbox',
             executable='async_slam_toolbox_node',
             name='slam_toolbox', output='screen',
             parameters=[slam_params_file, {'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('slam')))])

    # ── 5. Nav2 stack ─────────────────────────────────────────────────
    nav2_nodes = TimerAction(period=85.0, actions=[
        Node(package='nav2_controller', executable='controller_server',
             name='controller_server', output='screen',
             parameters=[nav2_params_file, {'use_sim_time': True}]),
        Node(package='nav2_planner', executable='planner_server',
             name='planner_server', output='screen',
             parameters=[nav2_params_file, {'use_sim_time': True}]),
        Node(package='nav2_behaviors', executable='behavior_server',
             name='behavior_server', output='screen',
             parameters=[nav2_params_file, {'use_sim_time': True}]),
        Node(package='nav2_bt_navigator', executable='bt_navigator',
             name='bt_navigator', output='screen',
             parameters=[nav2_params_file, {'use_sim_time': True}]),
        Node(package='nav2_lifecycle_manager',
             executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': ['planner_server',
                                         'controller_server',
                                         'behavior_server',
                                         'bt_navigator']}]),
    ])

    # ── 6. RViz2 ──────────────────────────────────────────────────────
    rviz_node = TimerAction(period=50.0, actions=[
        Node(package='rviz2', executable='rviz2', name='rviz2',
             output='screen', arguments=['-d', rviz_config_file],
             parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('use_rviz')))])

    # ── 7. Auto map saver ─────────────────────────────────────────────
    auto_map_saver = TimerAction(period=80.0, actions=[
        Node(package='disaster_sensors', executable='auto_map_saver',
             name='auto_map_saver', output='screen',
             parameters=[{'use_sim_time': True,
                          'output_dir': LaunchConfiguration('map_output_dir'),
                          'output_prefix': LaunchConfiguration('map_output_prefix'),
                          'save_interval': 30.0}])])

    # ── 8. Navigation policy (launched inside to share DDS on WSL2) ──
    # Each policy is conditionally launched based on nav_policy argument.
    policy_timer = 95.0  # after Nav2 lifecycle has time to activate

    frontier_node = TimerAction(period=policy_timer, actions=[
        Node(package='disaster_sensors', executable='frontier_explorer',
             name='frontier_explorer', output='screen',
             parameters=[{'use_sim_time': True}],
             condition=LaunchConfigurationEquals('nav_policy',
                                                 'frontier_explorer'))])

    potfield_node = TimerAction(period=policy_timer, actions=[
        Node(package='disaster_sensors',
             executable='potential_field_navigator',
             name='potential_field_navigator', output='screen',
             parameters=[{'use_sim_time': True}],
             condition=LaunchConfigurationEquals('nav_policy',
                                                 'potential_field_navigator'))])

    reactive_node = TimerAction(period=policy_timer, actions=[
        Node(package='disaster_sensors', executable='rl_navigator',
             name='rl_navigator', output='screen',
             parameters=[{'use_sim_time': True, 'mode': 'explore'}],
             condition=LaunchConfigurationEquals('nav_policy',
                                                 'rl_navigator'))])

    rl_model_node = TimerAction(period=policy_timer, actions=[
        Node(package='disaster_sensors', executable='rl_navigator',
             name='rl_navigator', output='screen',
             parameters=[{'use_sim_time': True, 'mode': 'navigate',
                          'model_path': LaunchConfiguration('rl_model_path')}],
             condition=LaunchConfigurationEquals('nav_policy',
                                                 'rl_navigator_model'))])

    # ── 9. EKF monitor (for collecting localisation + collision data) ─
    ekf_monitor_node = TimerAction(period=policy_timer, actions=[
        Node(package='disaster_sensors', executable='ekf_monitor',
             name='ekf_monitor', output='screen',
             parameters=[{'use_sim_time': True,
                          'output_ekf_csv': LaunchConfiguration('ekf_csv'),
                          'output_collision_csv': LaunchConfiguration('collision_csv')}],
             condition=IfCondition(LaunchConfiguration('launch_ekf_monitor')))])

    return LaunchDescription([
        LogInfo(msg=''),
        LogInfo(msg='╔══════════════════════════════════════════════╗'),
        LogInfo(msg='║   DISASTER ROBOT SIMULATOR — FULL LAUNCH     ║'),
        LogInfo(msg='╚══════════════════════════════════════════════╝'),
        LogInfo(msg=''),
        *args,
        set_model_path,
        gzserver, gzclient, robot_state_pub,
        delete_old_robot, spawn_robot,
        ekf_node, slam_node, nav2_nodes,
        rviz_node, auto_map_saver,
        frontier_node, potfield_node, reactive_node, rl_model_node,
        ekf_monitor_node,
    ])
