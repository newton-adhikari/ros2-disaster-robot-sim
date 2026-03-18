cmake_minimum_required(VERSION 3.8)
project(disaster_world)

find_package(ament_cmake REQUIRED)

# Install world files
install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)


# Set GAZEBO_MODEL_PATH so Gazebo can find our custom models at launch
ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/disaster_world.dsv.in"
)

ament_package()
