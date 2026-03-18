cmake_minimum_required(VERSION 3.8)
project(disaster_robot)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf/   DESTINATION share/${PROJECT_NAME}/urdf)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/meshes")
  install(DIRECTORY meshes/ DESTINATION share/${PROJECT_NAME}/meshes)
endif()

ament_package()
