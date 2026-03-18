# Disaster Navigation Robot — Autonomous Navigation with Multi-Modal Sensor Fusion and Reinforcement Learning

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Gazebo Classic](https://img.shields.io/badge/Gazebo-Classic%2011-orange)](https://classic.gazebosim.org/)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Disaster Navigation Robot is a complete ROS 2 simulator for search-and-rescue robots in collapsed building environments. It will includes sensor fusion using an Extended Kalman Filter, real-time mapping (SLAM), and autonomous navigation using reinforcement learning in a collapsed building environment.

The robot will explore unknown rubble areas on its own, create maps of the surroundings, and show that learned navigation works better than simple reactive methods.

---

## Overview

After an earthquake, collapsed buildings create very dangerous conditions for human rescue teams. This project builds an autonomous ground robot that can help in such situations.

The robot is designed to:

Navigate through rubble without needing a pre-built map

Find its position (localize) accurately even when wheels slip, using EKF sensor fusion

Create maps in real time using LiDAR-based SLAM

Explore on its own using either a simple reactive method or a trained PPO (reinforcement learning) agent
