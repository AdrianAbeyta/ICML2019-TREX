# ICML2019-TREX (openai_ros)

This is a ROS implementation of the TREX code described in the master branch. While the master branch is designed to improve performance of RL agents in the Atari and Mujoco environments, this branch extends TREX functionality to the domain of ROS robots. It does this by using the openai_ros package, originally developed by The Construct and modified for the University of Texas' robots.

There are a few changes from the original code, notably:
- RL and preference model training & evaluation are performed sequentially by different scripts
- The use of ROS parameters instead of arguments
- The use of stable_baselines instead of openai-baselines
- Creation of custom ROS environments for Gym, which are loaded into the Gym environment register at runtime
- General updates to use more recent versions of Tensorflow, Gym, and other supporting software.

# Prerequisites
This assumes you have the following software installed:
- Ubuntu 20.04
- ROS Noetic (required since TREX uses Python 3 and Noetic is the only Python 3 version of ROS 1)
- [TO DO] driver versions, NVidia / tensorflow, cuda, ...

# Installation

Copy over the stairs and UT meshes into the ~/.gazebo/models folder

make a catkin workspace

clone this, openai_ros, walrus_gazebo and walrus_description

# Usage
TREX (openai_ros) uses ROS to initiate all trainings via a launch file. Each launch file contains the scripts and configuration paramaters necessary for operation. Configurations for modifying variables can be can be found in the config folder along with a description of each variable. 

## Generate checkpointed RL models

## Train the TREX preference model
[TO DO] This generates a neural network to approximate the reward function and...

## Use the TREX preference model to train a new RL model

# Known Issues / Potential Improvements
Only uses GTraj trajectory types
