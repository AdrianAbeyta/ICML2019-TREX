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
Note: The example displayed will demonstrate learned obsticale avoidance of the Tutrlebot3 Burger by ROBOTIS. For demonstration of obsticale avoidance via the Walrus platform, please edit the env_id in the configurations to (INSERT WALRUS ENV_ID). 

- Inorder to train the TREX preference model you first must develop checkpointed RL models for use as demonstrations. Stable-baselines Proximal Policy Optimization (PP02) was used in combination with MLP inorder to generate the checkpoints. In the terminal of your choice, use the roslaunch command to begin training:

```
  $ roslaunch start_training_turtlebot3_ppo2.launch
```
Gazebo should appear and a visual representation of training will begin. The checkpoints will be saved in the designated save_path set in the configurations.

## Train the TREX preference model
- Once checkpoints are generated, a neural network will be created from these checkpoints to approximate a learned reward function. In the terminal of your choice, use the roslaunch command to begin training:

```
  $ roslaunch pref_model.launch
```
The checkpoints will be loaded based on the designated save_path set in the launch file. The learned reward function will be saved in the designated log_dir set in the launch file. 

## Use the TREX preference model to train a new RL model
- The Stable-baselines Proximal Policy Optimization (PP02) RL model will be trained once again. However, the RL model will now use the T-REX neural network learned reward rather than the true enviornment reward. In the terminal of your choice, use the roslaunch command to begin training:

```
  $ roslaunch TODO
```
The checkpoints will be loaded based on the designated save_path set in the launch file. The learned policy will be saved in the designated (TODO) set in the launch file. 

# Known Issues / Potential Improvements
Only uses GTraj trajectory types
