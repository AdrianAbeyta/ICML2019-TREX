# ICML2019-TREX (openai_ros)

This is a ROS implementation of the TREX code described in the master branch. While the master branch is designed to improve performance of RL agents in the Atari and Mujoco environments, this branch extends TREX functionality to the domain of ROS robots. It does this by using the openai_ros package, originally developed by The Construct and modified for the University of Texas' robots.

There are a few changes from the original code, notably:
- RL and preference model training & evaluation are performed sequentially by different scripts
- The use of ROS parameters instead of arguments
- The use of stable_baselines instead of openai-baselines
- Creation of custom ROS environments for Gym, which are loaded into the Gym environment register at runtime
- General updates to use more recent versions of Tensorflow, Gym, and other supporting software

# Prerequisites
This assumes you have the following software installed:
- Ubuntu 20.04
- ROS Noetic (required since TREX uses Python 3 and Noetic is the only Python 3 version of ROS 1)
- Conda (optional, but recommended to ensure Python version compatibility and package installation)

This software was run using NVidia driver 440.100, CUDA 10.2 on a GeForce RTX 2060 Super 8GB.

# Installation

NOTE: The UTBox folders and Github repos may require special access.

If you can't get access to UTBox links or UT Nuclear Robotics Github, please contact Dr. Pryor. For access to this (or any other Github repo), please contact me.

## Copy Gazebo models into your model folder
First, copy over the gazebo model/mesh files from the cloud drive here: https://utexas.app.box.com/folder/121632471058

Copy the /Stairs and /ut_mesh folders into your ~/.gazebo/models folder.

For example, the completed file path should be ~/.gazebo/models/ut_mesh

## Create & activate Conda environment (Optional but recommended)
In a terminal:
```
cd ~/trex_ros_ws/src
conda create --name TREX python=3.6 setuptools=45 numpy=1.16
conda activate TREX
```

## Install required Python packages
In a terminal:
```
pip install catkin_pkg
pip install empy
pip install Cython==0.27.3
pip install cffi==1.11.5
```

## Create a catkin workspace, clone packages, and build
```
cd ~
mkdir -p trex_ros_ws/src
cd ~/trex_ros_ws/src
git clone https://github.com/roboav8r/openai_ros.git
git clone -b openai --single-branch https://github.com/UTNuclearRobotics/walrus_description
git clone -b openai --single-branch https://github.com/UTNuclearRobotics/walrus_gazebo
git clone -b openai_ros --single-branch https://github.com/roboav8r/ICML2019-TREX.git
cd ~/trex_ros_ws/src

catkin build
```

## Clone Turtlebot3 model and environment
If you desire to use the Turtlebot3 Burger by ROBOTIS for simulation please clone and build these in your catkin workspace. 
```
cd ~
cd trex_ros_ws/src
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ..
catkin_make
```


## Install remaining Python packages
In a terminal:
```
cd trex_ros_ws/src
pip3 install -r requirements.txt
```

# Usage
TREX (openai_ros) uses ROS to initiate all trainings via a launch file. Each launch file contains the scripts and configuration paramaters necessary for operation. Configurations for modifying variables can be can be found in the config folder along with a description of each variable. 

## Generate checkpointed RL models
Note: The example displayed will demonstrate learned obsticale avoidance of the Tutrlebot3 Burger by ROBOTIS. For demonstration of obsticale avoidance via the Walrus platform, please edit the env_id in the configurations to (INSERT WALRUS ENV_ID). 

- In order to train the TREX preference model, you first must develop checkpointed RL models for use as demonstrations. Stable-baselines Proximal Policy Optimization (PP02) was used in combination with MLP in order to generate the checkpoints. In the terminal of your choice, use the roslaunch command to begin training:

```
  $ roslaunch gen_checkpoints_ppo2.launch
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

# Known Issues / Potential Improvements / Future Work

- Currently, the preference model only accepts GTraj trajectory types. This could be expanded to The other trajectory types in mujoco/preference_learning.py 
- Occasional simulation robot reset issues during training.
- Set up the git repo to add other repositories as submodules, so that a git clone --recursive of this repo is the only clone command needed.
- Alternate meshes of the UT campus can be found in https://utexas.app.box.com/folder/121689530768. These meshes feature fewer polygons for quicker loading, and have color as well. There is also a Pandas datafile which contains a lookup table to determine the terrain type for a given (x,y) position on the mesh. Ideally, you could get your current odometry (i.e. x,y pose) in gazebo, query the Pandas database with your (x,y), and determine the terrain classification. We wanted to implement this function but ran out of time in Summer 2020. The enclosed .boxnote contains more information.


