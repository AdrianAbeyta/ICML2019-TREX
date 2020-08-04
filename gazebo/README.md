# ICML2019 - Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations

## Preparing

- Install Robotic Operating System (ROS) for your supported machine. (Tested on ROS Melodic) 
```
http://wiki.ros.org/ROS/Installation
```

- Create a catkin workspace. More detail is given in the link below. 
```
http://wiki.ros.org/catkin/Tutorials/create_a_workspace
```
- Installing necessary packages in your catkin workspace. 
```
cd my_catkin_ws/src 
git clone https://bitbucket.org/theconstructcore/turtlebot.git
git clone https://bitbucket.org/theconstructcore/hokuyo_model.git
cd my_catkin_ws
catkin_make
source devel/setup.bash

```
## Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. This particular project requires ROS and Baselines integration of which both systems requre specific versions of python inorder to run correctly. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs 
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv: 
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/) 


## Installing Baselines In Your Virtual Enviornment

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 


## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
    
### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```
## Activate your virtual enviornment
You will need to activate your virtual enviormnet inorder to download the necessary python 3 modules within it. 

## Installation
- Once in your virtual envirnment: 
 ```
cd my_catkin_ws/src
 ```
 
Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, 
    ```bash 
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

## Training

- In a terminal.
```
roscore
```
- In a separate terminal.
```
roslaunch turtlebot_gazebo start_wall_world.launch
```
- In a separate terminal.
```
roslaunch turtlebot_gazebo put_turtlebot2_in_world.launch
```
- In a separate terminal from your virtual envirnment.

```
source /path/to/venv/bin/activate
cd /path/to/this/repo/.../gazebo/learner/baselines/baselines/deepq/experiments
python train_turtlebot2.py

```
