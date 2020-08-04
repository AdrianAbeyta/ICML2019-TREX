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
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
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


# Baselines

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

## Training

- Example Script with the default hyperparameters
```
python preference_learning.py --env_id Hopper-v2 --env_type mujoco --learners_path ./learner/demo_models/hopper/checkpoints --max_chkpt 60 --num_models 5 --max_steps 40 --noise 0.0 --traj_noise 0 --num_layers 2 --log_dir ./log/hopper/max60/gt_traj_no_steps_noise/ --preference_type 'gt_traj_no_steps_noise' --custom_reward preference_normalized --ppo_log_path preference_norm_ctrl_coeff_011 --ctrl_coeff 0.1 --D 10000 --stochastic --rl_runs 5
```

### Eval

- In order to evaluation, just put `--eval` option at the end.
```
python preference_learning.py --env_id Hopper-v2 --env_type mujoco --learners_path ./learner/demo_models/hopper/checkpoints --max_chkpt 60 --num_models 5 --max_steps 40 --noise 0.0 --traj_noise 0 --num_layers 2 --log_dir ./log/hopper/max60/gt_traj_no_steps_noise/ --preference_type 'gt_traj_no_steps_noise' --custom_reward preference_normalized --ppo_log_path preference_norm_ctrl_coeff_011 --ctrl_coeff 0.1 --D 10000 --stochastic --rl_runs 5 --eval
```
