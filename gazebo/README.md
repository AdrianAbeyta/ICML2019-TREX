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
