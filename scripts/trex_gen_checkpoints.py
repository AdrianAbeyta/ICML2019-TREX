#!/usr/bin/env python
import gym
import numpy
import time
#import PPO2
import rospy
import rospkg
from gym import wrappers
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
#from openai_ros.task_envs.turtlebot3 import turtlebot3_world

# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from functools import reduce

if __name__ == '__main__':

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("walrus/alpha")
    Epsilon = rospy.get_param("walrus/epsilon")
    Gamma = rospy.get_param("walrus/gamma")
    epsilon_discount = rospy.get_param("walrus/epsilon_discount")
    nepisodes = rospy.get_param("walrus/nepisodes")
    nsteps = rospy.get_param("walrus/nsteps")
    tot_timesteps = rospy.get_param("walrus/total_timesteps")
    running_step = rospy.get_param("walrus/running_step")
    n_checkpoints = rospy.get_param("walrus/n_checkpoints")

    # Initialize ROS node
    rospy.init_node('trex_checkpoints', anonymous=True, log_level=rospy.DEBUG)

    # Initialize OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param('walrus/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)   

    # Initializes the model to be trained
    model = PPO2('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=tot_timesteps)

    env.close()