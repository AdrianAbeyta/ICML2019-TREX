#!/usr/bin/env python

#Standard Packages 
import os
import matplotlib.pyplot as plt
import numpy
import time
#ROS Packages
import rospy
import rospkg
#Openai_ros,stable baselines,and gym packages
import gym
from gym import wrappers
from stable_baselines.common.callbacks import CheckpointCallback, EveryNTimesteps, BaseCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import results_plotter, PPO2, SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from stable_baselines.results_plotter import load_results, ts2xy
#from stable_baselines.common.vec_env import DummyVecEnv

if __name__ == '__main__':

    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    
    env=(StartOpenAI_ROS_Environment(task_and_robot_environment_name)).env
    #env = Monitor(env, log_dir)

    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    # Add relative path of folder you want checkpoints to save too. 
    log_dir = "/home/adrian/turtlebot_ws/src/openai_examples_projects/my_turtlebot3_openai_example/checkpoints/"
    os.makedirs(log_dir, exist_ok=True)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    outdir = log_dir + '/training_results'
    
    #Way in which to visualize the training. 
    env = Monitor(env, outdir,log_dir)
    rospy.loginfo("Monitor Wrapper started")

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Gamma = rospy.get_param("/turtlebot3/gamma")
    #nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")
    running_step = rospy.get_param("/turtlebot3/running_step")
    log_interval=rospy.get_param("turtlebot3/log_interval")
    save_freq=rospy.get_param("turtlebot3/save_freq")
    total_timesteps=rospy.get_param("turtlebot3/total_timesteps")
    
    # Add some param noise for exploration
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)

    # Initialises the algorithm that we are going to use for learning
    #Sets up inital criteria for learning. 
    PPO2= PPO2(MlpPolicy, env,gamma=Gamma, n_steps=nsteps, verbose=1)

    
    #Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    #Saves Model On desired Frequency
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='/home/adrian/turtlebot_ws/src/openai_examples_projects/my_turtlebot3_openai_example/checkpoints/', 
        name_prefix='rl_ppo2_model')

    #Ties all callbacks together to pass to learn policy. 
    callback = CallbackList([eval_callback,checkpoint_callback])


    
    #Initalized the learning algorithm 
    PPO2.learn(total_timesteps=int(total_timesteps),log_interval=int(log_interval), callback=callback)

    #Results of the reward function vs timesteps will be ploted once a mean reward above threshold is established. 
    results_plotter.plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "MLP Turtlebot3 PPO2 Reward")
    plt.show() 

    #Saves the final model  
    PPO2.save("turtlebot3_final_reward", cloudpickle=False)
    print("Training Is Complete! :)") 
    
    #Exits the env. 
    env.close()