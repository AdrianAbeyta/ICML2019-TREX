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
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import results_plotter, PPO2, SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from stable_baselines.results_plotter import load_results, ts2xy
#from stable_baselines.common.vec_env import DummyVecEnv

#if __name__ == '__main__':

rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)

# Init OpenAI_ROS ENV
task_and_robot_environment_name = rospy.get_param(
    '/turtlebot3/task_and_robot_environment_name')

env=(StartOpenAI_ROS_Environment(task_and_robot_environment_name))
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
env._max_episode_steps =100000
rospy.loginfo("Monitor Wrapper started")
last_time_steps = numpy.ndarray(0)
# Loads parameters from the ROS param server
# Parameters are stored in a yaml file inside the config directory
# They are loaded at runtime by the launch file
Gamma = rospy.get_param("/turtlebot3/gamma")
nepisodes = rospy.get_param("/turtlebot3/nepisodes")
nsteps = rospy.get_param("/turtlebot3/nsteps")
running_step = rospy.get_param("/turtlebot3/running_step")
log_interval=rospy.get_param("turtlebot3/log_interval")
save_freq=rospy.get_param("turtlebot3/save_freq")
total_timesteps=rospy.get_param("turtlebot3/total_timesteps")

# Add some param noise for exploration
#param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)

# Initialises the algorithm that we are going to use for learning
#Sets up inital criteria for learning. 
model= PPO2(MlpPolicy, env,gamma=Gamma, n_steps=nsteps, verbose=1)


#Stop training when the model reaches the reward threshold
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

#Saves Model On desired Frequency
checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='/home/adrian/turtlebot_ws/src/openai_examples_projects/my_turtlebot3_openai_example/checkpoints/', 
    name_prefix='rl_ppo2_model')

#Ties all callbacks together to pass to learn policy. 
callback = CallbackList([eval_callback,checkpoint_callback])



#Initalized the learning algorithm 
model.learn(total_timesteps=total_timesteps,log_interval=log_interval, callback=callback)

start_time = time.time()
highest_reward = 0

# Starts the main training loop: the one about the episodes to do
for x in range(nepisodes):
    rospy.logdebug("############### START EPISODE=>" + str(x))
    cumulated_reward = 0
    dones = False
    #Initalizes the env and gets the first state. 
    obs = env.reset()
    state = ''.join(map(str, obs))
    i=1
    #Test the model for nsteps 
    #for i in range(nsteps):
    while True:
        
        rospy.logwarn("############### Start Step=>" + str(i))
       
       #Chooses an action based on the current state via prediction analysis. 
        action, _states = model.predict(obs)
        
        rospy.logdebug("Next action is:%d", action)

        #Preform the action in the env and get feedback.
        obs, rewards, dones, info = env.step(action)
        #env.render()

        rospy.logdebug(str(obs) + " " + str(rewards))
        
        cumulated_reward += rewards
        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        nextState = ''.join(map(str, obs))

        # Make the algorithm learn based on the results
        rospy.logdebug("# state we were=>" + str(state))
        rospy.logdebug("# action that we took=>" + str(action))
        rospy.logdebug("# reward that action gave=>" + str(rewards))
        rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
        rospy.logdebug("# State in which we will start next step=>" + str(nextState))
        

        if not dones:
            #Keeps track of steps
            i= i+1
            rospy.logdebug("NOT DONE")
            states = nextState

        if dones:
            rospy.logdebug("DONE")
            last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
            break
        

        rospy.logwarn("############### END Step=>" + str(i))
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    l = last_time_steps.tolist()
    l.sort()

    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))


#Results of the reward function vs timesteps will be ploted once a mean reward above threshold is established. 
#results_plotter.plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "MLP Turtlebot3 PPO2 Reward")
#plt.show() 

#Saves the final model  
model.save("turtlebot3_final_reward", cloudpickle=False)
print("Training Is Complete!") 

#Exits the env. 
env.close()
