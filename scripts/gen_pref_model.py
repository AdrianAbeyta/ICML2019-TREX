#!/usr/bin/env python
import argparse
from pathlib import Path
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import numpy as np

# ROS packages
import rospy
import rospkg

def get_args():
       
    return args

# PPO Agent class. Functionally similar to stable_baselines PPO2 class.
class PPO2Agent(object):
    def __init__(self, env, path):

        # Initialize Tensorflow graph and configure GPU options
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():

                self.model_path = path
                self.model = PPO2.load(path)


    def act(self, obs, reward, done, env):

        # Based on current observation, generate next action and 
        with self.graph.as_default():
            with self.sess.as_default():
                action, states = self.model.predict(obs)

        return action


# TREX Preference Model
class PrefModel(object):
    def __init__(self,include_action,ob_dim,ac_dim,batch_size=64,num_layers=2,embedding_dims=256,steps=None):
        
        # Determine number of inputs (observation only, or observations plus action)
        self.include_action = include_action
        in_dims = ob_dim+ac_dim if include_action else ob_dim

        # Initialize graph parameters
        self.inp = tf.placeholder(tf.float32,[None,in_dims])
        self.x = tf.placeholder(tf.float32,[None,in_dims]) #[B*steps,in_dim]
        self.y = tf.placeholder(tf.float32,[None,in_dims])
        self.x_split = tf.placeholder(tf.int32,[batch_size]) # B-lengthed vector indicating the size of each steps
        self.y_split = tf.placeholder(tf.int32,[batch_size]) # B-lengthed vector indicating the size of each steps
        self.l = tf.placeholder(tf.int32,[batch_size]) # [0 when x is better 1 when y is better]
        self.l2_reg = tf.placeholder(tf.float32,[]) # [0 when x is better 1 when y is better]

        # Define num_layers linear layers
        with tf.variable_scope('weights') as param_scope:
            self.fcs = []
            last_dims = in_dims # input dimensions
            for l in range(num_layers):
                self.fcs.append(Linear('fc%d'%(l+1),last_dims,embedding_dims)) #(l+1) is gross, but for backward compatibility
                last_dims = embedding_dims
            self.fcs.append(Linear('fc%d'%(num_layers+1),last_dims,1))

        self.param_scope = param_scope

        # build graph
        def _reward(x):
            for fc in self.fcs[:-1]:
                x = tf.nn.relu(fc(x))
            r = tf.squeeze(self.fcs[-1](x),axis=1)
            return x, r

        self.fv, self.r = _reward(self.inp)

        _, rs_xs = _reward(self.x)
        self.v_x = tf.stack([tf.reduce_sum(rs_x) for rs_x in tf.split(rs_xs,self.x_split,axis=0)],axis=0)

        _, rs_ys = _reward(self.y)
        self.v_y = tf.stack([tf.reduce_sum(rs_y) for rs_y in tf.split(rs_ys,self.y_split,axis=0)],axis=0)

        logits = tf.stack([self.v_x,self.v_y],axis=1) #[None,2]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.l)
        self.loss = tf.reduce_mean(loss,axis=0)

        weight_decay = 0.
        for fc in self.fcs:
            weight_decay += tf.reduce_sum(fc.w**2)

        self.l2_loss = self.l2_reg * weight_decay

        pred = tf.cast(tf.greater(self.v_y,self.v_x),tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred,self.l),tf.float32))

        self.optim = tf.train.AdamOptimizer(1e-4)
        self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

        self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)
        
    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,D,batch_size=64,iter=128,l2_reg=0.01,noise_level=0.1,debug=False):
        """
        Training will be early-terminate when validation accuracy becomes large enough..?

        args:
            D: list of triplets (\sigma^1,\sigma^2,\mu)
            while
                sigma^{1,2}: shape of [steps,in_dims]
                mu : 0 or 1
        """
        sess = tf.get_default_session()

        idxes = np.random.permutation(len(D))
        train_idxes = idxes[:int(len(D)*0.8)]
        valid_idxes = idxes[int(len(D)*0.8):]

        def _batch(idx_list,add_noise):
            batch = []

            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            for i in idxes:
                batch.append(D[i])

            b_x,b_y,b_l = zip(*batch)
            x_split = np.array([len(x) for x in b_x])
            y_split = np.array([len(y) for y in b_y])
            b_x,b_y,b_l = np.concatenate(b_x,axis=0),np.concatenate(b_y,axis=0),np.array(b_l)

            if add_noise:
                b_l = (b_l + np.random.binomial(1,noise_level,batch_size)) % 2 #Flip it with probability 0.1

            return b_x,b_y,x_split,y_split,b_l

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_x,b_y,x_split,y_split,b_l = _batch(train_idxes,add_noise=True)

            loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.x_split:x_split,
                self.y_split:y_split,
                self.l:b_l,
                self.l2_reg:l2_reg,
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_x,b_y,x_split,y_split,b_l = _batch(valid_idxes,add_noise=False)
                    valid_acc = sess.run(self.acc,feed_dict={
                        self.x:b_x,
                        self.y:b_y,
                        self.x_split:x_split,
                        self.y_split:y_split,
                        self.l:b_l
                    })
                    tqdm.write(('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc)))


    def train_with_dataset(self,dataset,batch_size,include_action=False,iter=128,l2_reg=0.01,debug=False):
        sess = tf.get_default_session()

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_x,b_y,x_split,y_split,b_l = dataset.batch(batch_size=batch_size,include_action=include_action)
            loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.x_split:x_split,
                self.y_split:y_split,
                self.l:b_l,
                self.l2_reg:l2_reg,
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    tqdm.write(('loss: %f (l2_loss: %f), acc: %f'%(loss,l2_loss,acc)))

    def eval(self,D,batch_size=64):
        sess = tf.get_default_session()

        b_x,b_y,b_l = zip(*D)
        b_x,b_y,b_l = np.array(b_x),np.array(b_y),np.array(b_l)

        b_r_x, b_acc = [], 0.

        for i in range(0,len(b_x),batch_size):
            sum_r_x, acc = sess.run([self.sum_r_x,self.acc],feed_dict={
                self.x:b_x[i:i+batch_size],
                self.y:b_y[i:i+batch_size],
                self.l:b_l[i:i+batch_size]
            })

            b_r_x.append(sum_r_x)
            b_acc += len(sum_r_x)*acc

        return np.concatenate(b_r_x,axis=0), b_acc/len(b_x)

    def get_reward(self,obs,acs,batch_size=1024):
        sess = tf.get_default_session()

        if self.include_action:
            inp = np.concatenate((obs,acs),axis=1)
        else:
            inp = obs

        b_r = []
        for i in range(0,len(obs),batch_size):
            r = sess.run(self.r,feed_dict={
                self.inp:inp[i:i+batch_size]
            })

            b_r.append(r)

        return np.concatenate(b_r,axis=0)

class Linear(object) :
    def __init__(self,name,input_dim,output_dim,stddev=0.02) :
        with tf.variable_scope(name) :
            self.w = tf.get_variable('w',[input_dim, output_dim],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            self.b = tf.get_variable('b',[output_dim],
                                initializer=tf.constant_initializer(0.0))

    def __call__(self,input_var,name=None,w=None,b=None,**kwargs) :
        w = w if w is not None else self.w
        b = b if b is not None else self.b

        if( input_var.shape.ndims > 2 ) :
            dims = tf.reduce_prod(tf.shape(input_var)[1:])
            return tf.matmul(tf.reshape(input_var,[-1,dims]),w) + b
        else :
            return tf.matmul(input_var,w)+b
    def get_variables(self):
        return {'w':self.w,'b':self.b}

class GTDataset(object):
    def __init__(self,env):
        self.env = env
        self.unwrapped = env
        while hasattr(self.unwrapped,'env'):
            self.unwrapped = self.unwrapped.env

    def gen_traj(self,agent,min_length):
        # Initialize performance metric
        progress = -1 #TO DO: find a better progress metric - MuJoCo uses x-position displacement

        obs, actions, rewards = [self.env.reset()], [], []
        while True:
            rospy.logdebug('gen_traj while loop.')
            action = agent.act(obs[-1], None, None, self.env)
            ob, reward, done, _ = self.env.step(action)

            # Update performance metric with new maximum
            #if self.unwrapped.sim.data.qpos[0] > progress:
            #    progress = self.unwrapped.sim.data.qpos[0]

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

            if done:
                rospy.logdebug('Episode complete, length is ' + str(len(obs)) + '.')
                if len(obs) < min_length:
                    rospy.logdebug('Length below minimum trajectory length.')
                    obs.pop()
                    obs.append(self.env.reset())
                else:
                    rospy.logdebug('Length above minimum trajectory length.')
                    obs.pop()
                    break
            
        rospy.logdebug('Trajectory observations: ')
        rospy.logdebug(np.stack(obs,axis=0))
        rospy.logdebug('Trajectory actions: ')
        rospy.logdebug(len(actions))
        rospy.logdebug('Trajectory rewards: ')
        rospy.logdebug(np.array(rewards))


        # Combine the observations, actions, and rewards into a single python array of 3 numpy arrays.
        # Trajectory array has 3 numpy array elements: obs [n+1 x n_obs], actions [n x 1], and rewards [1 x n]  
        return (np.stack(obs,axis=0), np.stack(actions,axis=0), np.array(rewards)), progress

    def prebuilt(self,agents,min_length):

        rospy.logdebug('GTDataset.prebuilt() main')

        # Check to make sure at least one agent is provided
        assert len(agents)>0, 'no agent given'
        
        # Initialize empty trajectory list
        trajs = []

        # For each agent, propagate forward using gen_traj method
        for agent in tqdm(agents):
            rospy.logdebug('GTDataset.prebuilt() with agent ' + str(agent))
            traj, progress = self.gen_traj(agent,min_length)

            trajs.append(traj)
            tqdm.write('model: %s avg reward: %f progress: %f'%(agent.model_path,np.sum(traj[2]),progress))
        obs,actions,rewards = zip(*trajs)
        self.trajs = (np.concatenate(obs,axis=0),np.concatenate(actions,axis=0),np.concatenate(rewards,axis=0))

        print(self.trajs[0].shape,self.trajs[1].shape,self.trajs[2].shape)

    # Generate snippets of observation/action/reward dataset to train the preference model
    def sample(self,num_samples,steps,include_action=False):
        # Get all observations, actions, and rewards from trajectories
        obs, actions, rewards = self.trajs

        # Initialize empty data array
        D = []
        for _ in range(num_samples):
            print(len(obs))
            print(steps)
            x_ptr = np.random.randint(len(obs)-steps)
            y_ptr = np.random.randint(len(obs)-steps)

            if include_action:
                D.append((np.concatenate((obs[x_ptr:x_ptr+steps],actions[x_ptr:x_ptr+steps]),axis=1),
                         np.concatenate((obs[y_ptr:y_ptr+steps],actions[y_ptr:y_ptr+steps]),axis=1),
                         0 if np.sum(rewards[x_ptr:x_ptr+steps]) > np.sum(rewards[y_ptr:y_ptr+steps]) else 1)
                        )
            else:
                D.append((obs[x_ptr:x_ptr+steps],
                          obs[y_ptr:y_ptr+steps],
                          0 if np.sum(rewards[x_ptr:x_ptr+steps]) > np.sum(rewards[y_ptr:y_ptr+steps]) else 1)
                        )

        return D

# Backpropagate the reward function model to fit the trajectory data
def train(prefdir, env, chkptdir, min_length, include_action, num_models, steps, num_layers, embedding_dims, D_num_samples, l2_reg, noise):
    
    # If it's a discrete space
    if (type(env.action_space).__name__ == 'Discrete'):
        num_actions=env.action_space.n

    # If it's continuous
    elif (type(env.action_space).__name__ == 'Box'): 
        num_actions = env.action_space.shape[0]
    
    # If it's continuous
    if (type(env.observation_space).__name__ == 'Discrete'):
        num_observation= env.observation_space.n

    #If it's continuous
    elif (type(env.observation_space).__name__ == 'Box'):
        num_observation=env.observation_space.shape[0]

    # Convert directories to path objects
    pref_path = Path(prefdir)
    checkpoint_path = Path(chkptdir)

    # Check if model path exists already
    if pref_path.exists() :
        c = input('Log directory already exists. Continue to train a TREX preference model? [Y/etc]? ')
        if c in ['YES','yes','Y','y']:
            import shutil
            shutil.rmtree(str(pref_path))
        else:
            print('good bye')
            return

    # Create model path
    pref_path.mkdir(parents=True)
    with open(str(pref_path/'args.txt'),'w') as f:
        f.write( str(args) )
    
    pref_path = str(pref_path)

    # train_agents = [RandomAgent(env.action_space)] if args.random_agent else []
    train_agents=[] # Initialize empty list
    #models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) <= args.max_chkpt])
    models = sorted([p for p in checkpoint_path.glob('*.zip')])
    #print(models)
    for path in models:
        agent = PPO2Agent(env,str(path))
        train_agents.append(agent)

    #print(train_agents)

    # Initialize trajectory object type
    # preference_type == 'gt':
    dataset = GTDataset(env)

    # Generate actual trajectory data from trajectory object
    dataset.prebuilt(train_agents,min_length)

    ### Train Tensorflow TREX preference models
    # Initialize model objects

    models = []

    for i in range(num_models):

        with tf.variable_scope('model_%d'%i):
            rospy.logdebug('observation_space: ' + str(env.observation_space) + ', action_space: ' + str(env.action_space))
            rospy.logdebug('observation_space shape: ' + str(env.observation_space.shape) + ', action_space shape: ' + str(env.action_space.shape))
            models.append(PrefModel(include_action,num_observation,num_actions,steps=steps,num_layers=num_layers,embedding_dims=embedding_dims))# TODO: come back and fix the dimensions) 
    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    for i,model in enumerate(models):
        D = dataset.sample(num_samples=D_num_samples,steps=steps,include_action=include_action)

        if D is None:
            model.train_with_dataset(dataset,64,include_action=include_action,debug=True)
        else:
            model.train(D,l2_reg=l2_reg,noise_level=noise,debug=True)

        model.saver.save(sess,pref_path+'/model_%d.ckpt'%(i),write_meta_graph=False)

if __name__ == '__main__':

    # Initialize ROS node
    rospy.init_node('trex_checkpoints', anonymous=True, log_level=rospy.DEBUG)

    # Handle input arguments
    parser = argparse.ArgumentParser(description=None) # Create argument parser
    parser.add_argument('--namespace')
    args, unknown = parser.parse_known_args()
    
    # Init OpenAI_ROS Gym environment
    task_param_string = '/' + args.namespace + '/task_and_robot_environment_name'
    task_and_robot_environment_name = rospy.get_param(task_param_string)
    env=(StartOpenAI_ROS_Environment(task_and_robot_environment_name))
    rospy.loginfo("Gym environment done")

    # Load parameters from the ROS parameter server
    checkpoint_dir=rospy.get_param('/' + args.namespace + '/checkpoint_dir')
    pref_model_dir=rospy.get_param('/' + args.namespace + '/pref_model_dir')
    min_length=rospy.get_param('/' + args.namespace + '/min_length')
    include_action = rospy.get_param('/' + args.namespace + '/include_action')
    num_models=rospy.get_param('/' + args.namespace + '/num_models')
    steps =rospy.get_param('/' + args.namespace + '/steps')
    num_layers =rospy.get_param('/' + args.namespace + '/num_layers')
    embedding_dims =rospy.get_param('/' + args.namespace + '/embedding_dims')
    D_num_samples = rospy.get_param('/' + args.namespace + '/D')
    l2_reg = rospy.get_param('/' + args.namespace + '/l2_reg')
    noise = rospy.get_param('/' + args.namespace + '/noise')


    # Define directories where checkpoints are loaded from, and preference model is saved to (specified in /config/*.yaml file)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('trex_openai_ros') # Find package
    chkpt_dir = pkg_path + '/' + checkpoint_dir # Prepend package directory
    pref_dir = pkg_path + '/' + pref_model_dir # 

    # Generate log directory
    #pref_model_dir = Path(args.log_dir)

    # Load the checkpointed model(s) and train the preference model
    train(pref_dir, env, chkpt_dir, min_length, include_action, num_models, steps, num_layers, embedding_dims, D_num_samples, l2_reg, noise)

    # Close the environment when the preference model has been trained
    env.close()