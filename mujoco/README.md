# ICML2019 - Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations

## Configuration
This version was tested on a machine with:
 - Ubuntu 20.04
 - NVidia GeForce RTX 2060 Super (8 GB) GPU, Driver version 440.1
 - CUDA 10.1
 - cuDNN 7.6.5

## Install MuJoCo
- Obtain a 30-day free trial on the MuJoCo website (https://www.roboti.us/license.html) or free license if you are a student. The license key will arrive in an email with your account number.
- Download the getid_linux script from https://www.roboti.us/getid/getid_linux. By default, it should be in your Linux ~/Downloads folder. Now run the script. At a console:
```
  cd ~/Downloads
  chmod +x getid_linux
  ./getid_linux
  ```
  This will generate a computer ID.

- Go back to the MuJoCo license website (https://www.roboti.us/license.html) and scroll to the "Register Computer" block. Input your account number from Step 1 and your Computer ID from Step 2. You will be e-mailed an activation key file (mjkey.txt).

- Download the MuJoCoPro 1.5 binaries for Linux (https://www.roboti.us/download/mjpro150_linux.zip). Unzip the downloaded mjpro150 directory into ~/.mujoco/mjpro150, and place your license key (mjkey.txt) in ~/.mujoco/mjkey.txt.
- Lastly, add the following line to your ~/.bashrc
`export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH`

## Configuring Environment for TREX & Installing
### Conda Installation
If you haven't installed Conda yet, follow the instructions here:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
### Setting up the TREX Conda environment
- Create a Conda environment that uses the required versions of Python, NumPy, and setuptools. At a terminal:

`conda create --name TREX python=3.6 setuptools=45 numpy=1.16`
- Activate the TREX Conda environment. At a terminal:

`conda activate TREX`
### Installing TREX
- With the Conda environment active, at a terminal:
```
pip3 install Cython==0.27.3
pip3 install cffi==1.11.5
pip3 install -r requirements.txt
```

- unzip the checkpointed policies ([download here](https://github.com/dsbrown1331/learning-rewards-of-learners/releases/download/mujoco/mujoco_models.tar.gz)) used for learning and evaluation.
```
cd learner/demo_models
tar xzvf mujoco_models.tar.gz
```

## Training

- Example Script with the default hyperparameters
```
python3 preference_learning.py --env_id Hopper-v2 --env_type mujoco --learners_path ./learner/demo_models/hopper/checkpoints --max_chkpt 60 --num_models 5 --max_steps 40 --noise 0.0 --traj_noise 0 --num_layers 2 --log_dir ./log/hopper/max60/gt_traj_no_steps_noise/ --preference_type 'gt_traj_no_steps_noise' --custom_reward preference_normalized --ppo_log_path preference_norm_ctrl_coeff_011 --ctrl_coeff 0.1 --D 10000 --stochastic --rl_runs 5
```

### Eval

- In order to evaluation, just put `--eval` option at the end.
```
python preference_learning.py --env_id Hopper-v2 --env_type mujoco --learners_path ./learner/demo_models/hopper/checkpoints --max_chkpt 60 --num_models 5 --max_steps 40 --noise 0.0 --traj_noise 0 --num_layers 2 --log_dir ./log/hopper/max60/gt_traj_no_steps_noise/ --preference_type 'gt_traj_no_steps_noise' --custom_reward preference_normalized --ppo_log_path preference_norm_ctrl_coeff_011 --ctrl_coeff 0.1 --D 10000 --stochastic --rl_runs 5 --eval
```
