# Deep Q Learning - Pong

This repo contains a library for training a Deep Q Learning (DQN) agent which solves the game Pong. The game environment is provided from the Gymnasium library.

This project was a part of my Machine Learning class.

# Installation and usage

First, clone this repo from GitHub:

```shell
git clone https://github.com/aleksac99/dqn-pong/
```

This project is written as an installable library. To install it, run:

```shell
pip install .
```

Dependencies listed in `requirements.txt` will automatically be installed.

Finally, to start the training, run:

```shell
dqn-pong <CONFIG_FILE>
```

where `<CONFIG_FILE>` is a path to your configuration file. This argument is required.

## Run without installation

To run training without the installation of this package, first install the requirements manually:

```shell
pip install -r requirements.txt
```

Then, run the python module:

```shell
python -m dqn <CONFIG_FILE>
```

# Configuration files

There are two configuration files, for training and for simulation. Both configuration examples are located in the `configs` folder in the repo.

## Training configuration

Training configuration example is provided in `config.json`. **All fields in this file are required**. If you want to change configuration parameters, the recommendation is to copy the file and change parameters in the copy.

Training configuration file must contain the following parameters:

- `reward_goal`: Moving average reward goal. When the goal is reached, the training is finished
- `load_dqn_state_dict`: Path to saved model for training from checkpoint. Leave as `null` if training from scratch
- `gamma`: `gamma` parameter in training loss function
- `batch_size`: Training batch size
- `replay_memory_size`: Replay memory capacity
- `learning_rate`: Learning rate
- `epsilon_start`: Starting epsilon in epsilon-greedy policy
- `epsilon_end`: Final epsilon in epsilon-greedy policy
- `epsilon_decay_limit`: Number of steps required to reach `epsilon_end`
- `adam_beta1`: `beta1` parameter in Adam optimizer
- `adam_beta2`: `beta2` parameter in Adam optimizer
- `lr_scheduler_step_size`: Number of steps required to change learning rate
- `lr_scheduler_decay`: Multiplication factor for learning rate reduction
- `n_fixed_states`: Number of states to fix and use to estimate Q-values
- `max_n_epochs`: Maximum number of training epochs
- `target_dqn_update_after`: Number of steps required to update target network
- `ma_reward_n_episodes`: Number of episodes to use for moving average
- `out_dir`: Path to directory where to store output data
- `rewards_file`: Name of .txt file for storing reward logs
- `ma_rewards_file`: Name of .txt file for storing moving average reward logs
- `epsilons_file`: Name of .txt file for storing epsilon logs
- `fixed_states_q_file`: Name of .txt file for storing fixed states Q-value estimation logs
- `save_dqn_state_dict`: Name of .pt file for saving network state dict
- `dqn_type`: Specify which network to use. Possible values: `original`, `large`
- `difficulty`: Specify environment difficulty. Possible values: 0-3

# Results

This section will be written in the future.