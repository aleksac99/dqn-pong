# Deep Q Learning - Pong

This repo contains a library for training a Deep Q Learning (DQN) agent which solves the game Pong. The game environment is provided from the Gymnasium library.

This project was a part of my Machine Learning class.

# Installation and usage

First, clone this repo from GitHub:

```shell
git clone https://github.com/aleksac99/dqn_test/
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

# Configuration file

Configuration example is provided in `config.json`. **All fields in this file are required**. If you want to change configuration parameters, the recommendation is to copy the file and change parameters in the copy.

The explanation of config parameters will be written in the future.

# Results

This section will be written in the future.