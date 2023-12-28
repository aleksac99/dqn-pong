import torch
import gymnasium as gym
from dqn.wrapper import wrap
from dqn.model import DQN
from dqn.agent import DQNAgent
from dqn.trainer import Trainer
from torch.optim import Adam
from torch.nn import MSELoss

# Parameters
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10_000
LEARNING_RATE = 1e-4

EPSILON_DECAY = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Device: {device}')

env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4)
env = wrap(env)

state, info = env.reset()
agent = DQNAgent(
    DQN(state.shape, env.action_space.n),
    EPSILON_START, EPSILON_DECAY, EPSILON_END, device=device)
optimizer = Adam(agent.dqn.parameters(), LEARNING_RATE, (0.9, 0.999))
criterion = MSELoss(reduction='mean').to(device)
trainer = Trainer(env, agent, optimizer, criterion, GAMMA, BATCH_SIZE, REPLAY_SIZE, device)

trainer.init_memory()

# env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4, render_mode='human')
# env = wrap(env)
# trainer.env = env

print(trainer.fit(5))