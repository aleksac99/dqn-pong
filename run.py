import torch
import gymnasium as gym
from dqn.wrapper import wrap
from dqn.model import DQN
from dqn.agent import DQNAgent
from dqn.trainer import Trainer
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
from datetime import datetime

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Device: {device}')

env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4)
env = wrap(env)

state, info = env.reset()
dqn = DQN(state.shape, env.action_space.n)

if DQN_STATE_DICT is not None:
    dqn.load_state_dict(torch.load(DQN_STATE_DICT, map_location=device))
    # TODO: DELETE THESE PARAMETER CHANGES
    LEARNING_RATE = 1e-5
    EPSILON_START = 0.01

agent = DQNAgent(
    dqn,
    EPSILON_START, EPSILON_DECAY, EPSILON_END, device=device)
optimizer = Adam(agent.dqn.parameters(), LEARNING_RATE, (ADAM_BETA1, ADAM_BETA2))
criterion = MSELoss(reduction='mean').to(device)
lr_scheduler = StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_DECAY)
trainer = Trainer(env, agent, optimizer, criterion, lr_scheduler, GAMMA, BATCH_SIZE, REPLAY_SIZE, device)

trainer.init_memory_fixed_states(N_FIXED_STATES)

training_start = datetime.now()
print('Started training at:')
print(training_start.strftime("%d/%m/%Y %H:%M:%S"))

_, best_reward, episode = trainer.fit(MAX_N_EPOCHS)

print(f'Done in {episode} episodes | Best reward: {best_reward}')

training_end = datetime.now()
print('Finished training at:')
print(training_end.strftime("%d/%m/%Y %H:%M:%S"))

print('Time training:')
print(training_end - training_start)