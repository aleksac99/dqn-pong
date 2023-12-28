import torch
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from dqn.wrapper import wrap
from dqn.model import DQN
from dqn.agent import DQNAgent

env = env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4, render_mode='rgb_array')
env = wrap(env)
env = RecordVideo(env, 'out_pong')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


done = False
state, _ = env.reset()

state
dqn = DQN(state.shape, env.action_space.n)
dqn.load_state_dict(torch.load('out_pong/dqn_state_dict_20.pt', map_location=device))

agent = DQNAgent(
    dqn,
    0.,0.,0.,device
)

while not done:

    a = agent.get_action(state.unsqueeze(0), 'greedy', env.action_space, 0)
    state, reward, terminated, truncated, info = env.step(a)
    done = terminated or truncated
    env.render()

#env.close()