import torch
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from dqn.wrapper import wrap
from dqn.model import DQN
from dqn.agent import DQNAgent

env = env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4, render_mode='rgb_array')
env = wrap(env)
env = RecordVideo(env, 'out_pong_mar19')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


done = False
state, _ = env.reset()

state
dqn = DQN(state.shape, env.action_space.n)
dqn.load_state_dict(torch.load('out/dqn_res_mar19/dqn_state_dict_mar.pt', map_location=device))

agent = DQNAgent(
    dqn,
    0.,0.,0.,device
)

total_reward = 0.
while not done:

    a = agent.get_action(state.unsqueeze(0), 'greedy', env.action_space, 0)
    state, reward, terminated, truncated, info = env.step(a)
    total_reward += reward
    done = terminated or truncated
    env.render()

print(total_reward)
#env.close()