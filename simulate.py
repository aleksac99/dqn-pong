import torch
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo

from dqn.wrapper import wrap
from dqn.model import DQN
from dqn.agent import DQNAgent
from dqn.utils import Config, parse_args

# TODO: VECA MREZA: 9:06 + 16:37 - (213 epizoda za zagrevanje)

def main():

    args = parse_args()
    config = Config.from_json(args.config)

    simulate(config)


def simulate(config):

    env = env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4, render_mode='rgb_array')
    env = wrap(env)
    env = RecordVideo(env, config.out_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    state, _ = env.reset()
    dqn = DQN(state.shape, env.action_space.n)
    dqn.load_state_dict(torch.load(config.dqn_path, map_location=device))
    agent = DQNAgent(dqn, 0.01, 1, 0.01, device)
        
    done = False

    total_reward = 0.
    while not done:

        a = agent.get_action(state.unsqueeze(0), 'greedy', env.action_space, 0)
        state, reward, terminated, truncated, info = env.step(a)
        total_reward += reward
        done = terminated or truncated
        env.render()

    print(total_reward)

    env.close()

if __name__ == '__main__':
    
    main()