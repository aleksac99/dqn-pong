import os
from datetime import datetime
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
import gymnasium as gym

from dqn.wrappers import wrap
from dqn.models import DQNOriginal, DQNLarge
from dqn.agent import DQNAgent
from dqn.trainer import Trainer
from dqn.utils import parse_args, Config, Logger

def main():

    args = parse_args()
    config = Config.from_json(args.config)
    
    os.makedirs(config.out_dir, exist_ok=True)

    logger = Logger(
        config.out_dir,
        config.rewards_file,
        config.ma_rewards_file,
        config.epsilons_file,
        config.fixed_states_q_file)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'Device: {device}')

    env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4, difficulty=config.difficulty)
    env = wrap(env)

    state, info = env.reset()

    dqn = DQNOriginal(state.shape, env.action_space.n) if config.dqn_type=='original' else DQNLarge(state.shape, env.action_space.n)
    print(dqn)

    if config.load_dqn_state_dict is not None:
        dqn.load_state_dict(torch.load(config.dqn_state_dict, map_location=device))
        print(f'Successfully loaded {config.load_dqn_state_dict}')
    else:
        print(f'No checkpoint provided. Training from scratch.')

    agent = DQNAgent(
        dqn,
        config.epsilon_start,
        config.epsilon_decay_limit,
        config.epsilon_end,
        device)
    
    optimizer = Adam(
        agent.dqn.parameters(),
        config.learning_rate,
        (config.adam_beta1, config.adam_beta2))
    
    criterion = MSELoss(reduction='mean').to(device)

    lr_scheduler = StepLR(
        optimizer,
        step_size=config.lr_scheduler_step_size,
        gamma=config.lr_scheduler_decay)
    
    trainer = Trainer(
        env,
        agent,
        optimizer,
        criterion,
        lr_scheduler,
        logger,
        config.gamma,
        config.batch_size,
        config.replay_memory_size,
        device)
    
    trainer.init_memory_fixed_states(config.n_fixed_states)

    training_start = datetime.now()
    print('Started training at:')
    print(training_start.strftime("%d/%m/%Y %H:%M:%S"))

    _, best_reward, episode = trainer.fit(
        config.max_n_epochs,
        config.target_dqn_update_after,
        config.ma_reward_n_episodes,
        config.save_dqn_state_dict)

    print(f'Done in {episode} episodes | Best reward: {best_reward}')

    training_end = datetime.now()
    print('Finished training at:')
    print(training_end.strftime("%d/%m/%Y %H:%M:%S"))

    print('Time training:')
    print(training_end - training_start)

if __name__=='__main__':
    
    main()