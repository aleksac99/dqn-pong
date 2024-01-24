import os
from copy import deepcopy
from collections import deque
import torch

from dqn.utils import Logger


class Trainer:

    def __init__(
            self,
            env,
            agent,
            optimizer,
            criterion,
            lr_scheduler,
            logger: Logger,
            gamma,
            batch_size,
            replay_memory_size,
            device) -> None:
        
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.replay_memory = deque(maxlen=replay_memory_size)


        self.total_rewards = []
        self.moving_average_rewards = []
        self.fixed_states = []
        self.fixed_states_q = []
        self.epsilons = []
        self.best_rewards = []

    def init_memory_fixed_states(self, n_states):

        new_state, info = self.env.reset()

        for _ in range(self.replay_memory.maxlen):

            cur_state = new_state.unsqueeze(0)

            with torch.no_grad():
                action = self.agent.get_action(cur_state, method='random', actions=self.env.action_space, time=0)

            new_state, reward, terminated, truncated, info = self.env.step(action)

            self.replay_memory.append({
                'state': cur_state[0],
                'action': action,
                'reward': reward,
                'next_state': new_state,
                'terminal': terminated or truncated
            })

            if terminated or truncated:
                new_state, info = self.env.reset()

        
        fixed_states_idx = torch.randperm(len(self.replay_memory))[:n_states]
        fixed_states = [self.replay_memory[idx] for idx in fixed_states_idx]
        self.fixed_states = [sample['state'] for sample in fixed_states]

    def calc_mean_max_q_on_fixed_states(self):

        states = torch.stack(self.fixed_states).to(self.device)
        with torch.no_grad():
            qs = self.agent.dqn(states)
        return (qs.max(dim=1)[0]).mean().item()
            
    def fit(
            self,
            max_n_episodes,
            target_dqn_update_after,
            ma_reward_n_episodes,
            dqn_state_dict_name):

        best_ma_reward = -1e9
        time = 0

        for episode in range(max_n_episodes):

            total_reward = 0.
            next_state, info = self.env.reset()

            while True:

                time += 1

                cur_state = next_state.unsqueeze(0)
                with torch.no_grad():
                    action = self.agent.get_action(cur_state, method='eps_greedy', actions=self.env.action_space, time=time)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward

                self.replay_memory.append({
                'state': cur_state[0],
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'terminal': terminated or truncated})

                mini_batch_idx = torch.randperm(len(self.replay_memory))[:self.batch_size]
                mini_batch = [self.replay_memory[idx] for idx in mini_batch_idx]

                actions = [sample['action'] for sample in mini_batch]
                cur_states = torch.stack([sample['state'] for sample in mini_batch], dim=0).to(self.device)
                next_states = torch.stack([sample['next_state'] for sample in mini_batch], dim=0).to(self.device)
                rewards = torch.tensor([sample['reward'] for sample in mini_batch]).to(self.device)
                terminals = ~torch.tensor([sample['terminal'] for sample in mini_batch]).to(self.device)

                with torch.no_grad():
                    targets = (rewards + self.gamma * torch.max(self.agent.target_dqn(next_states), 1)[0] * terminals).detach()

                preds = self.agent.dqn(cur_states)[torch.arange(self.batch_size), actions]
                
                loss = self.criterion(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if time % target_dqn_update_after == 0:
                    self.agent.target_dqn = deepcopy(self.agent.dqn).to(self.device)
                    self.agent.target_dqn.eval()

                if terminated or truncated:
                    break

            self.lr_scheduler.step()

            mean_max_q = self.calc_mean_max_q_on_fixed_states()
            self.fixed_states_q.append(mean_max_q)
            self.logger.log_fixed_states_q(self.fixed_states_q)

            self.total_rewards.append(total_reward)
            self.logger.log_rewards(self.total_rewards)

            moving_average_reward = sum(self.total_rewards[-ma_reward_n_episodes:]) / min(len(self.total_rewards), ma_reward_n_episodes)
            self.moving_average_rewards.append(moving_average_reward)
            self.logger.log_ma_rewards(self.moving_average_rewards)

            self.epsilons.append(self.agent.get_epsilon(time))
            self.logger.log_epsilons(self.epsilons)

            print(f'Time: {time} | Episode {episode+1:3}: | Reward: {total_reward:.3f} | Moving average reward: {moving_average_reward:.3f} | Epsilon: {self.agent.get_epsilon(time):.3f} | Mean max q: {mean_max_q}')

            if moving_average_reward > best_ma_reward:

                best_ma_reward = moving_average_reward

                torch.save(
                    self.agent.dqn.state_dict(),
                    os.path.join(self.logger.base_dir, dqn_state_dict_name))

            if moving_average_reward > 20.:

                torch.save(
                    self.agent.dqn.state_dict(),
                    os.path.join(self.logger.base_dir, dqn_state_dict_name))
                
                return self.total_rewards, best_ma_reward, (episode + 1)

        return self.total_rewards, best_ma_reward, (episode + 1)