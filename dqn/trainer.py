from copy import deepcopy
from collections import deque
import torch


class Trainer:

    def __init__(self, env, agent, optimizer, criterion, gamma, batch_size, memory_capacity, device) -> None:
        
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.criterion = criterion
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.replay_memory = deque(maxlen=memory_capacity)

    def init_memory(self):

        new_state, info = self.env.reset()

        for _ in range(self.replay_memory.maxlen):

            cur_state = new_state.unsqueeze(0)

            action = self.agent.get_action(cur_state, method='random', actions=self.env.action_space, episode=0)

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
            
    def fit(self, n_episodes):

        total_rewards = []
        best_reward = -1e9

        for episode in range(n_episodes):

            total_reward = 0.
            next_state, info = self.env.reset()

            time = 0

            while True:

                time += 1

                cur_state = next_state.unsqueeze(0)
                action = self.agent.get_action(cur_state, method='eps_greedy', actions=self.env.action_space, episode=episode) # NOTE: with torch.no_grad()?
                next_state, reward, terminated, truncated, info = self.env.step(action)
                #self.env.render()
                total_reward += reward

                self.replay_memory.append({
                'state': cur_state[0],
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'terminal': terminated or truncated}) # NOTE: Maybe terminated only?

                mini_batch_idx = torch.randperm(len(self.replay_memory))[:self.batch_size]
                mini_batch = [self.replay_memory[idx] for idx in mini_batch_idx]

                actions = [sample['action'] for sample in mini_batch]
                cur_states = torch.stack([sample['state'] for sample in mini_batch], dim=0).to(self.device)
                next_states = torch.stack([sample['next_state'] for sample in mini_batch], dim=0).to(self.device)
                rewards = torch.tensor([sample['reward'] for sample in mini_batch]).to(self.device)
                terminals = ~torch.tensor([sample['terminal'] for sample in mini_batch]).to(self.device)

                with torch.no_grad():
                    #targets = (rewards + self.gamma * torch.max(self.agent.dqn(next_states), 1)[0] * terminals).detach()
                    targets = (rewards + self.gamma * torch.max(self.agent.target_dqn(next_states), 1)[0] * terminals).detach()

                preds = self.agent.dqn(cur_states)[torch.arange(self.batch_size), actions]
                
                loss = self.criterion(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if time % 1_000 == 0:
                    self.agent.target_dqn = deepcopy(self.agent.dqn).to(self.device)
                    self.agent.target_dqn.eval()

                if terminated or truncated:
                    break

            total_rewards.append(total_reward)
            with open('total_rewards.txt', 'w') as f:
                f.writelines([str(r) for r in total_rewards])

            print(f'Episode {episode+1:3}: | Reward: {total_reward:.3f} | Moving average reward: {sum(total_rewards[-100:])/min(len(total_rewards), 100):.3f} | Epsilon: {self.agent.get_epsilon(episode):.3f}')

            if total_reward > best_reward:
                best_reward = total_reward
                # self.agent.target_dqn = deepcopy(self.agent.dqn).to(self.device)
                # self.agent.target_dqn.eval()
                # for p in self.agent.target_dqn.parameters():
                #     p.requires_grad = False
                torch.save(self.agent.dqn.state_dict(), 'dqn_state_dict.pt')

        return total_rewards


    def eval(self, n_episodes):
        pass