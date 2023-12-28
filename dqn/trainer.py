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


        self.total_rewards = []
        self.mean_average_rewards = []
        self.fixed_states = []
        self.fixed_states_q = []
        self.epsilons = []
        self.best_rewards = []

    def init_memory_fixed_states(self, n_states):

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

        
        fixed_states_idx = torch.randperm(len(self.replay_memory))[:n_states]
        fixed_states = [self.replay_memory[idx] for idx in fixed_states_idx]
        self.fixed_states = [sample['state'] for sample in fixed_states]

    def calc_mean_max_q_on_fixed_states(self):

        states = torch.stack(self.fixed_states)
        with torch.no_grad():
            qs = self.agent.dqn(states)
        return (qs.max(dim=1)[0]).mean().item()
            
    def fit(self, n_episodes):

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

            mean_max_q = self.calc_mean_max_q_on_fixed_states()
            self.fixed_states_q.append(mean_max_q)
            with open('fixed_states_q.txt', 'w') as f:
                f.write("\n".join([str(r) for r in self.fixed_states_q]))

            self.total_rewards.append(total_reward)
            with open('total_rewards.txt', 'w') as f:
                f.write("\n".join([str(r) for r in self.total_rewards]))

            mean_average_reward = sum(self.total_rewards[-100:])/min(len(self.total_rewards), 100)
            self.mean_average_rewards.append(mean_average_reward)
            with open('mean_average_rewards.txt', 'w') as f:
                f.write("\n".join([str(r) for r in self.mean_average_rewards]))

            self.epsilons.append(self.agent.get_epsilon(episode))
            with open('epsilons.txt', 'w') as f:
                f.write("\n".join([str(r) for r in self.epsilons]))

            print(f'Episode {episode+1:3}: | Reward: {total_reward:.3f} | Moving average reward: {mean_average_reward:.3f} | Epsilon: {self.agent.get_epsilon(episode):.3f} | Mean max q: {mean_max_q}')

            if total_reward > best_reward:
                best_reward = total_reward

                tr = int(total_reward)
                name = f'dqn_state_dict_{tr}.pt' if total_reward > 18.5 else 'dqn_state_dict.pt'
                torch.save(self.agent.dqn.state_dict(), name)

            self.best_rewards.append(best_reward)
            with open('best_rewards.txt', 'w') as f:
                f.write("\n".join([str(r) for r in self.best_rewards]))

            if tr==20:
                return self.total_rewards, best_reward, (episode+1)

        return self.total_rewards, best_reward, (episode+1)