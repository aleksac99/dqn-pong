from copy import deepcopy
import numpy as np
import torch

class DQNAgent:

    def __init__(self, dqn, epsilon_start, epsilon_decay, epsilon_end, device) -> None:
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.dqn = dqn.to(device)
        self.target_dqn = deepcopy(self.dqn).to(device)
        self.target_dqn.eval()
        # for p in self.target_dqn.parameters():
        #     p.requires_grad = False
        self.device = device

    def get_epsilon(self, episode):
        return max(self.epsilon_end, self.epsilon_start - episode * (self.epsilon_start-self.epsilon_end)/100.)

    def get_action(self, state, method, actions, episode):

        epsilon = self.get_epsilon(episode)
        p = np.random.rand()

        if (method !='greedy' and p < epsilon) or (method == 'random'):
            action = np.random.choice(actions.n)
        elif method=='eps_greedy' or method=='greedy':
            state = state.to(self.device)
            qs = self.dqn(state)
            action = torch.argmax(qs, dim=1).item()
        else:
            raise ValueError

        return action