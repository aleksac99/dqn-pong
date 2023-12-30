import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, input_shape, n_actions):

        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU())

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions, bias=True)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return torch.tensor(o.shape).prod().item()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)

        return self.fc(x)