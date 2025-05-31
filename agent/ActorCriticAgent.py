import copy
import numpy
import torch
import random
import torchvision

class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorNet, self).__init__()
        self.state_to_hidden_proj = torch.nn.Linear(state_dim, hidden_dim)
        self.hidden_norm1 = torch.nn.LayerNorm(hidden_dim)
        self.activation = torch.nn.GELU()
        self.hidden_norm2 = torch.nn.LayerNorm(hidden_dim)
        self.middle_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_action_proj = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, state_embed, temperature=1):
        x = self.state_to_hidden_proj(state_embed)
        # layer norm should be applied just before activation?
        x = self.activation(self.hidden_norm1(x))
        x = self.middle_layer(x)
        x = self.activation(self.hidden_norm2(x))
        x = self.hidden_to_action_proj(x)

        return torch.softmax(x/temperature, dim=-1)


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(CriticNet, self).__init__()
        self.state_to_hidden_proj = torch.nn.Linear(state_dim, hidden_dim)
        self.hidden_norm1 = torch.nn.LayerNorm(hidden_dim)
        self.activation = torch.nn.GELU()
        self.hidden_norm2 = torch.nn.LayerNorm(hidden_dim)
        self.middle_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_action_proj = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state_embed):
        x = self.state_to_hidden_proj(state_embed)
        x = self.activation(self.hidden_norm1(x))
        x = self.middle_layer(x)
        x = self.activation(self.hidden_norm2(x))
        x = self.hidden_to_action_proj(x)

        return x


class ActorCriticAgent(object):
    def __init__(self, state_dim: int, action_num: int,
                 hidden_dim=128, batch_size=256, learning_rate=2e-4, max_experience_len=2048,
                 gamma=0.96, device=torch.device("cpu")):
        super(ActorCriticAgent, self).__init__()
        self.state_dim = state_dim
        self.action_num = action_num
        self.actor_network = ActorNet(state_dim, action_num, hidden_dim).to(device)
        self.critic_network = CriticNet(state_dim, hidden_dim).to(device)
        self.learning_rate = learning_rate
        self.max_experience_len = max_experience_len
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        # Define optimizers to reduce network loss
        self.act_optimizer = torch.optim.Adam(params=self.actor_network.parameters(), lr=self.learning_rate)
        self.crit_optimizer = torch.optim.Adam(params=self.critic_network.parameters(), lr=self.learning_rate)

        self.device = device

    def update(self, state, next_state, action, reward, dones):
        # experience replay is also not allowed since the experience's strategy is not the same with current one
        states = torch.tensor(numpy.array(state), dtype=torch.float).to(self.device)
        # states dim: (states_dim)
        next_states = torch.tensor(numpy.array(next_state), dtype=torch.float).to(self.device)
        # action dim: (1 )
        actions = torch.tensor(numpy.array(action), dtype=torch.int64).to(self.device)
        # reward dim: (1 )
        rewards = torch.tensor(numpy.array(reward), dtype=torch.float).to(self.device)
        dones = torch.tensor(numpy.array(dones), dtype=torch.float).to(self.device)
        # target from Bellman equation, = R + gamma * v(s')
        targets = rewards + self.gamma * self.critic_network(next_states) * (1 - dones)
        values = self.critic_network(states)
        # targets, values dim: (1)
        critic_loss = torch.nn.MSELoss()(targets.detach(), values)
        # the advantage will be used for updating the actor network.
        advantages = targets - values
        # torch.gather here generates a new tensor whose elements are specified by:
        # output[i][j] = input[i][index[i][j]] for dim==1
        log_actor_prob = torch.log(self.actor_network(states)[actions] + 1e-5)
        actor_loss = -(log_actor_prob * advantages.detach()).mean()
        # update actor and critic network
        self.crit_optimizer.zero_grad()
        critic_loss.backward()
        self.crit_optimizer.step()

        self.act_optimizer.zero_grad()
        actor_loss.backward()
        self.act_optimizer.step()

    def take_action(self, state, epsilon=0.1):
        if random.random()<epsilon:
            return random.choice(range(self.action_num))
        probs = self.actor_network(torch.tensor(state, dtype=torch.float).to(self.device))
        # Note since actor-critic is online learning algorithm, you should always sample
        # from the distribution generated from actor
        return torch.multinomial(probs, num_samples=1).item()

    def load_network(self):
        pass

    def save_network(self):
        pass