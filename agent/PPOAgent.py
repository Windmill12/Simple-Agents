import copy
import numpy
import torch
import random
import torchvision

class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorNet, self).__init__()
        self.state_to_hidden_proj = torch.nn.Linear(state_dim, hidden_dim)
        self.hidden_norm = torch.nn.LayerNorm(hidden_dim)
        self.activation = torch.nn.GELU()
        self.middle_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_action_proj = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, state_embed):
        x = self.state_to_hidden_proj(state_embed)
        x = self.activation(self.hidden_norm(x))
        x = self.middle_layer(x)
        x = self.activation(self.hidden_norm(x))
        x = self.hidden_to_action_proj(x)

        return torch.softmax(x, dim=-1)


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(CriticNet, self).__init__()
        self.state_to_hidden_proj = torch.nn.Linear(state_dim, hidden_dim)
        self.hidden_norm = torch.nn.LayerNorm(hidden_dim)
        self.activation = torch.nn.GELU()
        self.middle_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_action_proj = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state_embed):
        x = self.state_to_hidden_proj(state_embed)
        x = self.activation(self.hidden_norm(x))
        x = self.middle_layer(x)
        x = self.activation(self.hidden_norm(x))
        x = self.hidden_to_action_proj(x)

        return x


class PPOAgent(object):
    def __init__(self, state_dim: int, action_num: int,
                 hidden_dim=128, batch_size=64, learning_rate=2e-4, max_experience_len=1024,
                 gamma=0.96, device=torch.device("cpu")):
        super(PPOAgent, self).__init__()
        self.state_dim = state_dim
        self.action_num = action_num
        self.actor_network = ActorNet(state_dim, action_num, hidden_dim).to(device)
        self.critic_network = CriticNet(state_dim, hidden_dim).to(device)
        self.learning_rate = learning_rate
        self.max_experience_len = max_experience_len
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lambda_ = 0.7
        self.num_epochs = 8
        self.eps = 0.2
        # In classical PPO algorithm, the experience may contain multiple episodes
        # When generating the experience, the policy of agent won't update
        self.experience = []
        # Define optimizers to reduce network loss
        self.act_optimizer = torch.optim.Adam(params=self.actor_network.parameters(), lr=self.learning_rate)
        self.crit_optimizer = torch.optim.Adam(params=self.critic_network.parameters(), lr=self.learning_rate)

        self.device = device

    def update(self, state, next_state, action, reward, dones):
        if len(self.experience) < self.max_experience_len:
            if len(self.experience) == 0:
                print("Collecting experience for updating...")
            self.experience.append([state, next_state, action, reward, dones])

        else:
            print("Updating agent according to collected experience")
            states, next_states, actions, rewards, dones = zip(*self.experience)
            # zip is a function that zips a list of iterable to a single list
            # for example, zip([a, b], [c, d], [e, f]) = [[a, c, e], [b, d, f]]
            states = torch.tensor(numpy.array(states), dtype=torch.float).to(self.device)
            # states dim: (experience_len, states_dim)
            next_states = torch.tensor(numpy.array(next_states), dtype=torch.float).to(self.device)
            # action dim: (experience_len )
            actions = torch.tensor(numpy.array(actions), dtype=torch.int64).to(self.device)
            # unsqueeze the actions for torch.gather
            actions = actions.unsqueeze(-1)
            # reward dim: (experience_len )
            rewards = torch.tensor(numpy.array(rewards), dtype=torch.float).to(self.device)
            dones = torch.tensor(numpy.array(dones), dtype=torch.float).to(self.device)
            # obtain TD target and value first. Since self.critic_network() returns (experience_len, 1)
            # squeeze -1 dimension first to avoid broadcasting errors
            target = rewards + self.gamma * (self.critic_network(next_states).squeeze(-1)) * (1 - dones)
            value = self.critic_network(states).squeeze(-1)
            # after converting this would be a numpy array which can be easily flipped
            delta_advantages = (target - value).cpu().detach().numpy()
            done_list = dones.cpu().numpy()
            # obtain generalized advantage value
            # the advantage list will be a list which contains [A_n, A_{n-1}...., A_0]
            advantage = 0
            advantage_list = []
            for delta, done in zip(delta_advantages[::-1], done_list[::-1]):
                # if the episode ends, previous advantage should not be added
                advantage = advantage * self.gamma * self.lambda_ * (1 - done) + delta
                advantage_list.append(advantage)
            # flip the tensor to get G.A.E.
            advantage_list = torch.tensor(advantage_list).flip(0).unsqueeze(-1)
            advantage_list = (advantage_list - advantage_list.mean()) / (advantage_list.std() + 1e-8)
            # torch.gather here generates a new tensor whose elements are specified by:
            # output[i][j] = input[i][index[i][j]] for dim==1
            old_log_prob = torch.log(self.actor_network(states).gather(1, actions).detach() + 1e-8)

            for _ in range(self.num_epochs):
                # We divide the indices into several mini-batches, and send them to update the networks
                indices = torch.randperm(states.size(0))
                for start in range(0, states.size(0), self.batch_size):
                    end = min(start + self.batch_size, states.size(0))
                    # sample the experience to obtain a small batch
                    batch_indices = indices[start:end]
                    states_batch = states[batch_indices]
                    actions_batch = actions[batch_indices]
                    advantages_batch = advantage_list[batch_indices]
                    # Importance sampling ratio. If the old strategy and current strategy is the same the loss reduces to
                    # log(\pi_{\phi}(a_t|s_t)) * A_t
                    log_prob = torch.log(self.actor_network(states_batch).gather(1, actions_batch) + 1e-8)
                    ratios = torch.exp(log_prob - old_log_prob[batch_indices])
                    # Calculate actor loss. The loss is given by clipped importance sampling results
                    # The strategy won't update if it is too far away from original one, since the loss is clipped
                    loss1 = ratios * advantages_batch
                    loss2 = torch.clamp(ratios, 1-self.eps, 1+self.eps) * advantages_batch
                    actor_loss = torch.mean(-torch.min(loss1, loss2))
                    # Calculate critic loss
                    critic_loss = torch.nn.MSELoss()(self.critic_network(states_batch).squeeze(-1),
                                                     target[batch_indices].detach())
                    # update actor and critic network
                    self.crit_optimizer.zero_grad()
                    critic_loss.backward()
                    self.crit_optimizer.step()

                    self.act_optimizer.zero_grad()
                    actor_loss.backward()
                    self.act_optimizer.step()

            # Reset experience
            self.experience = []

    def take_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.action_num))
        probs = self.actor_network(torch.tensor(state, dtype=torch.float).to(self.device))
        # Note since actor-critic is online learning algorithm, you should always sample
        # from the distribution generated from actor
        return torch.multinomial(probs, num_samples=1).item()

    def load_network(self):
        pass

    def save_network(self):
        pass