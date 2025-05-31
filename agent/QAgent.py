import copy
import numpy
import torch
import random
import torchvision


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.state_to_hidden_proj = torch.nn.Linear(state_dim, hidden_dim)
        self.middle_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_v_proj = torch.nn.Linear(hidden_dim, action_num)
        self.activation = torch.nn.GELU()

        self.input_norm = torch.nn.LayerNorm(state_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, state_embed) -> torch.Tensor:
        x = self.input_norm(state_embed)
        x = self.state_to_hidden_proj(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.middle_layer(x)
        return self.hidden_to_v_proj(self.activation(x))


class QAgent(object):
    def __init__(self, state_dim: int, action_num: int,
                 hidden_dim=128, batch_size=64, learning_rate=2e-4, max_experience_len=20000,
                 gamma=0.96, device=torch.device("cpu")):
        # It looks like having a large experience replay buffer is very important for complex tasks
        super(QAgent, self).__init__()
        self.state_dim = state_dim
        self.action_num = action_num
        self.Q_network = QNetwork(state_dim, action_num, hidden_dim).to(device)
        self.Q_network_copy = copy.deepcopy(self.Q_network)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_experience_len = max_experience_len
        self.hidden_dim = hidden_dim
        self.experience = []
        self.gamma = gamma
        self.sync_step = 0

        self.optimizer1 = torch.optim.Adam(params=self.Q_network.parameters(), lr=self.learning_rate)
        self.device = device

    def update(self, state, next_state, action, reward, dones):
        # dones is a 0 or 1 integer to indicate whether the episode is finished or not
        experience_length = len(self.experience)
        # batched deep learning to update the Q_network
        if experience_length < 4*self.batch_size:
            # Don't update when do not have enough experience
            self.experience.append([state, next_state, action, reward, dones])
            # exit before update
            return
        elif experience_length >= self.max_experience_len:
            self.experience.pop(0)
            self.experience.append([state, next_state, action, reward, dones])
        else:
            self.experience.append([state, next_state, action, reward, dones])
        # Sample the batch from experience
        sample_experience = random.sample(self.experience, self.batch_size)
        states, next_states, actions, rewards, dones = zip(*sample_experience)
        # zip is a function that zips a list of iterable to a single list
        # for example, zip([a, b], [c, d], [e, f]) = [[a, c, e], [b, d, f]]
        states = torch.tensor(numpy.array(states), dtype=torch.float).to(self.device)
        # states dim: (batch_size, states_dim)
        next_states = torch.tensor(numpy.array(next_states), dtype=torch.float).to(self.device)
        # action dim: (batch_size )
        actions = torch.tensor(numpy.array(actions), dtype=torch.int64).to(self.device)
        # reward dim: (batch_size )
        rewards = torch.tensor(numpy.array(rewards), dtype=torch.float).to(self.device)
        dones = torch.tensor(numpy.array(dones), dtype=torch.float).to(self.device)
        # error dim: (batch_size )
        if self.sync_step == 100:
            # DQN algorithm for reducing the variance of updating networks
            self.Q_network_copy.load_state_dict(self.Q_network.state_dict())
            self.sync_step = 0
        else:
            self.sync_step += 1
        target = rewards + self.gamma*torch.max(self.Q_network_copy(next_states), dim=1)[0]*(1-dones)
        # detach the target to avoid degeneration
        loss = torch.nn.MSELoss()(torch.gather(
            self.Q_network(states), dim=1, index=actions.unsqueeze(1)).squeeze(), target.detach())
        # torch.gather gathers values of a tensor along some dimension
        # torch.max gives the values and indexes of the maximum element of a tensor.
        self.optimizer1.zero_grad()
        loss.backward()
        self.optimizer1.step()

    def take_action(self, state, epsilon=0.1, temperature=0.3):
        # epsilon greedy strategy
        q_values = self.Q_network(torch.tensor(state, dtype=torch.float).to(self.device))
        if random.random() > epsilon:
            return q_values.argmax().item()
        else:
            return random.choice(range(self.action_num))

    def load_network(self, filepath):
        self.Q_network = torch.load(filepath)
        self.Q_network_copy = copy.deepcopy(self.Q_network)
        self.optimizer1 = torch.optim.Adam(params=self.Q_network.parameters(), lr=self.learning_rate)

    def save_network(self, filepath):
        torch.save(self.Q_network, filepath)


class VideoEncoderCNN(torch.nn.Module):
    def __init__(self, video_dim, output_dim: int, hidden_dim=16, resized_size=(64, 64)):
        super(VideoEncoderCNN, self).__init__()
        self.video_dim = video_dim
        self.output_dim = output_dim
        self.resized_size = resized_size
        self.image_transform = torchvision.transforms.Normalize(0.1, 255)
        self.cnn_network = torch.nn.Sequential(
            torch.nn.Conv2d(video_dim[-1], hidden_dim, kernel_size=7, stride=2, padding=3),
            # (H/2, W/2)
            torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # (H/4, W/4)
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU()
            # (H/8, W/8)
        )
        self.output_layer = torch.nn.Linear(resized_size[0]*resized_size[1]//64*hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x dim: (H, W, C) or (N, H, W, C)
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(x, size=self.resized_size, mode="bilinear", align_corners=False)
            x = self.image_transform(x)
            # x dim: (N, C, H, W)
            x = self.cnn_network(x)
            return self.output_layer(x.reshape(x.size(0), -1))
        elif x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
            x = self.image_transform(x)
            # x dim: (1, C, H, W)
            x = torch.nn.functional.interpolate(x, size=self.resized_size, mode="bilinear", align_corners=False)
            # x dim: (1, C, H, W)
            x = self.cnn_network(x)
            return self.output_layer(x.reshape(-1))
        else:
            raise RuntimeError("The image input tensor must be 3D or 4D")


class VideoQAgent(QAgent):
    def __init__(self, video_dim, action_num: int, latent_dim=64, frame_skip_cycle=5, device=torch.device("cpu"), *args):
        super().__init__(latent_dim, action_num, *args)
        self.frame_stack_size = 3
        video_dim2 = [dim for dim in video_dim]
        video_dim2[-1] = video_dim[-1] * self.frame_stack_size
        self.Q_network = torch.nn.Sequential(
            VideoEncoderCNN(video_dim2, latent_dim),
            QNetwork(latent_dim, action_num)
        ).to(device)
        self.video_dim = video_dim
        self.device = device
        self.Q_network_copy = copy.deepcopy(self.Q_network)
        self.optimizer1 = torch.optim.Adam(params=self.Q_network.parameters(), lr=self.learning_rate)
        self.past_states = []
        self.past_next_states = []
        self.time_step = 0
        self.frame_skip_cycle = frame_skip_cycle

    def update(self, state, next_state, action, reward, dones):
        # This function stacks previous frames and update the model
        if len(self.past_next_states) < self.frame_stack_size * self.frame_skip_cycle:
            self.past_next_states.append(next_state)
            return
        else:
            self.past_next_states.pop(0)
            self.past_next_states.append(next_state)
        # past states dim: (D, H, W, C)
        states = numpy.flip(numpy.array(self.past_states), axis=0)[::self.frame_skip_cycle].transpose(
            (1, 2, 0, 3)).reshape(self.video_dim[0], self.video_dim[1], -1)
        next_states = numpy.flip(numpy.array(self.past_next_states), axis=0)[::self.frame_skip_cycle].transpose(
            (1, 2, 0, 3)).reshape(self.video_dim[0], self.video_dim[1], -1)

        super(VideoQAgent, self).update(states, next_states, action, reward, dones)

    def take_action(self, state, epsilon=0.1):
        self.time_step += 1

        if len(self.past_states) < self.frame_stack_size * self.frame_skip_cycle:
            self.past_states = [state, ]*self.frame_stack_size * self.frame_skip_cycle
        else:
            self.past_states.pop(0)
            self.past_states.append(state)

        states = numpy.flip(numpy.array(self.past_states), axis=0)[::self.frame_skip_cycle].transpose(
            (1, 2, 0, 3)).reshape(self.video_dim[0], self.video_dim[1], -1)
        return super(VideoQAgent, self).take_action(states)
