import torch
import sys
import numpy as np
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from model import DQN_MLP, DDQN_MLP
from memory import Memory


class Agent:
    def __init__(
        self,
        size_board,
        hidden_dim,
        output_dim,
        decay_rate,
        capacity,
        batch_size,
        gamma,
        learning_rate,
        dueling,
    ):
        # Use cuda
        self.__use_cuda = torch.cuda.is_available()

        # Models
        if dueling:
            # Usage of dueling models
            self.__policy_net = DDQN_MLP(
                size_board * size_board, hidden_dim, output_dim
            )
            self.__policy_net.train()

            self.__target_net = DDQN_MLP(
                size_board * size_board, hidden_dim, output_dim
            )
            print("Creating a dueling DQN...")
        else:
            self.__policy_net = DQN_MLP(size_board * size_board, hidden_dim, output_dim)
            self.__policy_net.train()

            self.__target_net = DQN_MLP(size_board * size_board, hidden_dim, output_dim)
            print("Creating a no-dueling DQN")

        # Target model is the model that will calculate the q_next_state_values
        self.update_target_net()

        for param in self.__target_net.parameters():
            param.requires_grad = False

        # Epsilon and decay rate
        self.__epsilon_start = 1.
        self.__epsilon_stop = 0.01
        self.__decay_rate = decay_rate
        self.__decay_step = 0
        self.__epsilon_threshold = 0
        # threshold 阀值
        # Memory
        self.__memory = Memory(capacity, size_board)
        # capacity 容量

        # Batch size
        self.__batch_size = batch_size

        # Gamma
        self.__gamma = gamma

        # Learning rate
        self.__learning_rate = learning_rate

        # Optimizer
        self.__optimizer = optim.Adam(self.__policy_net.parameters(), lr=learning_rate)

        if self.__use_cuda:
            self.__policy_net = self.__policy_net.cuda()
            self.__target_net = self.__target_net.cuda()

        self.__variable = (
            lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
            if self.__use_cuda
            else autograd.Variable(*args, **kwargs)
        )

    # Copy parameters of policy net to target net
    def update_target_net(self):
        self.__target_net.load_state_dict(self.__policy_net.state_dict())

    # Update threshold that divide the agent to choose a random action or a guided action
    def __update_epsilon(self):
        self.__epsilon_threshold = self.__epsilon_stop + (
            self.__epsilon_start - self.__epsilon_stop
        ) * np.exp(-self.__decay_rate * self.__decay_step)

    def __sample_batch(self):
        state, next_state, action, reward, done = self.__memory.sample(
            self.__batch_size
        )

        if self.__use_cuda:
            to_float_tensor, to_long_tensor = (
                torch.cuda.FloatTensor,
                torch.cuda.LongTensor,
            )

        else:
            to_float_tensor, to_long_tensor = torch.FloatTensor, torch.LongTensor

        return (
            to_float_tensor(state),
            to_float_tensor(next_state),
            to_long_tensor(action),
            to_float_tensor(reward),
            to_float_tensor(done),
        )

    def store_memory(self, state, next_state, action, reward, done):
        self.__memory.store(state, next_state, action, reward, done)

    def selection_action(self, valid_movements, state):
        sample = np.random.rand(1)

        # Increase decay step to decrease epsilon threshold exponentially
        self.__decay_step += 1

        # Update epsilon threshold
        self.__update_epsilon()

        if sample > self.__epsilon_threshold:
            with torch.no_grad():
                output = self.__policy_net(
                    self.__variable(torch.from_numpy(state).float())
                )
                output_np = output.cpu().detach().numpy()
                ordered = np.flip(
                    np.argsort(np.expand_dims(output_np, axis=0), axis=1)
                )[0]
                for action in ordered:
                    if valid_movements[action] != 0:
                        return action

        else:
            return np.random.choice(np.nonzero(valid_movements)[0])

    def train_model(self):
        if len(self.__memory) < self.__batch_size:
            return -1

        state, next_state, action, reward, done = self.__sample_batch()

        action = action.unsqueeze(0)
        action = action.view(self.__batch_size, 1)
        # print(action)
        q_values = self.__policy_net(state).gather(1, action)
        with torch.no_grad():
            next_q_values = self.__target_net(next_state).max(1)[0]
            target_q_values = reward + self.__gamma * (1 - done) * next_q_values

        # Loss
        # loss = (q_values - target_q_values).pow(2).mean()
        loss = F.smooth_l1_loss(q_values, target_q_values.view(self.__batch_size, 1))

        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    def get_threshold(self):
        return self.__epsilon_threshold

    def save_model(self, path):
        torch.save(self.__policy_net.state_dict(), path)
