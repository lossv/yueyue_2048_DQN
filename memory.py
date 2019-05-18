import numpy as np
import random


class Memory:
    def __init__(self, capacity, size_board):
        # Capacity of memory
        self.__capacity = capacity
        self.__filled = 0

        # Size of the board will be used to create the arrays
        self.__size_bard = size_board

        # Arrays
        self.__state = np.zeros((capacity, size_board * size_board))
        self.__next_state = np.zeros((capacity, size_board * size_board))
        self.__action = np.zeros(capacity)
        self.__reward = np.zeros(capacity)
        self.__done = np.zeros(capacity, dtype=np.int64)

        # Pointer to arrays
        self.__data_pointer = 0

    # Store the experience
    def store(self, state, next_state, action, reward, done):
        self.__state[self.__data_pointer] = state
        self.__next_state[self.__data_pointer] = next_state
        self.__action[self.__data_pointer] = action
        self.__reward[self.__data_pointer] = reward
        self.__done[self.__data_pointer] = done

        self.__data_pointer += 1

        if self.__filled < self.__capacity:
            self.__filled += 1

        # Replacement of the experiences. The older ones leave the array and put the news.
        if self.__data_pointer == (self.__capacity - 1):
            self.__data_pointer = 0

    # Sample (batch_size numbers of)expriences in a randomized
    def sample(self, batch_size):
        random_indexes = random.sample(range(0, self.__filled), batch_size)

        return (
            self.__state[random_indexes],
            self.__next_state[random_indexes],
            self.__action[random_indexes],
            self.__reward[random_indexes],
            self.__done[random_indexes],
        )

    def __len__(self):
        return self.__filled
