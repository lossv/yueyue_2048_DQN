from __future__ import division
import numpy as np
import math
import itertools
import random
import os
import copy


def weighted_choice(iterable, probability):
    ''' randomly chooses element according to specified
    probability densities
    '''
    total = sum(probability)
    if total == 0:
        return None
    r = random.uniform(0, total)
    current = 0
    for i, p in zip(iterable, probability):
        current += p
        if current >= r:
            return i


class Board():

    def __init__(self, size):
        self.board = np.array([[0] * size for i in range(size)])
        self.score = 0
        self.size = size
        for i in range(2):
            self.random_insert()

    def print_board(self):
        for row in self.board:
            print(' '.join(['{:5s}'.format(str(2 ** i)) if i > 0 else '{:5s}'.format(str(0)) for i in row]))

    def remove_blank_tile(self, row, pos):
        row[pos] = 0
        return np.concatenate((row[:pos], row[pos + 1:], np.array([0])))

    def swipe_without_merge(self, row):
        for i in range(len(row) - 1):
            while row[i] == 0 and any([j != 0 for j in row[i:]]):
                row = self.remove_blank_tile(row, i)
        return row

    def swipe_row(self, row):
        row = self.swipe_without_merge(row)
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] += 1
                row = self.remove_blank_tile(row, i + 1)
        return row

    def swipe_board_left(self):
        for i, row in enumerate(self.board):
            self.board[i, :] = self.swipe_row(row)

    def swipe_board(self, direction):
        self.board = np.rot90(self.board, direction)
        old_board = np.copy(self.board)
        self.swipe_board_left()
        if not np.array_equal(old_board, self.board) and (self.board == 0).sum() > 0:
            self.random_insert()
        self.board = np.rot90(self.board, 4 - direction)

    def random_insert(self):
        possible_inserts = [(y, x) for (y, x) in itertools.combinations_with_replacement(range(self.size), 2) if
                            self.board[y, x] == 0]
        self.board[random.choice(possible_inserts)] = weighted_choice([1, 2], [0.9, 0.1])

    def game_over(self):
        if (self.board == 0).sum() > 0:
            return False
        for row in self.board:
            if any([row[i] == row[i + 1] for i in range(self.size - 1)]):
                return False
        for row in np.rot90(self.board, 1):
            if any([row[i] == row[i + 1] for i in range(self.size - 1)]):
                return False
        return True


class Game():

    def __init__(self, size=4):
        self.game_board = Board(size)
        self.turn_number = 0

    def report_game(self):
        print('Final board: ')
        self.game_board.print_board()
        print()
        print('Max tile: ', 2 ** self.game_board.board.max())

    def user_play(self):
        while not self.game_board.game_over():
            self.game_board.print_board()
            choice = {'a': 0, 'w': 1, 'd': 2, 's': 3}[input('')]
            os.system('clear')
            self.game_board.swipe_board(int(choice))
        self.game_board.print_board()
        print('Game over!')
        self.report_game()

    def random_play(self):
        while not self.game_board.game_over():
            choice = random.randint(1, 4)
            self.game_board.swipe_board(choice)
            self.turn_number += 1
        return self.turn_number

    def survival_trial(self, direction):
        temp_game = Game(self.game_board.size)
        temp_game.game_board.board = np.copy(self.game_board.board)
        temp_game.game_board.swipe_board(direction)
        return temp_game.random_play()

    def average_survival(self, direction, trials):
        total = 0
        for i in range(trials):
            total += self.survival_trial(direction)
        return total / trials

    def monty_carlo_choice(self, trials):
        possible_moves = []
        for i in range(4):
            temp_board = Board(4)
            temp_board.board = np.copy(self.game_board.board)
            temp_board.swipe_board(i)
            if not np.all(np.equal(temp_board.board, self.game_board.board)):
                possible_moves.append(i)
        return max(possible_moves, key=lambda i: self.average_survival(i, trials))

    def monty_carlo_play(self, trials):
        while not self.game_board.game_over():
            choice = self.monty_carlo_choice(trials)
            last_move = choice
            print("choice ", choice)
            self.game_board.swipe_board(choice)
            self.turn_number += 1
            print('turn', self.turn_number)
            self.game_board.print_board()
            print()
        return self.turn_number


if __name__ == '__main__':
    # game = Game()
    # game.user_play()
    # game.random_play()
    # game.report_game()
    #
    game2 = Game()
    game2.monty_carlo_play(100)
    game2.report_game()
