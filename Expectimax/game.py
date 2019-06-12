import collections, random, operator
import matplotlib.pyplot as plt
import numpy as np
import player
import sys
import math
import pickle

from Expectimax import gameutil

util = gameutil.gameutil()

import time

METHODS = ['direness', 'simple', 'weighted', 'max']
ALL_BOARDS = [3, 4, 5, 10, 50, 100]


def printGames(games):
    for i in range(len(games)):
        print(util.bitToBoard(games[i]))
        print("Score: ", util.getScore(games[i]))
        print()


def executeAction(games, action):
    done = False
    for i in range(len(games)):
        previous = games[i]
        games[i] = util.swipe(action, games[i])
        if (games[i] != previous):
            games[i] = util.placeRandomTile(games[i])
        if util.isEnd(games[i]):
            done = True
    return done


def play2048(num_boards, fill, method, depth):
    games = [util.newBoard() for _ in range(num_boards)]
    agent = player.Player(depth, util.evalFn, util, fill)

    done = False
    total = 0.0

    highest_tiles = []
    while not done:
        start = time.time()
        print("===================")
        print("CURRENT GAMES")
        printGames(games)
        values = collections.defaultdict(float)
        count = collections.defaultdict(float)

        if method == 'direness':
            # look only at top 30% most dire boards
            most_dire = np.argsort([util.direness(games[i]) for i in range(num_boards)])
            most_dire = most_dire[::-1][0: int(np.ceil((.3 * num_boards)))]
            for i in most_dire:
                action, vals = agent.getAction(games[i])
                for move, score in vals:
                    values[move] += score * util.direness(games[i])
                    count[move] += 1

        elif method == 'max':
            # take the highest scored action for any board
            for game in games:
                action, vals = agent.getAction(game)
                empty_pos = util.emptyPos(game)
                for move, score in vals:
                    values[move] = max(values[move], score)
                    count[move] = 1

        else:
            # if simple average, sum the values for each action across every board
            # OR
            # if weighted average, sum the values for each action across every board
            # weighted by the board's direness    
            for game in games:
                action, vals = agent.getAction(game)
                empty_pos = util.emptyPos(game)
                for move, score in vals:
                    values[move] += score if method == 'simple' else score * util.direness(game)
                    count[move] += 1

        for key in values:
            values[move] /= count[move]

        action = max(values.items(), key=operator.itemgetter(1))[0]

        print(action)

        print("TAKING ACTION: " + util.convertToText(action) + "\n")

        done = executeAction(games, action)

        printGames(games)

        print("Time: " + str(time.time() - start))
        print("===================")

        if done:
            for game in games:
                total += util.getScore(game)
                highest_tiles.append(util.getHighest(game))

    min_highest_tile = min(highest_tiles)
    max_highest_tile = max(highest_tiles)
    score = total / float(num_boards)

    return score, min_highest_tile, max_highest_tile


####################################################


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Use the AI to play 2048")
    parser.add_argument('-d', '--depth', help="Depth for Expectimax", default=2, type=int)
    parser.add_argument('-b', '--nboards', help="Number of boards to play on", default=2, type=int)
    parser.add_argument('-g', '--ngames', help="Number of full games to play", default=1, type=int)
    parser.add_argument('-m', '--method', help="Which strategy to use", default='simple',
                        choices=('direness', 'simple', 'weighted', 'max'))
    parser.add_argument('-f', '--fill', help="Use fill (1) or sampling (0) for Expectimax", default=1, type=int)
    parser.add_argument('-n', '--name', help="File name to use to store the data")

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    depth = args.depth
    nboards = args.nboards
    print(nboards)
    num_games = args.ngames
    method = args.method
    fill = args.fill

    # for num_boards in ALL_BOARDS:
    #    print("-----------------------------------------")
    #    print("EXECUTING " + str(num_boards) + " BOARDS")
    scores = []
    counter_max = collections.defaultdict(int)
    counter_min = collections.defaultdict(int)
    for i in range(num_games):
        curr_score, min_highest_tile, max_highest_tile = play2048(nboards, fill, method, depth)
        scores.append(curr_score)
        counter_max[max_highest_tile] += 1
        counter_min[min_highest_tile] += 1
    average = sum(scores) / float(num_games)
    variance = sum([(score - average) ** 2 for score in scores])

    print("AVERAGE: ", average)
    print("VARIANCE: ", variance)
    print("min_highest_tile: ", counter_min)
    print("max_highest_tile: ", counter_max)

    # data = [scores, counter_min, counter_max, average, variance]
    # filename = 'd' + str(depth) + 'b' + str(nboards) + 'g' + str(num_games) + 'f' + str(fill) + 'm' + str(method)
    # with open(filename, "wb") as f:
    #        pickle.dump(data, f)

    #    print("FINISHED EXECUTING " + str(num_boards) + " BOARDS")
    #    print("-----------------------------------------")


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
