import numpy as np
import random as rand


class gameutil:
    def __init__(self):
        self.size = 4
        self.initTables()
        self.weights1 = [7, 6, 5, 4, 6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1]
        self.weights2 = [4, 5, 6, 7, 3, 4, 5, 6, 2, 3, 4, 5, 1, 2, 3, 4]
        self.weights3 = [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7]
        self.weights4 = [4, 3, 2, 1, 5, 4, 3, 2, 6, 5, 4, 3, 7, 6, 5, 4]

    def newBoard(self):
        return 1 << (4 * rand.randint(0, 15))

    def getLegalMoves(self, board):
        legalmoves = set()
        for action in range(4):
            new = self.swipe(action, board)
            if self.swipe(action, board) != board:
                legalmoves.add(action)
        return legalmoves

    def initTables(self):
        self.tableL = {}
        self.tableR = {}
        self.scoreTable = {}
        num = self.size << 2
        for a in range(num):
            for b in range(num):
                for c in range(num):
                    for d in range(num):
                        row = np.asarray([a, b, c, d])
                        if sum(row) == 0:
                            self.tableL[0] = 0
                            self.tableR[0] = 0
                            self.scoreTable[0] = 0
                            continue
                        rowL = row[row != 0].copy()
                        rowR = row[row != 0]
                        rowlist = rowL.tolist()
                        for i in range(rowL.size - 1):
                            if rowlist[i] == rowlist[i + 1]:
                                rowlist[i] += 1
                                rowlist[i + 1] = 0
                        newrowL = [x for x in rowlist if x != 0]
                        newrowL = np.asarray(newrowL + np.zeros(self.size - len(newrowL)).tolist())
                        newrowL = list(map(int, newrowL))
                        rowlist = rowR.tolist()
                        for i in range(rowR.size - 1, 0, -1):
                            if rowlist[i] == rowlist[i - 1]:
                                rowlist[i] += 1
                                rowlist[i - 1] = 0
                        newrowR = [x for x in rowlist if x != 0]
                        newrowR = np.asarray(np.zeros(self.size - len(newrowR)).tolist() + newrowR)
                        newrowR = list(map(int, newrowR))
                        key = row[0] << 12 | row[1] << 8 | row[2] << 4 | row[3]
                        valL = newrowL[0] << 12 | newrowL[1] << 8 | newrowL[2] << 4 | newrowL[3]
                        valR = newrowR[0] << 12 | newrowR[1] << 8 | newrowR[2] << 4 | newrowR[3]
                        self.tableL[key] = valL
                        self.tableR[key] = valR
                        score = 0
                        for x in range(self.size):
                            val = row[x]
                            if val > 1:
                                score += (val - 1) * (1 << val)
                        self.scoreTable[key] = score

    def getScore(self, board):
        row1 = (0xFFFF << 48 & board) >> 48
        row2 = (0xFFFF << 32 & board) >> 32
        row3 = (0xFFFF << 16 & board) >> 16
        row4 = 0xFFFF & board
        return self.scoreTable[row1] + self.scoreTable[row2] + self.scoreTable[row3] + self.scoreTable[row4]

    # should return a list of new boards
    def generateSuccessor(self, action, board):
        new = self.swipe(action, board)
        empty_pos = self.emptyPos(new)
        post_actions = []
        for i in range(len(empty_pos)):
            emp = empty_pos[i]
            post_actions.append(self.placeTile(emp, new))
        return post_actions

    def generateSuccessorHeuristic(self, action, board):
        heuristic = self.swipe(action, board)
        empty_pos = self.emptyPos(heuristic)
        for i in range(len(empty_pos)):
            emp = empty_pos[i]
            heuristic = self.placeTile(emp, heuristic)
        return heuristic

    def placeTile(self, pos, board):
        return board | 1 << (4 * pos)

    def placeRandomTile(self, board):
        empty_pos = self.emptyPos(board)
        if len(empty_pos) == 0:
            return board

        tileval = 1

        pos = rand.choice(empty_pos)
        return board | (tileval << (4 * pos))

    def swipe(self, action, board):
        if (action == 3):
            return self.swipeLeft(board)
        elif (action == 0):
            return self.swipeUp(board)
        elif (action == 1):
            return self.swipeRight(board)
        else:
            return self.swipeDown(board)

    def emptyPos(self, board):
        lst = []
        for x in range(16):
            i = 0xF << (4 * x)
            if i & board == 0:
                lst.append(x)

        return lst

    def getTile(self, board, k):
        x = 1 << (0xF & (board >> (4 * k)))
        return x if x > 1 else 0

    def countZeros(self, board):
        count = 0
        for k in range(16):
            count += int(self.getTile(board, k) == 0)
        return count

    def swipeLeft(self, board):
        row1 = (0xFFFF << 48 & board) >> 48
        row2 = (0xFFFF << 32 & board) >> 32
        row3 = (0xFFFF << 16 & board) >> 16
        row4 = 0xFFFF & board
        return self.tableL[row1] << 48 | self.tableL[row2] << 32 | self.tableL[row3] << 16 | self.tableL[row4]

    def swipeRight(self, board):
        row1 = (0xFFFF << 48 & board) >> 48
        row2 = (0xFFFF << 32 & board) >> 32
        row3 = (0xFFFF << 16 & board) >> 16
        row4 = 0xFFFF & board
        return self.tableR[row1] << 48 | self.tableR[row2] << 32 | self.tableR[row3] << 16 | self.tableR[row4]

    def swipeUp(self, board):
        transpose = self.transpose(board)
        row1 = (0xFFFF << 48 & transpose) >> 48
        row2 = (0xFFFF << 32 & transpose) >> 32
        row3 = (0xFFFF << 16 & transpose) >> 16
        row4 = 0xFFFF & transpose
        return self.transpose(
            self.tableL[row1] << 48 | self.tableL[row2] << 32 | self.tableL[row3] << 16 | self.tableL[row4])

    def swipeDown(self, board):
        transpose = self.transpose(board)
        row1 = (0xFFFF << 48 & transpose) >> 48
        row2 = (0xFFFF << 32 & transpose) >> 32
        row3 = (0xFFFF << 16 & transpose) >> 16
        row4 = 0xFFFF & transpose
        return self.transpose(
            self.tableR[row1] << 48 | self.tableR[row2] << 32 | self.tableR[row3] << 16 | self.tableR[row4])

    def transpose(self, board):
        c1 = board & 0xF0F00F0FF0F00F0F
        c2 = board & 0x0000F0F00000F0F0
        c3 = board & 0x0F0F00000F0F0000
        c = c1 | (c2 << 12) | (c3 >> 12)
        d1 = c & 0xFF00FF0000FF00FF
        d2 = c & 0x00FF00FF00000000
        d3 = c & 0x00000000FF00FF00
        return d1 | (d2 >> 24) | (d3 << 24)

    def isEnd(self, board):
        grid = self.bitToBoard(board)
        for i in range(4):
            for j in range(4):
                e = grid[i, j]
                if not e:
                    return False
                if j and e == grid[i, j - 1]:
                    return False
                if i and e == grid[i - 1, j]:
                    return False
        return True

    def getHighest(self, board):
        highest = 0
        for k in range(16):
            val = 1 << ((board >> (4 * k)) & 0xF)
            if val > 1:
                if val > highest:
                    highest = val
        return highest

    def bitToBoard(self, board):
        cboard = np.zeros(16).astype(int)
        for k in range(16):
            cboard[k] = 1 << ((board >> (4 * k)) & 0xF)
            if cboard[k] == 1:
                cboard[k] = 0
        return cboard[::-1].reshape((4, 4))

    def smoothness(self, board):
        sm = 0.0
        for r in range(4):
            for k in range(3):
                sm += abs(self.getTile(board, 4 * r + k) - self.getTile(board, 4 * r + k + 1))

        for c in range(4):
            for k in range(3):
                sm += abs(self.getTile(board, 4 * k + c) - self.getTile(board, 4 * (k + 1) + c))

        return -sm  # penalize high disparity

    def openTilePenalty(self, board, n=5):
        # return util.countZeros(board) - n
        return -((self.countZeros(board) - n) ** 2)

    def weightedGrid(self, board):
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        sum4 = 0.0
        for i in range(16):
            val = 1 << ((board >> (4 * i)) & 0xF)
            if val > 1:
                sum1 += self.weights1[i] * val
                sum2 += self.weights2[i] * val
                sum3 += self.weights3[i] * val
                sum4 += self.weights4[i] * val
        return max(sum1, sum2, sum3, sum4)

    def evalFn(self, board, isEnd, score):
        if isEnd:
            return float('-inf')

        eval = 0.0
        eval += score
        eval += self.weightedGrid(board)
        # eval += monotonicity(currentGameState, k=10.0)
        # eval += 10 * openTilePenalty(currentGameState)
        eval += 2 * self.smoothness(board)

        return eval

    def direness(self, board):
        return (-self.smoothness(board) / 100.0 / (self.countZeros(board) + .5)) ** 3

    def convertToText(self, action):
        if action == 0:
            return 'Up'
        elif action == 1:
            return 'Right'
        elif action == 2:
            return 'Down'
        else:
            return 'Left'
