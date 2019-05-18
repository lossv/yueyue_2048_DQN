import numpy as np


def get_zeros(state):
    state = state[state == 0]
    return state.size


def get_variance(state):
    return np.var(state)


def get_smooth(state):
    pass


def get_smooth(state):
    row = state.T[0].size    #行数
    col = state[0].size
    sum = 0
    for i, j in enumerate(state):
        for m, n in enumerate(j):
            if 0 < i < row - 1 and m < col - 1:   # 中间部分
                a = [i - 1, i + 1, i, i]
                b = [m, m, m - 1, m + 1]
                array = n - state[a, b]
                print("a = ", a)
                print("b = ", b)
                sum += np.log(np.abs(array.sum() + 1))
            elif i == 0 and m == 0:         # 左上角
                a = [i + 1, i]
                b = [m, m + 1]
                array = n - state[a, b]
                sum += np.log(np.abs(array.sum() + 1))
            elif i == 0 and m == col - 1:     # 右上角
                a = [i + 1, i]
                b = [m, m - 1]
                array = n - state[a, b]
                sum += np.log(np.abs(array.sum() + 1))
            elif i == row - 1 and m == 0:   # 左下角
                a = [i - 1, i]
                b = [m, m + 1]
                array = n - state[a, b]
                sum += np.log(np.abs(array.sum() + 1))
            elif i == row - 1 and m == col - 1:  # 右下角
                a = [i - 1, i]
                b = [m, m - 1]
                array = n - state[a, b]
                sum += np.log(np.abs(array.sum() + 1))
            elif i == 0 and 0 < m < col - 1:   # 上
                a = [i + 1, i, i]
                b = [m, m + 1, m - 1]
                array = n - state[a, b]
                sum += np.log(np.abs(array.sum() + 1))
            elif i == row - 1 and 0 < m < col - 1:  # 下
                a = [i - 1, i, i]
                b = [m, m + 1, m - 1]
                array = n - state[a, b]
                sum += np.log(np.abs(array.sum() + 1))
            elif 0 < i < row - 1 and col == 0:
                a = [i - 1, i + 1, i]
                b = [m, m, m + 1]
                array = n - state[a, b]
                sum += np.log(np.abs(array.sum() + 1))
            elif 0 < i < row - 1 and col == 0:
                a = [i - 1, i + 1, i]
                b = [m, m, m - 1]
                array = n - state[a, b]
                sum += np.log(np.abs(array.sum() + 1))
    return sum


def test2(state):
    sum = 0
    for i in state:
        array = i[0] - i
        sum += np.log(np.abs(array.sum() + 1))
    for i in state.T:
        array = i[0] - i
        sum -= np.log(np.abs(array.sum() + 1))
    print(sum)
    return abs(sum)



x = [1, 2, 3, 4, 5, -11, 1, 1]
x = np.array(x)
# print(x.sum())

c = np.arange(0, 16)
c = c.reshape(4, 4)
# print(c)

# i, m = 1, 1
# n = 4
# a = [i - 1, i + 1, i, i]
# b = [m, m, m - 1, m + 1]
# array = n - c[a, b]
# print("a = ", a)
# print("b = ", b)
# print(np.log(np.abs(array.sum())))

print(test2(c))

x = x.reshape(2, 4)
# print(c[0].size)
# j -= state[a, b]

















# print(y)
# print(y.sum(axis=1))
# print(x[..., 0:1])
# print(a.var())
