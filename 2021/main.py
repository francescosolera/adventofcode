from itertools import cycle, product, chain
from collections import defaultdict, Counter, namedtuple, deque

import numpy as np
import re

'''DESIGN RULE OF THUMBS:
- keep all functions equal or below 10 lines of code (def line and return line excluded)
    - def line is always: def dayX(input_file_path)
    - return line is always: return r1, r2
- no solutions should be unreasonably slow
- use numpy only when we know the allocation size in advance, otherwise use python data structures
- define a function if it's called more than once
- define a variable if it's used more than once
- no semicolon allowed
'''


def day1(input_file_path):
    """DAY 1 - 3 LINES - https://adventofcode.com/2021/day/1

    Compunting the difference of a sum/average over a sliding window let us cancel out the terms shared by the two
    windows; in the case of a 3x1 windows two terms out of three are shared.
    """
    data = np.array(tuple(map(int, open(input_file_path, 'r'))))
    def count_neg_diff_k(raw, k): return sum(raw[k:] - raw[0:-k] > 0)
    r1, r2 = count_neg_diff_k(data, 1), count_neg_diff_k(data, 3)
    return r1, r2


def day2(input_file_path):
    """DAY 2 - 4 LINES - https://adventofcode.com/2021/day/2

    In the second problem, it only changes the meaning of depth. In the first problem depth is really aim.
    """
    totals, effect, data = defaultdict(int), {'down': [1, 0, 0], 'up': [-1, 0, 0], 'forward': [0, 1, 1]}, tuple(map(lambda l: l.strip().split(" "), open(input_file_path, 'r')))
    for cmd, how_much in data:
        for v, q in zip(['aim', 'forward', 'depth'], effect[cmd]): totals[v] += q * int(how_much) * np.power(totals['aim'], int(v == 'depth'))
    r1, r2 = totals['forward'] * totals['aim'], totals['forward'] * totals['depth']
    return r1, r2


def day3(input_file_path):
    """DAY 3 - 5 LINES - https://adventofcode.com/2021/day/3

    """
    data = np.array(tuple(map(lambda l: [int(x) for x in l.strip()], open(input_file_path, 'r'))))
    def gamma(subset, q): return np.array((np.sum(subset, axis=0) >= (subset.shape[0] / 2.)) == q, dtype=np.int)
    def bin2dec(n): return int("".join(map(str, n)), 2)
    def f2(subset, p, q): return bin2dec(subset[0]) if subset.shape[0] == 1 else f2(subset[subset[:, p] == gamma(subset, q)[p], :], p+1, q)
    r1, r2 = bin2dec(gamma(data, 1)) * bin2dec(gamma(data, 0)), f2(data, 0, 1) * f2(data, 0, 0)
    return r1, r2


def day4(input_file_path):
    """DAY 4 - 10 LINES - https://adventofcode.com/2021/day/4
    """
    with open(input_file_path, 'r') as f:
        drawn = f.readline().strip().split(",")
        data = np.array(list(map(lambda s: s.strip().replace("\n", " ").replace("  ", " ").split(" "), f.read().split("\n\n"))))
    r1, r2, winners, c = None, None, np.zeros(data.shape[0]), np.zeros(shape=(data.shape[0], 5, 5))
    def board_score_last_winner(board, drawn_, el_): return sum([int(v) for v in board if v not in drawn_]) * int(el_)
    for i, el in enumerate(drawn):
        for board, pos in zip(*np.where(data == el)): c[board, pos // 5, np.mod(pos, 5)] = 1
        winners[np.where(np.logical_and(winners == 0, np.any(np.logical_or(np.sum(c, axis=1) == 5, np.sum(c, axis=2) == 5), axis=1)))[0]] = i+1
        if r1 is None and sum(winners > 0) > 0: r1 = board_score_last_winner(data[np.argmax(winners)], drawn[:i+1], el)
        if r2 is None and sum(winners > 0) == data.shape[0]: r2 = board_score_last_winner(data[np.argmax(winners)], drawn[:i+1], el)
    return r1, r2


def day5(input_file_path):
    """DAY 5 - 6 LINES - https://adventofcode.com/2021/day/5
    """
    data = np.array(list(map(lambda s: [int(v) for _, v in re.search("(?P<x1>[0-9]+),(?P<y1>[0-9]+) -> (?P<x2>[0-9]+),(?P<y2>[0-9]+)", s.strip()).groupdict().items()], open(input_file_path, 'r'))))
    F = np.zeros((np.max(data)+1, np.max(data)+1, 2))
    for line in data:
        diff_x, diff_y, steps = line[2] - line[0], line[3] - line[1], max(abs(line[2] - line[0]), abs(line[3] - line[1]))
        for k in range(steps+1): F[int(line[1] + k * diff_y / steps), int(line[0] + k * diff_x / steps), int(diff_x == 0 or diff_y == 0)] += 1
    r1, r2 = np.sum(F[:, :, 1] > 1), np.sum(np.sum(F, axis=-1) > 1)
    return r1, r2


def day6(input_file_path):
    """DAY 6 - 5 LINES - https://adventofcode.com/2021/day/6
    """
    r1, r2, data = 0, 0, [np.sum(np.array(list(map(int, open(input_file_path, 'r').read().split(",")))) == i).astype('int64') for i in range(9)]
    for day in range(256):
        data = data[1:7] + [data[7] + data[0]] + [data[8]] + [data[0]]
        r1, r2 = sum(data) if day < 80 else r1, sum(data)
    return r1, r2


def day7(input_file_path):
    """DAY 7 - 3 LINES - https://adventofcode.com/2021/day/7
    """
    data = np.array(list(map(int, open(input_file_path, 'r').read().split(","))))
    def res(input, fun): return min([fun(np.abs(input - i)).sum() for i in range(input.min(), input.max())])
    r1, r2 = res(data, fun=lambda x: x), res(data, fun=lambda x: x * (x + 1) // 2)
    return r1, r2


if __name__ == "__main__":
    # print('DAY 1: ' + str(day1('input_1.txt')))
    # print('DAY 2: ' + str(day2('input_2.txt')))
    # print('DAY 3: ' + str(day3('input_3.txt')))
    # print('DAY 4: ' + str(day4('input_4.txt')))
    # print('DAY 5: ' + str(day5('input_5.txt')))
    # print('DAY 6: ' + str(day6('input_6.txt')))
    print('DAY 7: ' + str(day7('input_7.txt')))
