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
    """DAY 6 - 4 LINES - https://adventofcode.com/2021/day/6
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


def day8(input_file_path):
    """DAY 8 - 10 LINES - https://adventofcode.com/2021/day/8
    """
    original, data = {'abcefg': 0, 'cf': 1, 'acdeg': 2, 'acdfg': 3, 'bcdf': 4, 'abdfg': 5, 'abdefg': 6, 'acf': 7, 'abcdefg': 8, 'abcdfg': 9}, list(map(lambda s: s.strip().replace(" |", "").split(" "), open(input_file_path, 'r')))
    def get_chars_by_digit_length(entry, digit): return [x for x in entry if len(x) == {v: len(k) for k, v in original.items()}[digit]]
    def get_chars_by_freq(entry, digit): return [k for k, v in Counter("".join(entry[:10])).items() if v == Counter("".join(list(original.keys())))[digit]]
    r1, r2 = sum([len(get_chars_by_digit_length(entry[-4:], digit)) for entry in data for digit in [1, 4, 7, 8]]), 0
    for entry in data:
        dg = {k: set(get_chars_by_digit_length(entry[:10], k)[0]) for k in [1, 4, 7, 8]}
        ch = {k: get_chars_by_freq(entry[:10], k)[0] for k in ['b', 'e', 'f']}
        ch.update({'a': dg[7].difference(dg[1]).pop(), 'c': dg[1].difference(ch['f']).pop(), 'd': dg[4].difference(dg[7]).difference(ch['b']).pop(), 'g': dg[8].difference(dg[4]).difference(dg[7]).difference(ch['e']).pop()})
        ch = {v: k for k, v in ch.items()}
        r2 += sum([original["".join(sorted([ch[x] for x in entry[-i-1]]))] * (10 ** i) for i in range(4)])
    return r1, r2


def day9(input_file_path):
    """DAY 9 - 8 LINES - https://adventofcode.com/2021/day/9

    In second step, use union (quick-find) to merge elements not split by a 9 and find connected components.
    """
    data = np.loadtxt(input_file_path, converters={0: lambda line: [int(x) for x in line.decode()]}, dtype=np.int)
    h, w = data.shape
    def get_neighbors(i, j): return list(set([(max(0, i-1), j), (min(h-1, i+1), j), (i, max(0, j-1)), (i, min(w-1, j+1))]).difference([(i, j)]))
    r1 = sum([data[i, j] + 1 for i, j in [(i, j) for i, j in product(range(h), range(w)) if data[i, j] < min([data[ii, jj] for ii, jj in get_neighbors(i, j)])]])
    idx = np.array([i for i in range(h * w)]).reshape((h, w))
    for i, j in ((i_, j_) for i_, j_ in product(range(h), range(w)) if data[i_, j_] < 9):
        for ii, jj in ((ii_, jj_) for ii_, jj_ in get_neighbors(i, j) if data[ii_, jj_] < 9): idx[idx == idx[i, j]] = idx[ii, jj]
    r2 = np.prod([x[1] for x in Counter(idx.ravel()).most_common(3)])
    return r1, r2


def day10(input_file_path):
    """DAY 10 - 9 LINES - https://adventofcode.com/2021/day/10
    """
    r1, r2, lifo, data, corr = 0, [], [], list(map(lambda line: [ch for ch in line.strip()], open(input_file_path, "r"))), {"(": ")", "[": "]", "{": "}", "<": ">"}
    for line in data:
        for i, char in enumerate(line):
            if char in corr.keys(): lifo.append(char)
            elif corr[lifo.pop()] != char:
                r1, lifo = r1 + {")": 3, "]": 57, "}": 1197, ">": 25137}[char], []
                break
        if len(lifo) > 0: r2.append(sum([{"(": 1, "[": 2, "{": 3, "<": 4}[lifo.pop()] * (5 ** (len(lifo))) for _ in range(len(lifo))]))
    r2 = sorted(r2)[len(r2) // 2]
    return r1, r2


def day11(input_file_path):
    """DAY 11 - 10 LINES - https://adventofcode.com/2021/day/11
    """
    r1, r2, data = 0, 0, np.loadtxt(input_file_path, converters={0: lambda line: [v for v in line.decode()]}, dtype=np.int)
    while np.any(data > 0):
        data, checked, changed, r2 = data + 1, np.zeros_like(data), True, r2 + 1
        while changed:
            changed = False
            for i, j in [(ii, jj) for ii, jj in product(range(data.shape[0]), range(data.shape[1])) if data[ii, jj] > 9 and checked[ii, jj] == 0]:
                changed, checked[i, j], data[max(0, i - 1):i + 2, max(0, j - 1): j + 2] = True, checked[i, j] + 1, data[max(0, i - 1):i + 2, max(0, j - 1): j + 2] + 1
                break
        if r2 <= 100: r1 += np.sum(checked)
        data[data > 9] = 0
    return r1, r2


def day12(input_file_path):
    """DAY 12 - ? LINES - https://adventofcode.com/2021/day/12

    really bad and unoptimized code; haven't found right data scruture yet
    """
    r1, r2 = None, None
    data = list(map(lambda line: line.strip().split("-"), open(input_file_path, "r")))
    for i in range(len(data)): data[i] = data[i][::-1] if data[i][1] == "start" or data[i][0] == "end" else data[i]
    small_caves = set([x[0] for x in data if x[0] == x[0].lower()] + [x[1] for x in data if x[1] == x[1].lower()])
    visited_caves, changed = [link for link in data if "start" in link], True
    visited_caves_dict = defaultdict(int)
    while changed:
        changed = False
        for _ in range(len(visited_caves)):
            path = visited_caves.pop(0)
            if "end" in path:
                visited_caves.append(path)
                continue

            for _, new_el in [[a, b] if a == path[-1] else [b, a] for a, b in data if path[-1] in [a, b]]:
                if (new_el in small_caves and new_el in path) and (new_el in ["start", "end"] or (Counter([sc for sc in path if sc in small_caves]).most_common()[0][1] > 1)):
                    continue
                new_path = path + [new_el]
                if "_".join(new_path) not in visited_caves_dict:
                    visited_caves.append(new_path)
                    visited_caves_dict["_".join(new_path)] = -1
                    changed = True
    print(len(visited_caves))
    return r1, r2


if __name__ == "__main__":
    # print('DAY 1: ' + str(day1('input_1.txt')))
    # print('DAY 2: ' + str(day2('input_2.txt')))
    # print('DAY 3: ' + str(day3('input_3.txt')))
    # print('DAY 4: ' + str(day4('input_4.txt')))
    # print('DAY 5: ' + str(day5('input_5.txt')))
    # print('DAY 6: ' + str(day6('input_6.txt')))
    # print('DAY 7: ' + str(day7('input_7.txt')))
    # print('DAY 8: ' + str(day8('input_8.txt')))
    # print('DAY 9: ' + str(day9('input_9.txt')))
    # print('DAY 10: ' + str(day10('input_10.txt')))
    # print('DAY 11: ' + str(day11('input_11.txt')))
    print('DAY 12: ' + str(day12('input_12.txt')))
