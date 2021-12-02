from itertools import cycle, product, chain
from collections import defaultdict, Counter, namedtuple, deque

import numpy as np
import re

'''DESIGN RULE OF THUMBS:
- keep all functions below 10 lines of code (def line and return line excluded)
    - def line is always: def dayX(input_file_path)
    - return line is always: return r1, r2
- no solutions should be unreasonably slow
- use numpy only when we know the allocation size in advance, otherwise use python data structures
- define a function if it's called more than once
- define a variable if it's used more than once
'''


def day1(input_file_path):
    """DAY 1 - 4 LINES - https://adventofcode.com/2021/day/1

    Compunting the difference of a sum/average over a sliding window let us cancel out the terms shared by the two
    windows; in the case of a 3x1 windows two terms out of three are shared.
    """
    data = np.array(tuple(map(int, open(input_file_path, 'r'))))
    def count_neg_diff_k(raw, k):
        return sum(raw[k:] - raw[0:-k] > 0)
    return count_neg_diff_k(data, 1), count_neg_diff_k(data, 3)


def day2(input_file_path):
    """DAY 2 - 7 LINES - https://adventofcode.com/2021/day/2

    In the second problem, it only changes the meaning of depth. In the first problem depth is really aim.
    """
    data = tuple(map(lambda l: l.strip().split(" "), open(input_file_path, 'r')))
    totals = defaultdict(int)
    effect = {'down': [1, 0, 0], 'up': [-1, 0, 0], 'forward': [0, 1, 1]}
    for cmd, how_much in data:
        for v, q in zip(['aim', 'forward', 'depth'], effect[cmd]):
            totals[v] += q * int(how_much) * np.power(totals['aim'], int(v == 'depth'))
    return totals['forward'] * totals['aim'], totals['forward'] * totals['depth']


if __name__ == "__main__":
    # print('DAY 1: ' + str(day1('input_1.txt')))
    print('DAY 2: ' + str(day2('input_2.txt')))
