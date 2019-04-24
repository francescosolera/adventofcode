from itertools import cycle, product, chain
from collections import defaultdict, Counter

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
	"""DAY 1 - 7 LINES - https://adventofcode.com/2018/day/1

	As I cycle through data, I use a dictionary as a hash map to return the first reappearing frequency (cumulative sum).
	"""
	data = tuple(map(int, open(input_file_path, 'r')))
	frequency, seen = 0, defaultdict(lambda: False)
	for d in cycle(data):
		frequency += d
		if seen[str(frequency)]: break
		seen[str(frequency)] = True
	r1, r2 = sum(data), frequency
	return r1, r2


def day2(input_file_path):
	"""DAY 2 - 6 LINES - https://adventofcode.com/2018/day/2

	The function count returns the number of elements counting to exactly a given cardinality.
	The loop checks if any two strings in data match by removing one different column at each iteration.
	"""
	data = tuple(map(str.strip, open(input_file_path, 'r')))
	def count(n): return sum(map(lambda s: n in Counter(s).values(), data))
	for i in range(26):
		most_common = Counter(map(lambda s: s[:i]+s[i+1:], data)).most_common(1)[0]
		if most_common[1] == 2: break
	r1, r2 =  count(2)*count(3), most_common[0]
	return r1, r2


def day3(input_file_path):
	"""DAY 3 - 4 LINES - https://adventofcode.com/2018/day/3

	The loop increments the numpy array counter inside the area defined by all elements d in data.
	Then I look for the one element d in data which does not overlap (has all 1 counts in its area).
	I also tested a solution without numpy, based on Counter and product(range, range), but it was so much slower.
	"""
	c, data = np.zeros((1000, 1000), dtype=int), tuple(map(lambda x: re.search(r'#(?P<id>\d+) @ (?P<x>\d+),(?P<y>\d+): (?P<w>\d+)x(?P<h>\d+)', x).groupdict(), open(input_file_path, 'r')))
	for d in data: c[int(d['x']): int(d['x'])+int(d['w']), int(d['y']): int(d['y'])+int(d['h'])] +=1
	non_overlap = [d for d in data if np.all(c[int(d['x']): int(d['x'])+int(d['w']), int(d['y']): int(d['y'])+int(d['h'])] == 1)][0]
	r1, r2 = np.sum(c>1), int(non_overlap['id'])
	return r1, r2


def day4(input_file_path):
	"""DAY 4 - 9 LINES - https://adventofcode.com/2018/day/4

	Parse the input text file and sort lines by date.
	Each guard has a dict entry counting the number of times he was asleep in the minutes from 00:00 to 00:59.
	"""
	dd, data = defaultdict(lambda: np.zeros((60, 1))), sorted(map(lambda x: re.search(r'\[(?P<Y>\d+)-(?P<M>\d+)-(?P<D>\d+) (?P<h>\d+):(?P<m>\d+)\] (Guard #(?P<id>\d+)|\w*.*(?P<event>(wakes|asleep)).*\w*)', x).groupdict(), open(input_file_path, 'r')), key=lambda x: (x['Y'], x['M'], x['D'], x['h'], x['m']))
	for d in data:
		if d['id'] is not None: id = d['id']
		if d['event'] == 'asleep': asleep_from = int(d['m'])
		if d['event'] == 'wakes': dd[id][range(asleep_from, int(d['m']))] += 1
	sleepest = [{'id': id, 'overall': np.sum(dd[id]), 'sleepest_minute': np.argmax(dd[id]), 'sleepest_minute_count': np.max(dd[id])} for id in dd.keys()]
	s1 = sorted(sleepest, key=lambda x: x['overall'], reverse=True)[0]
	s2 = sorted(sleepest, key=lambda x: x['sleepest_minute_count'], reverse=True)[0]
	r1, r2 = int(s1["id"])*int(s1["sleepest_minute"]), int(s2["id"])*int(s2["sleepest_minute"])
	return r1, r2

def day5(input_file_path):
	"""DAY 5 - 10 LINES - https://adventofcode.com/2018/day/5

	The function reduce_polymer (modulo) loops through the polymer.
	When we find two consecutive chars whose ord distance is 32 (upper case and lower case), we remove them from the list.
	We then take one step back since the reduction may have enabled other two reacting units to touch.
	Every time we get to the start of the sequence, we check whether we have converged.
	I also tested a solution using re.subn() instead of list del(), but the size of the string and the number of times it was copied made it very inefficient.
	"""
	data = [ord(x) for x in open(input_file_path, 'r').read().strip()]
	def reduce_polymer(polymer, i=0, last_length=0):
		while (i == 0 and last_length != len(polymer)) or i != 0:
			if i == 0: last_length = len(polymer)
			hit = abs(polymer[i] - polymer[i+1]) == 32
			if hit: del polymer[i+1], polymer[i]
			i = (i - 1 if hit else i + 1) % (len(polymer) - 1)
		return len(polymer)
	def purify_polymer(polymer, unit): return [x for x in polymer if x not in [unit, unit-32]]
	r1, r2 = reduce_polymer(data), min([reduce_polymer(purify_polymer(data, ord('a')+i)) for i in range(26)])
	return r1, r2


def day6(input_file_path):
	"""DAY 6 - 9 LINES - https://adventofcode.com/2018/day/6

	The key insight is that a voronoi region is infinite if it touches the side of the bounding box tight around all points.
	"""
	data = [(int(x.split(', ')[0]), int(x.split(', ')[1])) for x in open(input_file_path, 'r')]
	counts, region, x_min, x_max, y_min, y_max = np.zeros(len(data)), 0, min([el[0] for el in data]), max([el[0] for el in data]), min([el[1] for el in data]), max([el[1] for el in data])
	def distance(x, y): return [abs(x-i) + abs(y-j) for i, j in data]
	inf = set([np.argmin(distance(i, j)) for i,j in chain(product([x_min], range(y_min, y_max)), product([x_max], range(y_min, y_max)), product(range(x_min, x_max), [y_min]), product(range(x_min, x_max), [y_max]))])
	for pos in product(range(x_min, x_max), range(y_min, y_max)):
		s = distance(pos[0], pos[1])
		if np.argmin(s) not in inf and s.count(min(s)) == 1: counts[np.argmin(s)] += 1
		if sum(s) < 10000: region += 1
	r1, r2 = int(max(counts)), region
	return r1, r2


def day7(input_file_path):
	"""DAY 7 - 9 LINES - https://adventofcode.com/2018/day/7

	The key insight is that a voronoi region is infinite if it touches the side of the bounding box tight around all points.
	"""
	data = tuple(map(lambda x: re.search(r'Step (?P<a>\w+) must be finished before step (?P<b>\w+) can begin.', x).groupdict(), open(input_file_path, 'r')))
	print(data)

	return r1, r2


if __name__ == "__main__":
	# print('DAY 1: ' + str(day1('input_1.txt')))
	# print('DAY 2: ' + str(day2('input_2.txt')))
	# print('DAY 3: ' + str(day3('input_3.txt')))
	# print('DAY 4: ' + str(day4('input_4.txt')))
	# print('DAY 5: ' + str(day5('input_5.txt')))
	# print('DAY 6: ' + str(day6('input_6.txt')))
	print('DAY 7: ' + str(day7('input_7.txt')))