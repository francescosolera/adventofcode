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
	"""DAY 7 - 10 LINES - https://adventofcode.com/2018/day/7

	If a function terminates without a return value, it returns None, as in def "next".
	Such function returns the first element in the alphabet which is not done and not ongoing and has no
	unresolved dependencies linking to it. Ongoing work can be defined as all tasks assigned to elves
	and not found in o2. The line starting with "if next(" could be greatly simplified with assignment
	expressions introduced in python 3.8 PEP 572.
	"""
	elves, o2, r1, r2, data = [{'t': '.', 'r': -1} for _ in range(5)], '.', '', -1, tuple(map(lambda x: re.search(r'Step (?P<a>\w+) must be finished before step (?P<b>\w+) can begin.', x).groupdict(), open(input_file_path, 'r')))
	def next(done, ongoing):
		for a in sorted(set([d for dd in data for d in dd.values()])-set(done)-set(ongoing)):
			if sum(1 for x in data if x['b'] == a and x['a'] not in done) == 0: return a
	while next(r1, '') is not None: r1 += next(r1, '')
	while not all([e['t'] in o2 and e['t'] != '.' for e in elves]):
		for e in [elf for elf in elves if elf['r'] == 0]: o2 += e['t']
		for e in [elf for elf in elves if elf['t'] in o2]:
			if next(o2, ''.join([e['t'] for e in elves if e['t'] not in o2])) is not None: e['t'], e['r'] = next(o2, ''.join([e['t'] for e in elves if e['t'] not in o2])), 60 + ord(next(o2, ''.join([e['t'] for e in elves if e['t'] not in o2]))) - ord('A') + 1
		for e in elves: e['r'], r2 = e['r']-1, r2+1 if e == elves[-1] else r2
	return r1, r2


def day8(input_file_path):
	"""DAY 8 - 7 LINES - https://adventofcode.com/2018/day/8
	
	This is a recursive solution.
	The tree is built in a depth-first fashion where each node is a tuple of children (nodes) or meta values.
	Once the tree is built, recursive functions can be called to explore values of the tree.
	We also have an equally long iterative solution as follows.
	The tree is built in a depth-first fashion (true condition of if).
	As leaves are reached and the traversal goes upstream, each node with all its children explored (else condition of if)
	gets a value computed as i) the sum of its meta and ii) the sum of the values of the children indexed by the meta.
	To ease indexing, each node has an appending naming as follows: name-of-childer = name-of-parent + '.' + child_#
	so that its value can get easily hashed out with the meta indexing of the parent.
	# 1: 	i, all_m, all_v, nodes, data = 0, [], defaultdict(int), [{'id': '0', 'c': 1, 'm': 0, 'd': 0}], [int(x) for x in open(input_file_path, 'r').read().split(' ')]
	# 2:	while i < len(data):
	# 3:		if nodes[-1]['d'] < nodes[-1]['c']: nodes[-1]['d'], i, nodes = nodes[-1]['d'] + 1, i + 2, nodes + [{'id': nodes[-1]['id'] + '.' + str(nodes[-1]['d']+1), 'c': data[i], 'm': data[i+1], 'd': 0}]
	# 4:		else:
	# 5:			i, all_m, all_v[nodes[-1]['id']] = i + nodes[-1]['m'], all_m + [data[i:i+nodes[-1]['m']]], sum(data[i:i+nodes[-1]['m']]) if nodes[-1]['c'] == 0 else sum([all_v[nodes[-1]['id']+'.'+str(j)] for j in data[i:i+nodes[-1]['m']]])
	# 6:			del nodes[-1]
	# 7:	r1, r2 = sum([x for y in all_m for x in y]), all_v['0.1']
	"""
	Tree, data = namedtuple('Tree', ['children', 'meta']), [int(x) for x in open(input_file_path, 'r').read().strip().split(' ')]
	def tree(nums):
		k, n = nums.popleft(), nums.popleft()
		return Tree([tree(nums) for _ in range(k)], [nums.popleft() for _ in range(n)])
	def sum_meta_r1(node): return sum(node.meta) + sum(map(sum_meta_r1, node.children))
	def sum_meta_r2(node): return sum(node.meta) if not node.children else sum([sum_meta_r2(node.children[c-1]) for c in node.meta if c <= len(node.children)])
	r1, r2 = sum_meta_r1(tree(deque(data))), sum_meta_r2(tree(deque(data)))
	return r1, r2


def day9(input_file_path):
	"""DAY 9 - 10 LINES - https://adventofcode.com/2018/day/9
	
	I first tried a solution with list (insert and del at index) but it was way much slower.
	Deques with rotate has to be preferred in this case because append and pop are much faster.
	"""
	data = re.search(r'(?P<n_players>\w+) players; last marble is worth (?P<n_points>\w+) points', open(input_file_path, 'r').read()).groupdict()
	game, players, r1, r2 = deque([0]), defaultdict(int), 0, 0
	for marble in range(1, 100 * int(data['n_points']) + 1):
		game.rotate(-1)
		if marble % 23 == 0:
			game.rotate(8)
			players[marble % int(data['n_players'])] += marble + game.pop()
			game.rotate(-1)
		else: game.append(marble)
		r1, r2 = max(players.values()) if marble == int(data['n_points']) else r1, max(players.values()) if marble == 100 * int(data['n_points']) else r2
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
	print('DAY 9: ' + str(day9('input_9.txt')))
