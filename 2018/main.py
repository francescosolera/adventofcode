from itertools import cycle, product
from collections import defaultdict, Counter

import numpy as np
import re

def day1(input_file_path): # 8 LINES
	data = tuple(map(int, open(input_file_path, 'r')))
	frequency, seen = 0, defaultdict(lambda: False)
	for d in cycle(data):
		frequency += d
		if seen[str(frequency)]: break
		seen[str(frequency)] = True
	return sum(data), frequency

def day2(input_file_path): # 7 LINES
	data = tuple(map(str.strip, open(input_file_path, 'r')))
	def count(n): return sum(map(lambda s: n in Counter(s).values(), data))
	for i in range(26):
		most_common = Counter(map(lambda s: s[:i]+s[i+1:], data)).most_common(1)[0]
		if most_common[1] == 2: break
	return count(2)*count(3), most_common[0]

def day3(input_file_path): # 8 LINES
	def parse(x): return re.search(r'#(?P<id>\d+) @ (?P<x>\d+),(?P<y>\d+): (?P<w>\d+)x(?P<h>\d+)', x).groupdict()
	data = tuple(map(parse, open(input_file_path, 'r')))
	def product_xywh(d): return product(range(int(d['x']), int(d['x'])+int(d['w'])), range(int(d['y']), int(d['y'])+int(d['h'])))
	c = Counter()
	for d in data: c += Counter(product_xywh(d))
	non_overlap = [d for d in data if all([c[pos] == 1 for pos in product_xywh(d)])][0]
	return sum([x > 1 for x in c.values()]), int(non_overlap['id'])

def day4(input_file_path): # 10 LINES
	dd, data = defaultdict(lambda: np.zeros((60, 1))), sorted(map(lambda x: re.search(r'\[(?P<Y>\d+)-(?P<M>\d+)-(?P<D>\d+) (?P<h>\d+):(?P<m>\d+)\] (Guard #(?P<id>\d+)|\w*.*(?P<event>(wakes|asleep)).*\w*)', x).groupdict(), open(input_file_path, 'r')), key=lambda x: (x['Y'], x['M'], x['D'], x['h'], x['m']))
	for d in data:
		if d['id'] is not None: id = d['id']
		if d['event'] == 'asleep': asleep_from = int(d['m'])
		if d['event'] == 'wakes': dd[id][range(asleep_from, int(d['m']))] += 1
	sleepest = [{'id': id, 'overall': np.sum(dd[id]), 'sleepest_minute': np.argmax(dd[id]), 'sleepest_minute_count': np.max(dd[id])} for id in dd.keys()]
	s1 = sorted(sleepest, key=lambda x: x['overall'], reverse=True)[0]
	s2 = sorted(sleepest, key=lambda x: x['sleepest_minute_count'], reverse=True)[0]
	return int(s1["id"])*int(s1["sleepest_minute"]), int(s2["id"])*int(s2["sleepest_minute"])

def day5(input_file_path): # 11 LINES
	data = [ord(x) for x in open(input_file_path, 'r').read().strip()]
	def reduce_polymer(polymer, i=0, last_length=0):
		while (i == 0 and last_length != len(polymer)) or i != 0:
			if i == 0: last_length = len(polymer)
			hit = abs(polymer[i] - polymer[i+1]) == 32
			if hit: del polymer[i+1], polymer[i]
			i = (i - 1 if hit else i + 1) % (len(polymer) - 1)
		return len(polymer)
	def purify_polymer(polymer, unit): return [x for x in polymer if x not in [unit, unit-32]]
	return reduce_polymer(data), min([reduce_polymer(purify_polymer(data, ord('a')+i)) for i in range(26)])


def day6(input_file_path):
	data = tuple(map(lambda x: re.search(r'(?P<x>\d+), (?P<y>\d+)', x).groupdict(), open(input_file_path, 'r')))
	

if __name__ == "__main__":
	# print('DAY 1: ' + str(day1('input_1.txt')))
	# print('DAY 2: ' + str(day2('input_2.txt')))
	# print('DAY 3: ' + str(day3('input_3.txt')))
	# print('DAY 4: ' + str(day4('input_4.txt')))
	print('DAY 5: ' + str(day5('input_5.txt')))
	# print('DAY 6: ' + str(day6('input_6.txt')))