import numpy as np
import cv2
from collections import defaultdict


def day1_a():
    values = []

    with open("input_1.txt", "r") as f:
        data = f.readlines()

    for line in data:
        numbers = [c for c in line if c in '0123456789']
        values.append(int(numbers[0]+numbers[-1]))

    print(sum(values))


def day1_b():
    values = []

    data = """two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen""".split("\n")

    with open("input_1.txt", "r") as f:
        data = f.readlines()

    pairs = [('one', '1'),
            ('two', '2'),
            ('three', '3'),
            ('four', '4'),
            ('five', '5'),
            ('six', '6'),
            ('seven', '7'),
            ('eight', '8'),
            ('nine', '9')]

    for line in data:
        line = line.strip()
        original_line = line
        digits = ""

        pos, found = -1, False
        while pos <= len(line) and not found:
            pos += 1
            for word, c in pairs:
                if line[pos:pos+len(word)] == word or line[pos] == c:
                    digits += c
                    found = True
                    break

        pos, found = -1, False
        line = line[::-1]
        while pos <= len(line) and not found:
            pos += 1
            for word, c in pairs:
                if line[pos:pos+len(word)] == word[::-1] or line[pos] == c:
                    digits += c
                    found = True
                    break

        print(original_line, int(digits))
        values.append(int(digits))

    print(sum(values))

def day2_a():
    data = """Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green
""".splitlines()
    
    with open("input_2.txt") as f:
        data = f.readlines()

    games_ok = 0
    max_cubes = {'red': 12, 'blue': 14, 'green': 13}
    for line in data:
        game_id, line = line.strip().split(": ")
        line = line.replace(";", ",")
        for sample in line.split(", "):
            pair = sample.strip().split(" ")
            if max_cubes[pair[1]] < int(pair[0]):
                break
        else:
            games_ok += int(game_id.split(" ")[1])

    print(games_ok)

def day2_b():
    data = """Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green
""".splitlines()
    
    with open("input_2.txt") as f:
        data = f.readlines()

    total_power = 0
    for line in data:
        game_id, line = line.strip().split(": ")
        min_cubes = {'red': 0, 'blue': 0, 'green': 0}
        line = line.replace(";", ",")
        for sample in line.split(", "):
            pair = sample.strip().split(" ")
            min_cubes[pair[1]] = max(min_cubes[pair[1]], int(pair[0]))

        total_power += min_cubes['red'] * min_cubes['green'] * min_cubes['blue']

    print(total_power)

def day3_a():
    data = """467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598..""".splitlines()
    
    with open("input_3.txt") as f:
        data = f.read().splitlines()
    
    total = 0
    n, m = len(data), len(data[0])
    symbol_matrix = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            if data[i][j] not in ".0123456789":
                symbol_matrix[i, j] = 1
                data[i] = data[i][:j] + "." + data[i][j+1:]
    adjacency_matrix = cv2.filter2D(symbol_matrix, -1, np.ones((3, 3)))
    for i in range(n):
        tokens = [el for el in data[i].replace(".", ",.,").split(",") if len(el) > 0]
        pos = 0
        for token in tokens:
            if token != "." and sum(adjacency_matrix[i, pos:pos+len(token)]) > 0:
                total += int(token)

            pos += len(token)

    print(total)

def day3_b():
    data = """467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598..""".splitlines()
    
    with open("input_3.txt") as f:
        data = f.read().splitlines()
    
    gear_values = defaultdict(list)
    n, m = len(data), len(data[0])
    symbol_matrix = np.zeros((n, m), dtype=np.float32)
    gear_code = 1
    for i in range(n):
        for j in range(m):
            if data[i][j] == "*":
                symbol_matrix[i, j] = gear_code
                gear_code += 1
            if data[i][j] not in ".0123456789":
                data[i] = data[i][:j] + "." + data[i][j+1:]
    adjacency_matrix = cv2.filter2D(symbol_matrix, -1, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT, anchor=(-1, -1))
    for i in range(n):
        tokens = [el for el in data[i].replace(".", ",.,").split(",") if len(el) > 0]
        pos = 0
        for token in tokens:
            if token == ".":
                pos += 1
                continue
            gear_value = np.max(adjacency_matrix[i, pos:pos+len(token)])
            if gear_value > 0:
                gear_values[gear_value].append(int(token))

            pos += len(token)

    total = 0
    for l in gear_values.values():
        if len(l) == 2:
            total += l[0] * l[1]

    print(total)

if __name__ == "__main__":
    # day1_b()
    # day2_b()
    # day3_a()
    day3_b()
