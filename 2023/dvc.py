from collections import defaultdict
from itertools import combinations

def warmup_solla():
    with open("warmup_input.txt") as f:
        data = f.read().splitlines()

    votes = defaultdict(set)
    for line in data:
        person, day = line.split(" ")
        votes[int(day)].add(person)

    sorted_votes = sorted([(k, len(v)) for k, v in votes.items()], key=lambda x: x[1], reverse=True)

    valid_dates = []
    max_partecipations = 0
    for (date1, c1), (date2, c2) in combinations(sorted_votes, 2):
        if c1 < max_partecipations / 2:
            break

        if abs(date1-date2) < 7 or c1 + c2 < max_partecipations:
            continue

        valid_dates.append((min(date1, date2), max(date1, date2), c1, c2))
        max_partecipations = c1 + c2

    print(valid_dates, max_partecipations)
    

def warmup_pietro():
    import pandas as pd
    import itertools

    with open("warmup_input.txt") as f:
        data = f.read()

    names = data.split()[::2]
    days = data.split()[1::2]

    df = pd.DataFrame({"name": names, "day": days})
    df["day"] = df["day"].astype(int)
    votes_per_day = df.groupby("day").sum()["name"].apply(lambda x: len(set(x)))

    best_days = []
    for i in reversed(range(1, votes_per_day.max() + 1)):
        combinations = itertools.combinations(votes_per_day[votes_per_day == i].index, 2)
        for day_couple in combinations:
            if abs(day_couple[1] - day_couple[0]) > 6:
                best_days.append(tuple(sorted(day_couple)))
        if len(best_days):
            print("You can choose among these days:", *best_days)
            break


if __name__ == "__main__":
    warmup_solla()
    # warmup_pietro()