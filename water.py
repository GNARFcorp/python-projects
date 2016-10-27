import math, random, itertools
import numpy as np
field = np.zeros([10, 10], np.dtype("int8"))
field.fill(-1)
field[1:-1,1:-1] = 0
field[0:8,5] = -1
print(field)
for y in range(1,5):
    for x in range(6,9):
        field[y][x] = random.randint(0, 20)


def ndim_grid(start,stop):
    # Set number of dimensions
    ndims = len(start)

    # List of ranges across all dimensions
    L = [np.arange(start[i],stop[i]) for i in range(ndims)]

    # Finally use meshgrid to form all combinations corresponding to all
    # dimensions and stack them as M x ndims array
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

def cell_operation(neighbors):
    bottom = neighbors[2][1]

    #Check if the bottom block is not full
    if neighbors[2][1] == 0:
        neighbors[2][1] = neighbors[1][1]
        neighbors[1][1] = 0
    elif neighbors[2][1] > 0 and neighbors[2][1] <= neighbors[1][1] + 2:
        moved_water = (neighbors[1][1] + 2) - neighbors[2][1]
        if moved_water > neighbors[1][1]:
            moved_water = neighbors[1][1]
        neighbors[2][1] += moved_water
        neighbors[1][1] -= moved_water
    else:
        # euqalize pressure on the plane
        flat_neigh = [[1, 0], [1, 2]]
        # we check randomly through the list
        # TODO: Make it determined
        random.shuffle(flat_neigh)
        for neigh in flat_neigh:
            if neighbors[tuple(neigh)] >= 0:
                diff = neighbors[1][1] - neighbors[tuple(neigh)]
                if diff != 0:
                    if (diff % 2) == 0:
                        neighbors[tuple(neigh)] += diff / 2
                        neighbors[1][1] -= diff / 2
                    else:
                        neighbors[tuple(neigh)] += diff / 2 + 1
                        neighbors[1][1] -= diff / 2
        # push water upwards if pressure difference is greater than 2
        if neighbors[0][1] >= 0 and neighbors[0][1] + 2 <= neighbors[1][1]:
            moved_water = neighbors[1][1] - (neighbors[0][1] + 2)
            neighbors[1][1] -= moved_water
            neighbors[0][1] += moved_water
    return neighbors

for n in range(60):
    # We want to check every second block and alternate between an
    # offset of 0 and 1. We also ignore the borders.
    # It will look like this (. = Ignored, X = Selected):
    # n=1: ..........
    #      .X.X.X.X..
    #      ..X.X.X.X.
    #      .X.X.X.X..
    #      ..X.X.X.X.
    #      ..........

    # n=2: ..........
    #      ..X.X.X.X.
    #      .X.X.X.X..
    #      ..X.X.X.X.
    #      .X.X.X.X..
    #      ..........
    if n > 55:
        print(field)

    for y, plane in enumerate(field[1:-1], start=1):
        for x in range(((y % 2) ^ (n % 2)) + 1, len(field / 2) - 1, 2): # I'm actually sorry for this code; couldn't be bothered to write it cleaner.
            if field[y, x] > 0:
                neighborhood = field[y-1:y+2, x-1:x+2]
                new_neighbors = cell_operation(neighborhood)
                field[y-1:y+2, x-1:x+2] = new_neighbors
print(field)
