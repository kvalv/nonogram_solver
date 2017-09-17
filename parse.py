import numpy as np


def parse_file(filename):
    L = []
    with open(filename) as f:
        n_cols, n_rows = [int(x) for x in f.readline().split(' ')]

        for line in f:
            L.append([int(x) for x in line.split(' ')])

    rows, cols = L[0:n_rows], L[n_rows:]
    return (n_cols, n_rows, cols, rows)


# def state_to_occupancy(state, debug=False):
#     """
#     given a state, return a 6x6 np.ndarray where 1 indicates
#     whether a cell is occupie or not.
# 
#     debug=False -- the entries in the np.ndarray has values 0 if
#     not occupied and 1 if occupied.
# 
#     debug=True -- the entries in the np.ndarray has values corresponding
#     to their car number. Car number 0 has value -1 to differentiate
#     from the other 0-entries.
#     """
#     occupancy_grid = np.matrix(np.zeros((6, 6)))
# 
#     for i, line in enumerate(state):
#         (o, x, y, s) = line
# 
#         if o == 0:  # horizontal
#             if debug:
#                 if i == 0:
#                     occupancy_grid[x:x + s, y] += -1
#                 occupancy_grid[x:x + s, y] += i
#             else:
#                 occupancy_grid[x:x + s, y] += 1
#         else:       # vertical
#             if debug:
#                 if i == 0:
#                     occupancy_grid[x:x + s, y] += -1
#                 occupancy_grid[x, y:y + s] += i
#             else:
#                 occupancy_grid[x, y:y + s] += 1
# 
#     return occupancy_grid.transpose()
