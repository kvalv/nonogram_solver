import numpy as np


def heuristic_fun(node):
    return 0
    x, y = node.state
    candidate_lengths = np.array([len(e) for e in np.append(x, y)])
    # return 0
    return np.log(candidate_lengths).sum()
