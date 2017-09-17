import numpy as np
import copy

from node import Node

def reshape(arr):
    """
    Given array, reshapes each element such that they have shape (a,b) instead
    of (b) only """
    L = []
    for each in arr:
        if len(each.shape) == 1:
            L.append(np.reshape(each, [1, -1]))
        else:
            L.append(each)
    return L


def alter_state(state, index, new_entry, change_row=True):
    """
    Returns a copy of `state` where row has been changed to `new_entry` in `index`.
    If `change_row=False`, then the column is changed instead on same location.
    """

    state_copy = copy.deepcopy(state)
    new_entry = copy.deepcopy(new_entry)

    row, col = state_copy

    if change_row:
        row[index] = new_entry
    else:
        col[index] = new_entry

    return state_copy


def build_nonograms_from_state(state, row_lengths, col_lengths, n_cols, n_rows):
    row_segments, col_segments = state

    L_rows = [len(np.array(each).shape) == 1 for each in row_segments]
    L_cols = [len(np.array(each).shape) == 1 for each in col_segments]

    M1, M2 = [], []
    for segment, row_len in zip(row_segments, row_lengths):
        M1.append(binary_map(segment, row_len, n_cols))
    for segment, col_len in zip(col_segments, col_lengths):
        M2.append(binary_map(segment, col_len, n_rows))
    M1 = np.matrix(M1)
    M2 = np.rot90(np.matrix(M2))


def state_is_feasible(state, row_lengths, col_lengths, n_rows, n_cols):
    """
    Check whether a state is feasible or not.

    This is done by checking if there is no row/column
    with zero candidates. That cannot be the case, so the state must
    be infeasible.
    
    """
    rows, cols = state

    M1, M2 = [], []
    S1, S2 = [], []
    rows = reshape(rows)
    cols = reshape(cols)
    for i, each in enumerate(rows):
        X = [binary_map(x, row_lengths[i], n_cols) for x in each]
        s = np.logical_or.reduce(X).astype(np.int64)
        if each.shape[0] != 1:
            line = np.zeros(n_cols)
        else:
            line = binary_map(each[0], row_lengths[i], n_cols)

        M1.append(line)
        S1.append(s)
    # M1 = np.rot90(np.rot90(np.matrix(M1)))

    for j, each in enumerate(cols):
        X = [binary_map(x, col_lengths[j], n_rows) for x in each]
        s = np.logical_or.reduce(X).astype(np.int64)
        if each.shape[0] != 1:
            line = np.zeros(n_rows)
        else:
            line = binary_map(each[0], col_lengths[j], n_rows)

        M2.append(line)
        S2.append(s)

    M1 = np.rot90(np.matrix(M1), k=2).astype(np.int64)
    M2 = np.rot90(np.matrix(M2), k=-1).astype(np.int64)

    S1 = np.rot90(np.matrix(S1), k=2)
    S2 = np.rot90(np.matrix(S2), k=-1)

    # M2 = np.matrix(M2)
    ###import pdb; pdb.set_trace()

    pred1 = np.all(np.multiply(M1, S2) == M1)
    pred2 = np.all(np.multiply(M2, S1) == M2)
    retval= pred1 and pred2
    if retval:
        print('yay')
        import pdb; pdb.set_trace()
    return retval
    ###pass

    ###return all([len(x) != 0 for x in rows] + [len(y) != 0 for y in cols])


def binary_map(line, lengths, num_cells):
    """
    line: [0, 3, 5]
    lengths: [1, 1, 2]
    num_cells: 10
    """
    # if len(line.shape) == 1:
    #     line = np.reshape(line, [1, -1])

    L = np.zeros(num_cells)
    for (variable, length) in zip(line, lengths):
        L[variable:variable + length] = 1
    return L
