# begin

import copy
import itertools
import numpy as np

from exception import InfeasibleStateException


def bitmap(indices, lengths, tot_length):
    L = np.zeros(tot_length)
    for i, l in zip(indices, lengths):
        L[i:i + l] = 1
    return L.astype(np.int64)


def generate_segments_by_constraints(lengths, tot_length, from_index=0,  index=0, the_segment=None):
    """
    Generates segments for lengths[idx]. Assumes i < idx is processed
    and j > idx unprocessed
    """

    if index == 0:
        the_segment = np.zeros(tot_length, dtype=np.int64)

    min_idx = np.int64(from_index)

    length_ahead = np.sum(lengths[index:]) + len(lengths[index:]) - 1
    max_idx = (tot_length - length_ahead).astype(np.int64)

    last_index = index == len(lengths) - 1

    for each in range(min_idx, max_idx + 1):
        new_segment = copy.deepcopy(the_segment)
        new_segment[each:each + lengths[index]] += np.int64(1)
        from_index_prime = each + lengths[index] + 1

        if not last_index:
            for var in generate_segments_by_constraints(lengths, tot_length, from_index_prime, index + 1, new_segment):
                yield var
        else:
            yield new_segment
            # yield np.tile(new_segment, [1, 1])

        # raise StopIteration


def find_common_cells(rows_or_columns, cell_value=1):
    """
    Given candidates for a row / column, find all the indices where the rows / columns
    all have `cell_value`.

    Example:
    rows_or_columns = [ [1,0,0,1,0],
                        [1,0,1,1,0],
                        [1,1,0,0,0] ]
    cell_value = 1

    returns: np.array([0])
    """

    X = np.vstack(rows_or_columns)
    if cell_value == 0:
        retval =  np.where(np.logical_not(np.logical_or.reduce(X)))[0]
        if retval:
            print('yay')
        return retval

    return np.where(np.logical_and.reduce(X))[0]


def keep_by_cell_value(rows_or_columns, index, cell_value=1):
    """
    Given candidates for a row / column, filter out all those rows/columns
    that do not have `cell_value` on the index `index`.

    Example:
    rows_or_columns = [ [1,0,0,1,0],
                        [1,0,1,1,0],
                        [1,1,0,0,0] ]
    index = 3
    cell_value = 1

    returns: [ [1,0,0,1,0],
               [1,0,1,1,0] ]

    """

    # old version. 0.697 --> 0.187 with the new one
    # X = np.vstack(rows_or_columns)
    # indices = np.where(X[:, index] == cell_value)[0]
    # return X[indices, :]

    T = np.array(rows_or_columns)
    return T[np.where(T[:,index])[0]]


def enforce_cell_constraints(row_candidates, col_candidates, column_size, row_size, cell_value=1):
    """
    row_candidates = X
    col_candidates = Y

    Y' = [each where each entry in keep cells that contains critical_cells_of_X]
    X' = [each where each entry in keep cells that contains critical_cells_of_Y']

    returns X', Y'
    """

    rows = copy.deepcopy(row_candidates)
    cols = copy.deepcopy(col_candidates)

    def find_critical_cells(segment_candidates, cell_value):
        """
        Example: segment_candidates = [ [[0, 1, 0], [1, 1, 0]],
                                        [[1, 1, 1]],
                                        [[0, 1, 1], [1, 1, 0]] ]
                 --> critical = [[1], [0, 1, 2], [1]]

                 returns [(0,1), (1,0), (1,1), (1,2), (2,1)]
        """

        L = []
        for idx, candidates in enumerate(segment_candidates):
            critical_indices = find_common_cells(candidates, cell_value)
            entry = (idx, critical_indices)
            L.append(entry)

        retval = itertools.chain.from_iterable(
            [itertools.product([x], y) for x, y in L])
        return retval

    while True:

        _rows = copy.deepcopy(rows)
        _cols = copy.deepcopy(cols)

        for (row_idx, col_idx) in find_critical_cells(rows, cell_value):
            cell_index = column_size - row_idx  - 1 # rebase for columns
            new = keep_by_cell_value(cols[col_idx], cell_index, cell_value)
            if len(new) == 0:
                raise InfeasibleStateException

            cols[col_idx] = new

        for (row_idx, col_idx) in find_critical_cells(cols, cell_value):
            row_idx, cell_index = column_size - 1 - col_idx, row_idx
            new = keep_by_cell_value(rows[row_idx], cell_index, cell_value)
            if len(new) == 0:
                raise InfeasibleStateException

            rows[row_idx] = new

        # import pdb; pdb.set_trace()

        L = [np.equal(len(x),len(y)) for x,y in zip(_rows, rows)]
        row_unchanged = np.all(L)
        L = [np.equal(len(x),len(y)) for x,y in zip(_cols, cols)]
        col_unchanged = np.all(L)

        if row_unchanged and col_unchanged:
            return rows, cols
