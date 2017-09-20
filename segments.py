import copy
import numpy as np

from exception import InfeasibleStateException


def generate_segments_by_constraints(
        lengths, tot_length, min_index=0,  index=0, the_segment=None):
    """
    Generates segments for lengths[idx]. Assumes i < idx is processed
    and j > idx unprocessed
    """
    gen = generate_segments_by_constraints

    if index == 0:
        the_segment = np.zeros(tot_length, dtype=np.int64)

    length_ahead = np.sum(lengths[index:]) + len(lengths[index:]) - 1
    max_idx = (tot_length - length_ahead).astype(np.int64)

    last_index = index == len(lengths) - 1

    for each in range(min_index, max_idx + 1):
        new_segment = copy.deepcopy(the_segment)
        new_segment[each:each + lengths[index]] += np.int64(1)
        min_index_prime = each + lengths[index] + 1

        if not last_index:
            for next_ in gen(lengths, tot_length, min_index_prime, index + 1, new_segment):
                yield next_
        else:
            yield new_segment


def _find_common_cells(rows_or_columns, cell_value=1):
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

    T = np.array(rows_or_columns)
    if cell_value == 0:
        return np.where(np.logical_not(np.logical_or.reduce(T)))[0]
    else:
        return np.where(np.logical_and.reduce(T))[0]


def keep_by_cell_value(rows_or_columns, index, cell_value):
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
    T = np.array(rows_or_columns)
    retval = T[T[:, index] == cell_value]
    return retval


def find_critical_cells(segment_candidates):
    """
    Finds common 0's and 1's in `segment_candidates`

    Example: segment_candidates = [ [[0, 1, 0], [1, 1, 0]],
                                    [[1, 1, 1]],
                                    [[0, 1, 1], [1, 1, 0]] ]
             --> critical = [[1], [0, 1, 2], [1]]

             returns [(0,1), (1,0), (1,1), (1,2), (2,1)]
    """

    L = []
    for idx, candidates in enumerate(segment_candidates):
        critical_blacks = _find_common_cells(candidates, cell_value=1)
        if len(critical_blacks):
            L.append((critical_blacks, idx, 1))

        critical_whites = _find_common_cells(candidates, cell_value=0)
        if len(critical_whites):
            L.append((critical_whites, idx, 0))

    return L


def enforce_cell_constraints(row_candidates, col_candidates, column_size, row_size,  max_iter=1):
    """
    row_candidates = X
    col_candidates = Y

    Y' = [each where each entry in keep cells that contains critical_cells_of_X]
    X' = [each where each entry in keep cells that contains critical_cells_of_Y']

    returns X', Y'
    """

    rows = copy.deepcopy(row_candidates)
    cols = copy.deepcopy(col_candidates)

    step = 0
    while True:
        step += 1

        _rows = copy.deepcopy(rows)
        _cols = copy.deepcopy(cols)

        for indices, row_idx, cell_value in find_critical_cells(rows):
            for col_idx, each in zip(indices, cols[indices]):
                X = np.array(each)
                filtered = X[X[:, column_size - 1 - row_idx] == cell_value]
                if len(filtered) == 0:
                    raise InfeasibleStateException

                cols[col_idx] = filtered

        for indices, col_idx, cell_value in find_critical_cells(cols):
            row_indices = column_size - 1 - indices
            for row_idx, each in zip(row_indices, rows[row_indices]):
                row_candidates = np.array(each)
                filtered = row_candidates[row_candidates[:, col_idx] == cell_value]
                if len(filtered) == 0:
                    raise InfeasibleStateException

                rows[row_idx] = filtered

        row_unchanged = np.all([np.equal(len(x), len(y)) for x, y in zip(_rows, rows)])
        col_unchanged = np.all([np.equal(len(x), len(y)) for x, y in zip(_cols, cols)])

        if (row_unchanged and col_unchanged) or (step == max_iter):
            return rows, cols


def iter_enforce_cell_constraints(row_candidates, col_candidates, column_size, row_size):

    x, y = None, None
    for i in [3, 1]:
        try:
            x, y = enforce_cell_constraints(row_candidates, col_candidates, column_size, row_size, max_iter=i)
            return (x, y)
        except InfeasibleStateException:
            pass
    raise InfeasibleStateException
