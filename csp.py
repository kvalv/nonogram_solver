# begin
import numpy as np

import utils
from parse import parse_file
from a_star import A_Star
from node import Node
# end


# how to represent state?
# state = (row_segments, column_segments)
# where each is a list of segments

class Nonogram_Solver(A_Star):

    def __init__(self, heuristic_fun, n_cols, n_rows, col_lengths, row_lengths, window=None):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.col_lengths, self.row_lengths = col_lengths, row_lengths

        super().__init__(heuristic_fun, window)

    def generate_children(self, parent):
        """
        from `parent.state`, there exists at least one row / column with
        several segments. Make a child state for each such row/column
        where a single row/column is chosen.

        returns a generator for children
        """
        def make_node(from_state, parent):
            node = Node(from_state)
            self._attach_and_eval(node, parent)
            return node

        import pdb; pdb.set_trace()
        rows, cols = parent.state

        def get_lengths(n):
            return [len(x) for x in n] 

        n_candidates = sum(get_lengths(rows)) + sum(get_lengths(cols))
        print(f'number of candidates: {n_candidates}')


        for (j, col) in enumerate(cols):
            if len(col.shape) == 1:
                col = np.reshape(col, [1, -1])

            num_segments = len(col)
            # if num_segments == 1:
            #     continue

            for candidate in col:
                new_state =  utils.alter_state(parent.state, j, candidate, change_row=False)

                rows, cols = new_state
                # cols = self.filter_columns(cols, rows)
                import copy
                old_rows,old_cols = copy.deepcopy(rows), copy.deepcopy(cols)
                rows = self.filter_rows(rows, cols)
                new_state = (rows, cols)

                if utils.state_is_feasible(new_state, self.row_lengths, self.col_lengths, self.n_rows, self.n_cols):
                    yield make_node(new_state, parent)

        import pdb; pdb.set_trace()
        for (i, row) in enumerate(rows):
            if len(row.shape) == 1:
                row = np.reshape(row, [1, -1])

            num_segments = len(row)
            # if num_segments == 1:
            #     continue

            prev_state = copy.deepcopy(parent.state)
            for candidate in row:  # a specific row is chosen
                if type(candidate) != np.ndarray:
                    import pdb; pdb.set_trace()

                new_state = utils.alter_state(parent.state, i, candidate, change_row=True)
                
                rows, cols = new_state
                cols = self.filter_columns(cols, rows)
                rows = self.filter_rows(rows, cols)
                new_state = (rows, cols)
                if utils.state_is_feasible(new_state, self.row_lengths, self.col_lengths, self.n_rows, self.n_cols):
                    yield make_node(new_state, parent)

    def cost_fun(self, parent, child):
        return 1  # simply one more move from `parent` to `child`

    def goal_fun(self, node):
        row_segments = node.state[0]
        col_segments = node.state[1]
        ### row_segments = np.array([[0],[1,4],[1,4],[1,4],[0,2,5],[1,4],[2]])
        ### col_segments = np.array([[2,6],[1,3],[0,2,6],[0,2,6],[1,3],[2,6]])

        L_rows = [len(np.array(each).shape) == 1 for each in row_segments]
        L_cols = [len(np.array(each).shape) == 1 for each in col_segments]

        ###comb = [int(x) for x in L_rows + L_cols]
        ###average = sum(comb) / len(comb)
        ###print(average)
        ###if average > 0.4:
        ###    import pdb; pdb.set_trace()

        if not all(L_rows):
            return False
        if not all(L_cols):
            return False
        import pdb; pdb.set_trace()

        nono_rows, nono_cols = utils.build_nonograms_from_state(
            node.state, self.row_lengths, self.col_lengths,
            self.n_cols, self.n_rows)
        retval = np.all(nono_rows == nono_cols)

        ###M1, M2 = [], []
        ###for segment, row_len in zip(row_segments, self.row_lengths):
        ###    M1.append(utils.binary_map(segment, row_len, self.n_cols))
        ###for segment, col_len in zip(col_segments, self.col_lengths):
        ###    M2.append(utils.binary_map(segment, col_len, self.n_rows))
        ###M1 = np.matrix(M1)
        ###M2 = np.rot90(np.matrix(M2))
        ###retval = np.all(M1 == M2)
        if retval:
            import pdb; pdb.set_trace()

        return retval

    def generate_initial_state(self):
        row_segments = [self.generate_segments(self.row_lengths[i], self.n_cols)
                        for i in range(self.n_rows)]

        column_segments = [self.generate_segments(self.col_lengths[j], self.n_rows)
                           for j in range(self.n_cols)]

        return [row_segments, column_segments]

    def generate_segments(self, lengths, num_cells):
        """
        usage
         row_segments = [generate_segments(rows[i], n_cols)
                           for i in range(n_rows)]
         column_segments = [generate_segments(cols[j], n_rows)
                              for j in range(n_cols)]
        """
        final_output = []

        def generate_domain(prev_segments, lengths, num_cells, i):
            """
            prev_segments: cell location of beginning of preceding segment
            lengths: list of [len(1), ... len(n)] of all n variables
            num_cells: the number of cells in a given row / column
            i: the variable that we'll look at

            returns [x1, .., xm], the domain of allowed values for segment i
            """

            if len(prev_segments) == 0:
                earliest_idx = 0
            else:
                earliest_idx = prev_segments[-1] + lengths[i - 1] + 1

            remaining = lengths[i:]
            if len(remaining) == 1:  # special case, only one left
                latest_idx = num_cells - remaining[0]
            else:
                latest_idx = num_cells - (sum(remaining[1:]) + len(remaining))

            return [b for b in range(earliest_idx, latest_idx + 1)]

        prev_segments = []
        queue = [(0, [])]

        for (i, prev_segments) in queue:
            if i > len(lengths) - 1:  # all segments have been placed, skip
                continue

            for each in generate_domain(prev_segments, lengths, num_cells, i):
                new_segments = prev_segments + [each]
                if i == len(lengths) - 1:  # done
                    final_output.append(new_segments)
                queue.append((i + 1, new_segments))

        final_output = np.array(final_output)
        if len(final_output.shape) == 1:
            import pdb; pdb.set_trace()

        return np.array(final_output)

    def common_indices(self, segments, lengths, num_cells):
        """
        Given a set of segments, return the index of common segments

        """
        C = np.tile(segments, [1,1])
        X = [utils.binary_map(each, lengths, num_cells) for each in C]
        return np.where(np.logical_and.reduce(X))[0]

        ###if len(segments) == 0:
        ###    return np.array([])

        ###if len(segments.shape) == 1:
        ###    segments = np.reshape(segments, [1, -1])

        ###M = np.array([utils.binary_map(line, lengths, num_cells) for line in segments])
        ###if len(M) == 0:
        ###    return np.array([])

        ###common_segments = np.all(M, axis=0, out=np.zeros(M.shape[1]))

        ###if not any(common_segments):
        ###    return np.array([])
        ###return np.where(common_segments == 1)[0]

    def filter_by_cell_occupancy(self, segments, lengths, num_cells, index):
        """
        Removes all entries in `segments` that do not have value 1 in `index`

        `segments` (rows), find those who have common entries. Return the
        index of those. `lengths` and `num_cells` are used to generate maps

        `index` is the value where all segments must match
        """

        if len(segments) == 0:
            return np.array([])

        if len(segments.shape) == 1:
            segments = np.reshape(segments, [1, -1])


        args = segments, lengths, num_cells
        for each in segments:
            if type(each) == np.int64:
                import pdb; pdb.set_trace()
        maps = np.array([utils.binary_map(each, lengths, num_cells)
                         for each in segments])

        if maps.shape[0] == 1:
            filtered_idx = maps == 1
        filtered_idx = maps[:, index] == 1

        if not any(filtered_idx):
            return np.array([])

        return segments[filtered_idx, :]

    def filter_rows(self, row_segments, column_segments):
        col_idx = range()
        keep_rows = []

        for col_idx, segments, col_length in zip(range(self.n_cols), column_segments, self.col_lengths):
            import pdb; pdb.set_trace()
            segments = [np.tile(each, [1, 1]) for each in segments]

            inp = [np.reshape(each, [-1]) for each in segments]
            X = [utils.binary_map(np.reshape(each, [-1]), col_length, self.n_rows) for each in segments]
            common_indices = np.where(np.logical_or.reduce(X))[0]

            row_indices = self.n_rows - 1 - common_indices
            for row_idx in row_indices:
                the_rows = [utils.binary_map(np.reshape(each, [-1]), col_length, self.n_rows) for each in row_segments[row_idx]]
                keep_rows = row_segments[row_idx][np.where(the_rows[col_idx] == 1)[0]]
                row_segments[row_idx] 
                row_segments[row_idx]
            row_segments[row_indices]




        ###for j, (col_length, columns) in enumerate(zip(self.col_lengths, column_segments)):
        ###    indices = self.common_indices(columns, col_length, self.n_rows)

        ###    for each in indices:
        ###        ## import pdb; pdb.set_trace()
        ###        row_number = self.n_cols - each
        ###        X = row_segments[each]

        ###        # index here is j -- why is that the case?
        ###        args = (row_segments[row_number],self.row_lengths[row_number],self.n_cols,j)
        ###        if type(args[0]) == np.int64:
        ###            import pdb; pdb.set_trace()
        ###        segments_filtered = self.filter_by_cell_occupancy(
        ###            row_segments[row_number],
        ###            self.row_lengths[each],
        ###            self.n_cols,  # TODO: or is this n_rows?
        ###            j)

        ###        if len(segments_filtered) == 0:
        ###            row_segments[row_number] = segments_filtered
        ###            return row_segments

        ###        if type(segments_filtered) != np.ndarray:
        ###            import pdb; pdb.set_trace()

        ###        previous = row_segments[row_number]
        ###        new = segments_filtered
        ###        row_segments[row_number] = segments_filtered

        ###return row_segments

    def filter_columns(self, column_segments, row_segments):

        for j, (row_length, rows) in enumerate(zip(self.row_lengths, row_segments)):
        # for (i, segments) in enumerate(row_segments):

            # common indices for row
            indices = self.common_indices(rows, row_length, self.n_cols)
            # if len(indices) == 0:
            #     raise Exception('no feasible solution')

            for each in indices:
                col_number = each
                # col_number = n_rows - each

                # import pdb; pdb.set_trace()
                if len(column_segments[col_number].shape) == 1:
                    column_segments[col_number] = np.reshape(column_segments[col_number], [1, -1])

                args = (column_segments[col_number], self.col_lengths[col_number], self.n_rows, j, col_number)
                if len(args[0].shape) == 1:
                    import pdb; pdb.set_trace()

                segments_filtered = self.filter_by_cell_occupancy(
                    column_segments[col_number],
                    self.col_lengths[col_number],
                    self.n_rows,
                    j)

                previous = column_segments[each]
                new = segments_filtered

        return column_segments
