import numpy as np
import copy

from node import Node
import segments as sgm
from a_star import A_Star
from exception import InfeasibleStateException


class Nonogram_Solver(A_Star):

    def __init__(self, heuristic_fun, n_cols, n_rows, col_lengths, row_lengths, window=None):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.col_lengths, self.row_lengths = col_lengths, row_lengths
        super().__init__(heuristic_fun, window)

    def generate_children(self, parent):
        """
        step 1: find all critical cell of row. this is found by finding 'common_indices'
        step 2: filter out all columns based on these critical cells.
        step 3: find all critical cell of column. this is found by finding 'common_indices'
        step 4: filter out all columns based on these critical cells.
        step 5: for each possible candidate in all rows / columns -- enforce it to single
        """

        rows, cols = parent.state

        def make_node(from_state, parent):
            node = Node(from_state)
            # self._attach_and_eval(node, parent)
            return node

        column_size = self.n_rows
        row_size = self.n_cols
        try:
            rows_prime, cols_prime = sgm.enforce_cell_constraints(rows, cols, column_size, row_size, max_iter=1)
            rows, cols = rows_prime, cols_prime
        except InfeasibleStateException:
            pass

        # step 5
        def try_yield(rows, cols, column_size, row_size):
            """
            Tries to enforce cell constraints with step i. If all steps fail,
            raise InfeasibleStateException.
            """
            x, y = None, None
            for i in [3, 1]:
                try:
                    x, y = sgm.enforce_cell_constraints(rows, cols, column_size, row_size, max_iter=i)
                    break
                except InfeasibleStateException:
                    pass

            if x is None:
                raise InfeasibleStateException

            N = make_node([x, y], parent)
            return N

        def fewest_candidates(lines):
            """
            Find the index of most promising entry based on smallest entry length (except 1).
            """
            lengths = np.array([len(e) for e in lines])
            lengths[lengths == 1] = max(lengths)
            return np.argsort(lengths)[0]

        idx = fewest_candidates(rows)
        row_candidates = rows[idx]

        for row in row_candidates:
            rows_with_assumption = copy.deepcopy(rows)
            rows_with_assumption[idx] = np.tile(row, [1, 1])
            try:
                x, y = sgm.iter_enforce_cell_constraints(rows_with_assumption, cols, column_size, row_size)
                yield make_node([x, y], parent)
            except InfeasibleStateException:
                pass

        idx = fewest_candidates(cols)
        col_candidates = cols[idx]

        for col in col_candidates:
            cols_with_assumption = copy.deepcopy(cols)
            cols_with_assumption[idx] = np.tile(col, [1, 1])
            try:
                x, y = sgm.iter_enforce_cell_constraints(rows, cols_with_assumption, column_size, row_size)
                yield make_node([x, y], parent)
            except InfeasibleStateException:
                pass

    def cost_fun(self, parent, child):
        return 0

    def goal_fun(self, node):
        rows, cols = node.state

        retval = np.all([len(each) == 1 for each in np.append(rows, cols)])
        return retval

    def generate_initial_node(self):
        cols = np.array([list(sgm.generate_segments_by_constraints(l, self.n_rows)) for l in self.col_lengths])
        rows = np.array([list(sgm.generate_segments_by_constraints(l, self.n_cols)) for l in self.row_lengths])

        state = rows, cols
        node = self._make_initial_node(state)
        node.info = 0
        return node
