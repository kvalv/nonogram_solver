# begin
import argparse
import drawing
from parse import parse_file
from csp2 import Nonogram_Solver
from heuristic import heuristic_fun


parser = argparse.ArgumentParser()
parser.add_argument('filename', help='the filename of the puzzle to be solved')
args = parser.parse_args()


solver = Nonogram_Solver(heuristic_fun, *parse_file(args.filename))
initial_node = solver.generate_initial_node()

solver.solve(initial_node.state)  # noqa
solver.print_summary()

state = solver.solution.state
col_height = state[0][0].shape[1]
row_height = state[1][0].shape[1]

window = drawing.generate_window(col_height, row_height)
drawing.draw_state(state, window)
window.getMouse()
window.close()
# state = solver.solution.parent.state
# window = drawing.generate_window(col_height, row_height)
# drawing.draw_state(state, window)
