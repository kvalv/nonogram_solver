# begin
import argparse
import drawing
from parse import parse_file
from csp import Nonogram_Solver
from heuristic import heuristic_fun


parser = argparse.ArgumentParser()
parser.add_argument('filename', help='the filename of the puzzle to be solved')
args = parser.parse_args()

filename = args.filename

row_height, col_height, _, _ = parse_file(filename)
window = drawing.generate_window(col_height, row_height)

solver = Nonogram_Solver(heuristic_fun, *parse_file(filename), window=window)
initial_node = solver.generate_initial_node()

solver.solve(initial_node.state)  # noqa
solver.print_summary()

# click to close..
window.getMouse()
window.close()
