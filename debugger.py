import numpy as np
from parse import parse_file
import segments as sgm
from csp2 import Nonogram_Solver
import ipdb, traceback, sys, inspect
import debug_wrappers 
import drawing

if __name__ == '__main__':
    try:
        solver = Nonogram_Solver(heuristic_fun, *parse_file('data/reindeer.dat'))
        initial_node = solver.generate_initial_node()

        %prun solver.solve(initial_node.state)
        print(solver.summarize())
        x, y = solver.solution.state
        col_height = x[0].shape[1]
        row_height = y[0].shape[1]

        import pdb; pdb.set_trace()
        window = drawing.generate_window(col_height, row_height)

        drawing.draw_state((x,y), window)
        
        M = np.rot90(np.matrix([each[0] for each in x]), k=3)
        M = np.rot90(M,k=3)
        N = np.matrix([each[0] for each in y])
        N = np.rot90(N,k=3)

        print(M,'\n')
        print(N,'\n')
        is_solution = np.all(np.equal(M,N))
        print(f'found solution: {is_solution}')

        import pdb; pdb.set_trace()


    except:
        print('closing window')
        window.close()
        type, value, tb = sys.exc_info()
        traceback.print_tb(tb)
        print(type, value)
        frame = tb.tb_frame
        inner = [each for each in inspect.getinnerframes(tb)]
        outer = [each for each in inspect.getouterframes(frame)]
        combined = inner + outer

        def fnames(frames):
            return [each.function for each in frames]

        last_function_call = inspect.getinnerframes(tb)[-1].function
        db = lambda name: debug_wrappers.FUN(combined, name)
        frame = db(last_function_call)

        import ipdb; ipdb.set_trace()
# end
