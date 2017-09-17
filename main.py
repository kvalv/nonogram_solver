import numpy as np
from parse import parse_file
import segments as sgm
from csp2 import Nonogram_Solver
import ipdb, traceback, sys, inspect
import debug_wrappers 
import drawing

n_cols, n_rows, cols, rows = parse_file('dataset.dat')


def heuristic_fun(node):
    x, y = node.state
    z = np.append(x, y)
    size_penalty = sum([np.log(len(each)**2 + 1) for each in z])

    symmetry_bonus = 0
    single_indices = np.where([len(e) == 1 for e in z])[0]
    for each in z[single_indices]:
        each = each[0]
        rev = each[::-1]
        if np.all(np.equal(rev, each)):
            symmetry_bonus += 1

    L1 = [str(e) for e in z[single_indices]]
    L2 = np.unique(L1)
    equality_bonus = len(L1) - len(L2)
    # if equality_bonus or symmetry_bonus:
    #     print('bonuses:', equality_bonus, symmetry_bonus)
    return size_penalty - symmetry_bonus - equality_bonus
    # return 0


    return node.info

if __name__ == '__main__':
    try:
        # most difficult
        # cat, 8x9 0.7sec (0.35 with the new generate_children)
        # chick, 15x15 not solved with new
        # sailboat, 20x20, 2.6 (symmetric)
        # elephant: 15x15, 0.247s
        # rabbit: 20x15, 8.95s
        # clover: 15x15, 0.110
        # fox: 25x25, got errors
        # reindeer: 20x20,  takes lots of time
        # snail 20x15, takes lots of time
        solver = Nonogram_Solver(heuristic_fun, *parse_file('data/rabbit.dat'))
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
