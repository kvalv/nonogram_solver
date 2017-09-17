# begin
import segments as sgm




def sub_procedure(state):
    rows, cols = state

    for each in rows:
        if len(each) == 1:
            continue
        left, right = each[0], each[-1]
        # sub 1
        common_blacks = sgm.find_common_cells([left, right], cell_value=1)
        common_whites = sgm.find_common_cells([left, right], cell_value=0)

        # sub 2
        """
        find leftmost with fitting `common_blacks`
        find rightmost with fitting `common_whites`

        when found those two lines, fit all whites..

        """

        indices = np.where([np.all(e[common_blacks] == 1) for e in each])[0]
        import pdb; pdb.set_trace()
        left = each[indices[0]]
        right = each[indices[-1]]

        common_whites_prime = sgm.find_common_cells([left, right], cell_value=0)

        import pdb; pdb.set_trace()

sub_procedure(state)

# end
