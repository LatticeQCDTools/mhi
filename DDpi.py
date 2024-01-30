"""
Example script illustrating cubic-group irrep decompositions for DDpi.
"""
import numpy as np
import itertools
from mhi import mhi

def main():
    """
    Runs the cubic-group irrep decomposition for a variety of momentum
    configurations.
    """
    # Define projectors in the group algebra of S2 x S1
    perms = [np.array(list(perm) + [2]) for perm in itertools.permutations([0,1])]
    internal_symmetries = {
        'trivial': [mhi.WeightedPermutation(1, perm) for perm in perms],
        'sign': [mhi.WeightedPermutation(mhi.parity(perm), perm) for perm in perms],
    }

    rows = []
    for momenta in get_all_momenta():
        # Format the state in Latex
        state = f"$\left|D{tuple(momenta[0])},D{tuple(momenta[1])},pi{tuple(momenta[2])}\\right\\rangle$"
        state  = state.replace("pi", "\pi")

        for irrep in ['trivial', 'sign']:
            # Compute the irrep decomposition
            result = mhi.mhi(
                momenta=momenta,
                spin_irreps=['A1m','A1m','A1m'],
                internal_symmetry=internal_symmetries[irrep])

            # Format the results as rows of a table
            if len(result.decomp) == 0:
                continue
            columns = [
                result.little_name(latex=True),
                result.stab_name(latex=True),
                irrep.capitalize(),
                state,
                str(len(result.orbit)),
                result.format(latex=True)
            ]
            row = " & ".join(columns) + r"\\"
            if row not in rows:
                rows.append(row)

    for row in rows:
        print(row)

def get_all_momenta():
    """
    Get the list of momentum variations.

    Parameters
    ----------
    None

    Returns
    -------
    all_momenta : list
        list of ndarrays with the momenta
    """
    return [
        np.array([[0,0,0],[0,0,0],[0,0,0]]),
        np.array([[0,0,1],[0,0,-1],[0,0,0]]),
        np.array([[0,1,1],[0,-1,-1],[0,0,0]]),
        np.array([[1,1,1],[-1,-1,-1],[0,0,0]]),
        np.array([[2,1,0],[-2,-1,0],[0,0,0]]),
        np.array([[2,1,1],[-2,-1,-1],[0,0,0]]),
        np.array([[3,2,1],[-3,-2,-1],[0,0,0]]),
    ]


if __name__ == '__main__':
    main()