"""
Example script illustrating cubic-group irrep decompositions with isospin with
the three-pion system. Compares to reference results in Table 5 of
https://arxiv.org/pdf/2003.10974.pdf (Hansen, Romero-Lopez, and Sharpe).
"""
import numpy as np
import itertools
from collections import namedtuple
from mhi import mhi

KnownResult = namedtuple("KnownResult", ['trivial', 'standard', 'sign'])

def main():

    internal_symmetries = get_projectors()
    for momenta in get_momenta_variations():
        print("#"*40)
        print("Starting momenta=", momenta)
        known_result = get_known_results(momenta)._asdict()

        for irrep in ['trivial', 'sign', 'standard:1', 'standard:2']:
            print(f"--> Checking {irrep} ")

            # Compute cubic-group irrep decomposition
            result, _ = mhi.mhi(
                momenta=momenta,
                spin_irreps=['A1m','A1m','A1m'],
                internal_symmetry=internal_symmetries[irrep]
            )

            # Isolate list of irreps
            result = sorted([irrep for irrep, _ in result])

            # Get reference result
            if irrep in ('standard:1', 'standard:2'):
                ref = sorted(known_result['standard'])
            else:
                ref = sorted(known_result[irrep])

            # Compare to reference
            if result != ref:
                print(result)
                print(ref)
                raise ValueError(f"Mismatch for {irrep}")

def get_projectors():
    """
    Gets projectors in the group algebra of permutation group S3 for each irrep.

    Parameters
    ----------
    None

    Returns
    -------
    internal_symmetries : dict
        The projectors, where the keys are the irrep name and the values are
        lists of weighted permutations.
    """
    # Permutations in S3
    perms = [np.array(perm) for perm in itertools.permutations([0,1,2])]

    # Projections in the group algebra
    return {
        'trivial': [mhi.WeightedPermutation(1, perm) for perm in perms],
        'sign': [mhi.WeightedPermutation(mhi.parity(perm), perm) for perm in perms],
        'standard:1': [
            mhi.WeightedPermutation(+1, [0,1,2]),
            mhi.WeightedPermutation(+1, [1,0,2]),
            mhi.WeightedPermutation(-1, [2,0,1]),
            mhi.WeightedPermutation(-1, [0,2,1]),
        ],
        'standard:2': [
            mhi.WeightedPermutation(+1, [0,1,2]),
            mhi.WeightedPermutation(+1, [2,1,0]),
            mhi.WeightedPermutation(-1, [1,0,2]),
            mhi.WeightedPermutation(-1, [1,2,0]),
        ],
    }

def get_momenta_variations():
    """
    Gets a list of momenta variations corresponding to those used in Table 5 of
    https://arxiv.org/pdf/2003.10974.pdf (Hansen, Romero-Lopez, and Sharpe).

    Parameters
    ----------
    None

    Returns
    -------
    momenta_variations : list of ndarray
    """
    return [
        np.array([[0,0,0],[0,0,0],[0,0,0]]),
        np.array([[0,0,1],[0,0,-1],[0,0,0]]),
        np.array([[0,1,1],[0,-1,-1],[0,0,0]]),
        np.array([[0,1,1],[0,-1,0],[0,0,-1]]),
        np.array([[1,1,1],[-1,-1,-1],[0,0,0]]),
        np.array([[2,0,0],[-1,0,0],[-1,0,0]]),
        np.array([[1,1,1],[-1,-1,0],[0,0,-1]]),
        np.array([[2,1,0],[-1,-1,-1],[-1,0,1]]),
    ]


def get_known_results(momenta):
    """
    Gets known results for comparison from Table 5 of
    https://arxiv.org/pdf/2003.10974.pdf (Hansen, Romero-Lopez, and Sharpe).
    To interpret this table, note the correspondence between isospin and irrep:
    I = 3 <--> trivial
    I = 2 <--> standard
    I = 1 <--> I = 2 and 3 <--> trivial + standard
    I = 0 <--> sign.

    Parameters
    ----------
    momenta : (n, 4) ndarray
        The momenta of the pions
    Returns
    -------
    ref : namedtuple with names ('trivial', 'standard', 'sign')
        lists of cubic-group irreps appearing for each S3 irrep
    """
    signature = tuple(np.dot(k, k) for k in momenta)
    # Rij = R_i^{(j)} from Table 5 of https://arxiv.org/pdf/2003.10974.pdf
    R34 = ['A1m','Em','T2m','T1p','T2p']
    R24 = ['A1m','A2m','Em','Em','T1m','T2m','T1p','T1p','T2p','T2p']
    R04 = ['A2m','Em','T1m','T1p','T2p']
    R37 = ['A1m','Em','T1m','T2m','T2m','A2p','Ep','T1p','T1p','T2p']
    R316 = ['A1m','A2m','Em','Em','T1m','T1m','T1m','T2m','T2m','T2m',
            'A1p','A2p','Ep','Ep','T1p','T1p','T1p','T2p','T2p','T2p']
    results = {
        # key : (trivial, standard, sign)
        (0, 0, 0) : KnownResult(['A1m'], [], []),
        (1, 1, 0) : KnownResult(['A1m','Em'], ['A1m','Em','T1p'], ['T1p']),
        (2, 2, 0) : KnownResult(['A1m','Em','T2m'],['A1m','Em','T2m','T1p','T2p'],['T1p','T2p']),
        (2, 1, 1) : KnownResult(R34, R24, R04),
        (3, 3, 0) : KnownResult(['A1m','T2m'],['A1m','T2m','A2p','T1p'],['A2p','T1p']),
        (4, 1, 1) : KnownResult(['A1m','Em','T1p'],['A1m','Em','T1p'],[]),
        (3, 2, 1) : KnownResult(R37, R37 + R37, R37),
        (5, 3, 2) : KnownResult(R316, R316 + R316, R316),
    }
    return results[signature]


if __name__ == '__main__':
    main()