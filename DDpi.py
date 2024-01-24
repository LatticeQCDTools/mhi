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
        # Identify group names
        little, stab = mhi.make_little_and_stabilizer(momenta, group=mhi.make_oh())
        little_name = mhi.identify_stabilizer(little)
        stab_name = mhi.identify_stabilizer(stab)
        little_name = format_group_name(little_name)
        stab_name = format_group_name(stab_name)

        # Format the state in Latex
        state = f"$\left|D{tuple(momenta[0])},D{tuple(momenta[1])},pi{tuple(momenta[2])}\\right\\rangle$"
        state  = state.replace("pi", "\pi")

        for irrep in ['trivial', 'sign']:
            # Compute the irrep decomposition
            result, orbit = mhi.mhi(
                momenta=momenta,
                spin_irreps=['A1m','A1m','A1m'],
                internal_symmetry=internal_symmetries[irrep])

            # Format the results as rows of a table
            if len(result):
                result = sort_and_format(result.keys())
            else:
                continue
            columns = [
                little_name,
                stab_name,
                irrep.capitalize(),
                state,
                str(len(orbit)),
                result
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


def irrep_priority(irrep_name):
    """
    Evaluate the priority of an irrep for sorting.

    Parameters
    ----------
    irrep_name : str
        The irrep name

    Returns
    -------
    priority : int
        The priority of the irrep compared to others, for use in sorting
    """
    ordered_irreps = [
        'A1p', 'A2p', 'Ep', 'T1p', 'T2p',
        'A1m', 'A2m', 'Em', 'T1m', 'T2m',
        'A', 'A1', 'A2', 'B', 'B1', 'B2', 'E',
        "F", "F1", "F2", "G", "G1", "G2", "G1p", "G2p",
        "Hp", "G1m", "G2m", "Hm"
    ]
    priority = dict(zip(ordered_irreps, range(len(ordered_irreps))))
    if irrep_name not in ordered_irreps:
        raise ValueError("Bad irrep_name", irrep_name)
    return priority.get(irrep_name)

def irrep_to_latex(output):
    """
    Converts irrep names from internal string representations to latex.

    Parameters
    ----------
    output : str
        Some text containing the internal string representations of the irrep names

    Returns
    -------
    output : str
        The text, but with appropriate replacements, e.g., 'A1p' -> 'A_1^+.
    """
    irrep_map = {
        'A1p': 'A_1^+',
        'A2p': 'A_2^+',
        'Ep': 'E^+',
        'T1p': 'T_1^+',
        'T2p': 'T_2^+',
        'A1m': 'A_1^-',
        'A2m':  'A_2^-',
        'Em':  'E^-',
        'T1m':  'T_1^-',
        'T2m': 'T_2^-',
        'A': 'A',
        'A1': 'A_1',
        'A2': 'A_2',
        'B':  'B',
        'B1':  'B_1',
        'B2':  'B_2',
        'E': 'E',
        "F":  "F",
        "F1":  "F_1",
        "F2":  "F_2",
        "G": "G",
        "G1": "G_1",
        "G2": "G_2",
        "G1p": "G_1^+",
        "G2p": "G_2^+",
        "Hp": "H^+",
        "G1m": "G_1^-",
        "G2m": "G_2^-",
        "Hm": "H^-",
    }
    for old, new in irrep_map.items():
        output = output.replace(old, new)
    return output


def format_group_name(name):
    """
    Formats the group name in Latex, e.g., 'C2R' -> '$C_2^R$'.

    Parameters
    ----------
    name : str
        The name of the group

    Returns
    -------
    name_latex : str
        The name of the group in Latex
    """
    name_map = {
        'C2R': 'C_2^R',
        'C2P': 'C_2^P',
        'Oh': 'O_h',
        'C4v': 'C_4^v',
        'C3v': 'C_3^v',
        'C2v': 'C_2^v',
        'C1': 'C_1'
    }
    return '$' + name_map[name] + '$'

def sort_and_format(irrep_list):
    """
    Sorts and formats a list of (possibly degenerate) irreps to Latex, e.g.,
    ['A1p', 'A2p', 'Ep', 'Ep'] --> '$A_1^+ \oplus A_2^+ \oplus 2E^+$'.

    Parameters
    ----------
    irrep_list : list of str
        The irreps in the decomposition

    Returns
    -------
    decomp : str
        The decomposition in Latex format
    """
    # Count the degenerate copies
    irrep_counts = {}
    for irrep, _ in irrep_list:
        if irrep not in irrep_counts:
            irrep_counts[irrep] = 1
        else:
            irrep_counts[irrep] += 1

    # Sort the names according to a conventional ordering
    names = list(irrep_counts.keys())
    degeneracies = list(irrep_counts.values())
    names, degeneracies = zip(
            *sorted(zip(names, degeneracies),
            key=lambda pair: irrep_priority(pair[0])))

    # Convert the results to a latex string
    output = []
    for k, name in zip(degeneracies, names):
        if k == 1:
            output.append(name)  # "1" looks silly as a coefficient
        else:
            output.append(str(k)+name)
    output = r' \oplus '.join(output)
    output = irrep_to_latex(output)
    output = '$' + output + '$'
    return output






if __name__ == '__main__':
    main()