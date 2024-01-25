"""
Example script illustrating the interplay of the exchange group and extended
orbit with the momentum configuration.
"""
import numpy as np
from mhi import mhi

def main():
    """
    Runs the decomposition for four combinations of particle types and momenta.
    """
    momenta1 = np.array([[0,0,1],[0,0,-1],[0,2,0],[0,-2,0]])
    momenta2 = np.array([[0,0,1],[0,2,0],[0,0,-1],[0,-2,0]])
    choices = [
        # "piplus, piminus, Kplus, Kminus"
        (momenta1, ['b1', 'b3', 'b2', 'b4']),
        # "piplus and Kplus"
        (momenta1, ['b1', 'b1', 'b2', 'b2']),
        (momenta2, ['b1', 'b1', 'b2', 'b2']),
        # "four identical pions"
        (momenta1, ['b1', 'b1', 'b1', 'b1']),
    ]
    rows = []
    for momenta, labels in choices:
        print("#"*40)
        print("Labels", labels)
        print("Momenta", momenta)
        little_name, stab_name = identify_little_group_and_stabilizer(momenta)
        little_name = format_group_name(little_name)
        stab_name = format_group_name(stab_name)

        _, label_counts = np.unique(labels, return_counts=True)
        if len(label_counts) == len(labels):
            exchange_name = "$-$"
        else:
            exchange_name = r" \times ".join([f"S_{count}" for count in label_counts])
            exchange_name = "$" + exchange_name + "$"

        example_state = format_example_state(labels, momenta)

        internal_symmetry = mhi.make_exchange_group(labels)

        result, Dmm = mhi.mhi(
            momenta=momenta,
            spin_irreps=['A1m', 'A1m', 'A1m', 'A1m'],
            internal_symmetry=internal_symmetry,
            return_Dmm=True)

        irrep_decomposition = sort_and_format(result.keys())

        orbit_dim = Dmm.shape[-1]

        columns = [
            little_name,
            stab_name,
            exchange_name,
            example_state,
            str(orbit_dim),
            irrep_decomposition
        ]
        row = " & ".join(columns) + r"\\"
        rows.append(row)

    print("The table of irrep decompositions is given by:")
    for row in rows:
        print(row)


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


def identify_little_group_and_stabilizer(momenta):
    """
    Identifies the little group and stabilizer associated with a set of momenta.

    Parameters
    ----------
    momenta : (n, 3) array_like
        The momenta

    Returns
    -------
    (little_name, stabilizer_name) : (str, str)
        The names of the groups
    """
    little = mhi.make_stabilizer(
        momenta=np.sum(momenta, axis=0),
        group=mhi.make_oh())
    little_name = mhi.identify_stabilizer(little)

    stabilizer = mhi.make_stabilizer(
        momenta=momenta,
        group=mhi.make_oh())
    stabilizer_name = mhi.identify_stabilizer(stabilizer)
    return little_name, stabilizer_name


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

def format_example_state(labels, momenta):
    """
    Formats an example state in Latex, e.g.,
    $|\pi^+(\bm{p}_1), \pi^+(-\bm{p}_1), K^+(\bm{p}_2), K^+(-\bm{p}_2)\rangle$

    Parameters
    ----------
    labels : (n, ) array_like
        The labels for the particles, assumed to be 'b1', 'b2', 'b3', 'b4'
    momenta : (n, 3) array_like
        The particle momenta

    Returns
    -------
    state_name : str
        The state in Latex
    """
    particles = []
    for label, p in zip(labels, momenta):
        label = label.\
                replace('b1', '\pi^+').\
                replace('b2', 'K^+').\
                replace('b3', '\pi^-').\
                replace('b4', 'K^-')
        p = tuple(p)
        if p == (0,0,1):
            p = r'\bm{p}_1'
        elif p == (0,2,0):
            p = r'\bm{p}_2'
        elif p == (0,0,-1):
            p = r'-\bm{p}_1'
        elif p == (0,-2,0):
            p = r'-\bm{p}_2'
        else:
            raise ValueError("Unexpected momentum", p)
        particles.append(f"{label}({p})")
    return '$|' + ", ".join(particles) + r'\rangle$'

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
