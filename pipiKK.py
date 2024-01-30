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
        _, label_counts = np.unique(labels, return_counts=True)
        if len(label_counts) == len(labels):
            exchange_name = "$-$"
        else:
            exchange_name = r" \times ".join([f"S_{count}" for count in label_counts])
            exchange_name = "$" + exchange_name + "$"

        example_state = format_example_state(labels, momenta)

        internal_symmetry = mhi.make_exchange_group(labels)

        result = mhi.mhi(
            momenta=momenta,
            spin_irreps=['A1m', 'A1m', 'A1m', 'A1m'],
            internal_symmetry=internal_symmetry)

        orbit_dim = len(result.orbit)

        columns = [
            result.little_name(latex=True),
            result.stab_name(latex=True),
            exchange_name,
            example_state,
            str(orbit_dim),
            result.format(latex=True)
        ]
        row = " & ".join(columns) + r"\\"
        rows.append(row)

    print("The table of irrep decompositions is given by:")
    for row in rows:
        print(row)


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
                replace('b1', r'\pi^+').\
                replace('b2', r'K^+').\
                replace('b3', r'\pi^-').\
                replace('b4', r'K^-')
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


if __name__ == '__main__':
    main()
