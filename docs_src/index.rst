Overview
========
This documentation describes the `mhi` ("Multi-Hadron Interpolators") package
associated with the paper

    W. Detmold, W. I. Jay, G. Kanwar, P. E. Shanahan, and M. L. Wagman,
    "Multi-particle interpolating operators in quantum field theories with cubic symmetries."

Installation
============
After cloning the repo, move to the top-level directory containing setup.py. Then
use pip to install the module:

``$ pip install -e ./``

The flag "-e" is an "editable install", which means that local changes to the
code will appear when the code is run.

Usage
=====
The most flexible usage comes from the python module. For example::

    # Momenta for three particles in a boosted frame.
    momenta = np.array([[1,0,0],[0,1,0], [0,0,1],])

    # Compute block diagonalization for three distinguishable "pseudoscalar" operators
    # Block diagonalization matrices are stored in the dict result.decomp
    result = mhi.mhi(
        momenta,
        spin_irreps=['A1m','A1m','A1m'],
        internal_symmetry=None)
    print("Case 1:", result.format(latex=True))

    # Repeat the diagonalization including symmetrization associated
    # with irreps of some internal symmetry, labelled by Young diagrams.
    tableaux = [
        [[1,2,3]],      # trivial irrep of S3
        [[1,2],[3]],    # standard irrep of S3
        [[1,3],[2]],
        [[1],[2],[3]],  # sign irrep of S3
    ]
    for n, tableau in enumerate(tableaux):
        internal_symmetry = mhi.make_young_projector(tableau, n=3)
        result = mhi.mhi(momenta, spin_irreps=['A1m','A1m','A1m'], internal_symmetry=internal_symmetry)
        print(f"Case {n+2}:", result.format(latex=True))

which yields::

    Case 1: $A_1 \oplus A_2 \oplus 2E$
    Case 2: $A_2$
    Case 3: $E$
    Case 4: $E$
    Case 5: $A_1$

These results arise in the case of three pions with isospin, which is discussed explicitly in the accompanying paper.

A command-line interface to the module is also provided. Example usage is describe below in Tests.

Tests
=====

To test the command-line interface, run the command

``$ python -m mhi example_input.yaml <output_name.h5> --verbose``

The flag "-m" is necessary and directs python to run the module's main function. After running the main function, it is possible to test the ability to read the hdf5 file using

``$ python read.py <output_name.h5>``

To test the module against previously tabulated data, run the command

``$ python test.py <path/to/reference/data>``

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   code.rst
