import numpy as np
import os
import sys
from mhi import mhi
from mhi import basis_functions

def main():
    """
    Command-line script to test functionality of MHI code against tabulated
    reference data.
    """
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <reference/output/directory>")
        sys.exit(0)

    base = sys.argv[1]
    if not os.path.exists(base):
        print(base, "does not exist.")
        sys.exit(0)

    test_conjugation()
    test_groups_and_irreps(base)
    test_subgroups_and_irreps(base)

    setup1, setup2, setup3, setup4 = get_test_setups()
    tests = [
        (os.path.join(base, "spinless"), [], None, setup1),
        (os.path.join(base, "Npi"), ['n','pi'], ['G1p','A1m'], setup2),
        (os.path.join(base, "np"), ['n','p'], ['G1p','G1p'], setup2),
        (os.path.join(base, "nn"), ['n','n'], ['G1p','G1p'], setup3),
        (os.path.join(base, "nnn"), ['n','n','n'], ['G1p','G1p','G1p'], setup4),
    ]
    for base, particle_names, spin_irreps, setup in tests:
        print("Testing against files in", base)
        for fname_rep, fname_basis, momenta in setup:
            print("#"*40)
            if fname_rep:
                fname_rep = os.path.join(base,fname_rep)
            test_mhi(momenta,
                     particle_names,
                     spin_irreps,
                     fname_rep=fname_rep,
                     fname_basis=os.path.join(base, fname_basis))


# ----- end main ----- #


def test_conjugation():
    """
    Tests group conjugation for mapping conjugate subgroups into one another.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    oh = mhi.make_oh()
    momenta = [
        ([0,0,1], [0,1,0]),
        ([0,1,1], [1,1,0]),
        ([0,1,2], [1,2,0]),
        ([1,1,2], [2,1,1]),
        ([1,2,3], [3,2,1]),
    ]
    for momentum1, momentum2 in momenta:
        h1 = mhi.make_stabilizer(momentum1, oh)
        h2 = mhi.make_stabilizer(momentum2, oh)
        isomorphism = mhi.find_subgroup_isomorphism(oh, h1, h2)
        h2test = mhi.apply_isomorphism(h1, isomorphism)
        assert np.allclose(h2test, h2),\
            "Error with conjugation of little groups for momenta {momentum1} {momentum2}."
        print(f"Success: conjugation of little groups for momenta {momentum1} {momentum2}.")


def test_groups_and_irreps(base):
    """
    Tests construction of group presentations and irrep matrices, comparing
    results to tabulated reference data.

    Parameters
    ----------
    base : str
        Path to the output directory containing the reference data.

    Returns
    -------
    None
    """
    # Oh: Verify agreement of group presentations
    oh = mhi.make_oh()
    oh_ref = read_mathematica(
        os.path.join(base, "Oh/Group.dat"),
        oh.shape)
    assert np.allclose(oh, oh_ref), "Mismatch: Oh"
    print("Success for group: Oh")

    # OhD: Verify agreement of group presentations
    ohd = mhi.make_ohd()
    ohd_ref = read_mathematica(
        os.path.join(base,"OhD/Group.dat"),
        ohd.shape)
    assert np.allclose(ohd, ohd_ref), "Mismatch: OhD"
    print("Success for group: OhD")

    # Oh: Verify agreement of group irrep matrices
    Dmumu = mhi.make_irrep_from_group(oh)
    for irrep_name, table in Dmumu.items():
        prefix = irrep_name.replace("m", "_minus").replace("p", "_plus")
        ref = read_mathematica(
            os.path.join(base, f"Oh/{prefix}.dat"),
            table.shape)
        assert np.allclose(ref, table)
        print("Success for Oh irrep:", irrep_name)

    # OhD: Verify agreement of double-cover group irrep matrices
    Dmumu = mhi.make_irrep_from_groupD(oh)
    for irrep_name, table in Dmumu.items():
        prefix = irrep_name.replace("m", "_minus").replace("p", "_plus")
        ref = read_mathematica(
            os.path.join(base, f"OhD/{prefix}.dat"),
            table.shape)
        assert np.allclose(ref, table)
        print("Success for OhD irrep:", irrep_name)


def test_subgroups_and_irreps(base):
    """
    Tests construction of subgroup presentations and irrep matrices, comparing
    results to tabulated reference data.

    Parameters
    ----------
    base : str
        Path to the output directory containing the reference data.

    Returns
    -------
    None
    """
    oh = mhi.make_oh()

    # Verify agreement of subgroup presentations and irreps
    subgroups = {
        'C4v': [0,0,1],
        'C3v': [1,1,1],
        'C2v': [0,1,1],
        'C2R': [1,2,0],
        'C2P': [1,1,2],
        'C1':  [1,2,3],
    }
    for name, momenta in subgroups.items():

        # Subroups of Oh
        little = mhi.make_stabilizer(momenta=momenta, group=oh)
        little_name = mhi.identify_stabilizer(little)
        assert name == little_name, f"Misidentified subgroup? {name} vs {little_name}"
        little_ref = read_mathematica(
            os.path.join(base, f"{little_name}/Group.dat"),
            little.shape)
        assert np.allclose(little, little_ref), f"Mismatch: {little_name}"
        print(f"Success for subgroup: {little_name} of Oh")

        # Irreps of subgroups of Oh
        Dmumu = mhi.make_irrep_from_group(little)
        for irrep_name, table in Dmumu.items():
            ref = read_mathematica(
                os.path.join(base, f"{name}/{irrep_name}.dat"),
                table.shape)
            assert np.allclose(ref, table), f"Problem with {irrep_name}"
            print(f"Success for {little_name} irrep: {irrep_name}")

        # Subgroups of OhD
        little_double = mhi.make_spinorial_little_group(little)
        little_double_ref = read_mathematica(
            os.path.join(base, f"{little_name}D/Group.dat"),
            little_double.shape)
        assert np.allclose(little_double, little_double_ref), f"Mismatch: {little_name}"
        print(f"Success for subgroup: {little_name}D of OhD")

        # Irreps of subgroups of OhD
        Dmumu_double = mhi.make_irrep_from_groupD(little)
        for irrep_name, table in Dmumu_double.items():
            ref = read_mathematica(
                os.path.join(base, f"{name}D/{irrep_name}.dat"),
                table.shape)
            assert np.allclose(ref, table), f"Problem with {irrep_name}"
            print(f"Success for {little_name} irrep: {irrep_name}")

def test_mhi(momenta, particle_names, spin_irreps, fname_rep, fname_basis):
    """
    Verifies construction of momentum-(spin-)orbit representation matrices as
    well as the change-of-basis coefficients against tabulated reference data.

    Parameters
    ----------
    momenta : (nmomenta, 3) or (3, ) array_like
        The ordered momenta, with shape.
    particle_names : list or None
        The names of the particles, e.g., ('n', 'pi')
    fname_rep : str
        Path to the input file containing the data for the momentum orbit
        representation.
    fname_basis : str
        Path to the input file containing the data for the change-of-basis
        coefficients
    """
    internal_symmetry = mhi.make_exchange_group(particle_names)
    result = mhi.mhi(
        momenta,
        spin_irreps,
        internal_symmetry=internal_symmetry,
        verbose=True)

    # Check momentum-orbit representation matrices
    if fname_rep is not None:
        ref = read_mathematica(fname_rep, result.Dmm.shape)
        assert np.allclose(result.Dmm, ref), f"Trouble with {fname_rep}"
        print("Success:", os.path.split(fname_rep)[-1])

    # Check change-of-basis coefficients
    table = make_table(result.decomp)
    ref = read_mathematica(fname_basis, table.shape)
    assert np.allclose(table, ref), f"Error: trouble with {fname_basis}"
    print("Success:", os.path.split(fname_basis)[-1])


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


def make_table(proj):
    """
    Converts a "table" of block-diagonalization / change-of-basis coefficients,
    specified as a dict to an array.

    Parameters
    ----------
    proj : dict
        The block-diagonalization / change-of-basis coefficients.
        The keys are of the form (irrep, degeneracy count), while the values
        are the coefficients for that irrep.

    Returns
    -------
    table : ndarray
        The stacked table of coefficients, with the irreps appearing in a
        specified order.

    See Also
    --------
    irrep_priority : specifies the ordering of the irreps
    """
    irreps = [key for key, _ in proj.keys()]
    idxs = range(len(irreps))
    _, new_idxs  = zip(
        *sorted(zip(irreps, idxs),
        key=lambda pair: irrep_priority(pair[0])))
    rows = list(proj.values())
    return np.vstack([rows[idx] for idx in new_idxs])


def read_mathematica(fname, shape=None):
    """
    Reads a mathematica file

    Parameters
    ----------
    fname : str
        The full name of the file to read.

    Returns
    -------
    arr : ndarray
        The array of data.
    """
    with open(fname, 'rb') as ifile:
        ref = np.fromfile(ifile, dtype=complex)
        if shape is not None:
            ref = ref.reshape(shape)
        return ref

def get_test_setups():
    """
    Gets hard-coded setup information for running tests
    """
    # Momentum configurations for spinless case
    test_setup1 = [
        ("Dmm_000.dat", "basis_000.dat", np.array([[0,0,0],[0,0,0]])),
        ("Dmm_100.dat", "basis_100.dat", np.array([[0,0,1],[0,0,-1]])),
        ("Dmm_110.dat", "basis_110.dat", np.array([[0,1,1],[0,-1,-1]])),
        ("Dmm_111.dat", "basis_111.dat", np.array([[1,1,1],[-1,-1,-1]])),
        ("Dmm_112.dat", "basis_112.dat", np.array([[1,1,2],[-1,-1,-2]])),
        ("Dmm_120.dat", "basis_120.dat", np.array([[1,2,0],[-1,-2,0]])),
        ("Dmm_123.dat", "basis_123.dat", np.array([[1,2,3],[-1,-2,-3]])),
        ("Dmm_001_000.dat", "basis_001_000.dat", np.array([[0,0,1],[0,0,0]])),
        ("Dmm_110_000.dat", "basis_110_000.dat", np.array([[1,1,0],[0,0,0]])),
        ("Dmm_111_000.dat", "basis_111_000.dat", np.array([[1,1,1],[0,0,0]])),
        ("Dmm_120_000.dat", "basis_120_000.dat", np.array([[1,2,0],[0,0,0]])),
        ("Dmm_123_000.dat", "basis_123_000.dat", np.array([[1,2,3],[0,0,0]])),
        ("Dmm_110_m001.dat", "basis_110_m001.dat", np.array([[1,1,0],[0,0,1]])),
        ("Dmm_101_100.dat", "basis_101_100.dat", np.array([[1,0,1],[-1,0,0]])),
        ("Dmm_111_m1m10.dat", "basis_111_m1m10.dat", np.array([[1,1,1],[-1,-1,0]])),
        ("Dmm_211_m2m10.dat", "basis_211_m2m10.dat", np.array([[2,1,1],[-2,-1,0]])),
        ("Dmm_10m1_012.dat", "basis_10m1_012.dat", np.array([[1,0,-1],[0,1,2]])),
        ("Dmm_100_m010.dat", "basis_100_m010.dat", np.array([[0,0,1],[0,1,0]])),
        ("Dmm_112_00m2.dat", "basis_112_00m2.dat", np.array([[2,1,1],[-2,0,0]])),
        ("Dmm_10m2_012.dat", "basis_10m2_012.dat", np.array([[-2,0,1],[2,1,0]])),
        ("Dmm_101_02m1.dat", "basis_101_02m1.dat", np.array([[1,0,1],[0,2,-1]])),
    ]

    # Momentum configurations for Npi and np
    test_setup2 = [
        ("Dmm_000.dat", "basis_000.dat", np.array([[0,0,0],[0,0,0]])),
        ("Dmm_100.dat", "basis_100.dat", np.array([[0,0,1],[0,0,-1]])),
        ("Dmm_110.dat", "basis_110.dat", np.array([[0,1,1],[0,-1,-1]])),
        ("Dmm_111.dat", "basis_111.dat", np.array([[1,1,1],[-1,-1,-1]])),
        ("Dmm_112.dat", "basis_112.dat", np.array([[2,1,1],[-2,-1,-1]])),
        ("Dmm_120.dat", "basis_120.dat", np.array([[1,2,0],[-1,-2,0]])),
        ("Dmm_123.dat", "basis_123.dat", np.array([[1,2,3],[-1,-2,-3]])),
        ("Dmm_001_000.dat", "basis_001_000.dat", np.array([[0,0,1],[0,0,0]])),
        ("Dmm_110_000.dat", "basis_110_000.dat", np.array([[0,1,1],[0,0,0]])),
        ("Dmm_111_000.dat", "basis_111_000.dat", np.array([[1,1,1],[0,0,0]])),
        ("Dmm_120_000.dat", "basis_120_000.dat", np.array([[1,2,0],[0,0,0]])),
        ("Dmm_123_000.dat", "basis_123_000.dat", np.array([[1,2,3],[0,0,0]])),
        ("Dmm_110_m001.dat","basis_110_m001.dat", np.array([[1,1,0],[0,0,1]])),
        ("Dmm_101_100.dat", "basis_101_100.dat", np.array([[1,0,1],[-1,0,0]])),
        ("Dmm_111_m1m10.dat", "basis_111_m1m10.dat", np.array([[1,1,1],[-1,-1,0]])),
        ("Dmm_211_m2m10.dat", "basis_211_m2m10.dat", np.array([[2,1,1],[-2,-1,0]])),
        ("Dmm_10m1_012.dat", "basis_10m1_012.dat", np.array([[1,0,-1],[0,1,2]])),
        ("Dmm_100_m010.dat", "basis_100_m010.dat", np.array([[0,0,1],[0,1,0]])),
        ("Dmm_112_00m2.dat", "basis_112_00m2.dat", np.array([[2,1,1],[-2,0,0]])),
        ("Dmm_10m2_012.dat", "basis_10m2_012.dat", np.array([[-2,0,1],[2,1,0]])),
        ("Dmm_101_02m1.dat", "basis_101_02m1.dat", np.array([[1,0,1],[0,2,-1]])),
    ]

    # Momentum configurations for nn
    test_setup3 = [
        ("Dmm_proj_000.dat", "basis_000.dat", np.array([[0,0,0],[0,0,0]])),
        ("Dmm_proj_100.dat", "basis_100.dat", np.array([[0,0,1],[0,0,-1]])),
        ("Dmm_proj_110.dat", "basis_110.dat", np.array([[0,1,1],[0,-1,-1]])),
        ("Dmm_proj_111.dat", "basis_111.dat", np.array([[1,1,1],[-1,-1,-1]])),
        ("Dmm_proj_112.dat", "basis_112.dat", np.array([[2,1,1],[-2,-1,-1]])),
        ("Dmm_proj_120.dat", "basis_120.dat", np.array([[2,1,0],[-2,-1,0]])),
        ("Dmm_proj_123.dat", "basis_123.dat", np.array([[3,2,1],[-3,-2,-1]])),
        ("Dmm_proj_001_000.dat", "basis_001_000.dat", np.array([[0,0,1],[0,0,0]])),
        ("Dmm_proj_001_001.dat", "basis_001_001.dat", np.array([[0,0,1],[0,0,1]])),
        ("Dmm_proj_100_010.dat", "basis_100_010.dat", np.array([[0,0,1],[0,1,0]])),
        ("Dmm_proj_101_02m1.dat", "basis_101_02m1.dat", np.array([[1,0,1],[0,2,-1]])),
        ("Dmm_proj_101_100.dat", "basis_101_100.dat", np.array([[1,0,1],[-1,0,0]])),
        ("Dmm_proj_101_101.dat", "basis_101_101.dat", np.array([[1,0,1],[-1,0,1]])),
        ("Dmm_proj_10m1_012.dat", "basis_10m1_012.dat", np.array([[1,0,-1],[0,1,2]])),
        ("Dmm_proj_10m2_012.dat", "basis_10m2_012.dat", np.array([[-2,0,1],[2,1,0]])),
        ("Dmm_proj_110_000.dat", "basis_110_000.dat", np.array([[0,1,1],[0,0,0]])),
        ("Dmm_proj_110_110.dat", "basis_110_110.dat", np.array([[1,1,0],[1,1,0]])),
        ("Dmm_proj_110_m001.dat", "basis_110_m001.dat", np.array([[1,1,0],[0,0,1]])),
        ("Dmm_proj_111_000.dat", "basis_111_000.dat", np.array([[1,1,1],[0,0,0]])),
        ("Dmm_proj_111_111.dat", "basis_111_111.dat", np.array([[1,1,1],[1,1,1]])),
        ("Dmm_proj_111_m1m10.dat", "basis_111_m1m10.dat", np.array([[1,1,1],[-1,-1,0]])),
        ("Dmm_proj_111_m1m11.dat", "basis_111_m1m11.dat", np.array([[1,1,1],[-1,-1,1]])),
        ("Dmm_proj_112_00m2.dat", "basis_112_00m2.dat", np.array([[2,1,1],[-2,0,0]])),
        ("Dmm_proj_112_11m2.dat", "basis_112_11m2.dat", np.array([[2,1,1],[-2,1,1]])),
        ("Dmm_proj_120_000.dat", "basis_120_000.dat", np.array([[1,2,0],[0,0,0]])),
        ("Dmm_proj_120_120.dat", "basis_120_120.dat", np.array([[1,2,0],[1,2,0]])),
        ("Dmm_proj_121_0m1m1.dat", "basis_121_0m1m1.dat", np.array([[1,2,1],[-1,-1,0]])),
        ("Dmm_proj_123_000.dat", "basis_123_000.dat", np.array([[1,2,3],[0,0,0]])),
        ("Dmm_proj_123_123.dat", "basis_123_123.dat", np.array([[1,2,3],[1,2,3]])),
        ("Dmm_proj_211_m2m10.dat", "basis_211_m2m10.dat", np.array([[2,1,1],[-2,-1,0]])),
        ("Dmm_proj_211_m2m11.dat", "basis_211_m2m11.dat", np.array([[2,1,1],[-2,-1,1]])),
        ("Dmm_proj_220_m1m10.dat", "basis_220_m1m10.dat", np.array([[0,2,2],[0,-1,-1]])),
    ]

    # Momentum configurations for nnn
    test_setup4 = [
        (None, "basis_001.dat", np.array([[0,0,1],[0,0,-1],[0,0,0]])),
        (None, "basis_001_002.dat", np.array([[0,0,1],[0,0,2],[0,0,-3]])),
        (None, "basis_011.dat", np.array([[0,1,1],[0,-1,-1],[0,0,0]])),
        (None, "basis_011_022.dat", np.array([[0,-1,-1],[0,-2,-2],[0,3,3]])),
        (None, "basis_111.dat", np.array([[1,1,1],[-1,-1,-1],[0,0,0]])),
        (None, "basis_111_222.dat", np.array([[1,1,1],[2,2,2],[-3,-3,-3]])),
        (None, "basis_210.dat", np.array([[2,1,0],[-2,-1,0],[0,0,0]])),
        (None, "basis_011_0m10.dat", np.array([[0,1,1],[0,-1,0],[0,0,-1]])),
        (None, "basis_210_m200.dat", np.array([[2,1,0],[-2,0,0],[0,-1,0]])),
        (None, "basis_211.dat", np.array([[2,1,1],[-2,-1,-1],[0,0,0]])),
        (None, "basis_111_m1m10.dat", np.array([[1,1,1],[-1,-1,0],[0,0,-1]])),
        (None, "basis_0m11_210.dat", np.array([[0,-1,1],[2,1,0],[-2,0,-1]])),
        # OK, but very slow to run all the internal tests
        # (None, "basis_211_m2m10.dat", np.array([[2,1,1],[-2,-1,0],[0,0,-1]])),
    ]
    return test_setup1, test_setup2, test_setup3, test_setup4

if __name__ == "__main__":
    main()
