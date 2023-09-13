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
        print("usage: python {sys.argv[0]} <reference/output/directory>")
        sys.exit(0)

    base = sys.argv[1]
    if not os.path.exists(base):
        print(base, "does not exist.")
        sys.exit(0)

    test_groups_and_irreps(base)
    test_subgroups_and_irreps(base)

    # Momentum configurations for spinless case
    test_setup1 = [
        ("Dmm_000.dat", "basis_000.dat", np.array([[0,0,0]])),
        ("Dmm_100.dat", "basis_100.dat", np.array([[0,0,1],[0,0,-1]])),
        ("Dmm_110.dat", "basis_110.dat", np.array([[0,1,1],[0,-1,-1]])),
        ("Dmm_111.dat", "basis_111.dat", np.array([[1,1,1],[-1,-1,-1]])),
        ("Dmm_112.dat", "basis_112.dat", np.array([[1,1,2],[-1,-1,-2]])), #
        ("Dmm_120.dat", "basis_120.dat", np.array([[1,2,0],[-1,-2,0]])), #
        ("Dmm_123.dat", "basis_123.dat", np.array([[1,2,3],[-1,-2,-3]])), #
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
        ("Dmm_000.dat", "basis_000.dat", np.array([[0,0,0]])),
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

    tests = [
        (os.path.join(base, "spinless"), test_spinless, test_setup1),
        (os.path.join(base, "Npi"), test_nucleon_pi, test_setup2),
        (os.path.join(base, "np"), test_np, test_setup2),
    ]
    for base, test_fcn, setup in tests:
        print("Testing against files in", base)
        for fname_rep, fname_basis, momenta in setup:
            print("#"*40)
            test_fcn(momenta,
                     os.path.join(base, fname_rep),
                     os.path.join(base, fname_basis))

# ----- end main ----- #

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


def test_spinless(momenta, fname_rep, fname_basis):
    """
    Tests "pion-kaon" tables.

    Verifies construction of momentum-orbit representation matrices as well as
    the change-of-basis coefficients against tabulated reference data for two
    distinguishable spin-zero particles.

    Parameters
    ----------
    momenta : (nmomenta, 3) or (3, ) array_like
        The ordered momenta, with shape.
    fname_rep : str
        Path to the input file containing the data for the momentum orbit
        representation.
    fname_basis : str
        Path to the input file containing the data for the change-of-basis
        coefficients

    Returns
    -------
    None
    """
    oh = mhi.make_oh()
    little, _ = mhi.make_little_and_stabilizer(momenta, oh)
    orbit = mhi.make_momentum_orbit(momenta, little)
    Dmm = mhi.make_momentum_orbit_rep(orbit, little)
    Dmumu = mhi.make_irrep_from_group(little)

    # Check momentum-orbit representation matrices
    ref = read_mathematica(fname_rep, Dmm.shape)
    assert np.allclose(Dmm, ref), f"Trouble with {fname_rep}"
    print("Success:", os.path.split(fname_rep)[-1])

    # Check change-of-basis coefficients
    proj = mhi.project_basis(Dmm, Dmumu, verbose=True)
    table = make_table(proj)
    ref = read_mathematica(fname_basis, table.shape)
    assert np.allclose(table, ref), f"Error: trouble with {fname_basis}"
    print("Success:", os.path.split(fname_basis)[-1])


def test_nucleon_pi(momenta, fname_rep, fname_basis):
    """
    Tests "nucleon-pion" tables.

    Verifies construction of momentum-spin-orbit representation matrices as
    well as the change-of-basis coefficients against tabulated reference data
    for a pair of particles with spin-1/2 and spin-zero, respectively.

    Parameters
    ----------
    momenta : (nmomenta, 3) or (3, ) array_like
        The ordered momenta, with shape.
    fname_rep : str
        Path to the input file containing the data for the momentum orbit
        representation.
    fname_basis : str
        Path to the input file containing the data for the change-of-basis
        coefficients

    Returns
    -------
    None
    """
    oh = mhi.make_oh()
    little, _ = mhi.make_little_and_stabilizer(momenta, oh)
    little_double = mhi.make_spinorial_little_group(little)
    orbit = mhi.make_momentum_orbit(momenta, little)

    # Make momentum-orbit representation matrices
    Dmm = mhi.make_momentum_orbit_rep(orbit, little)

    # Make nucleon irrep matrices
    basis = basis_functions.basis_spinors['nucleon']['nucleon']
    Dmumu_nucleon = mhi.make_irrep_spinor(basis, little_double)

    # Make pion double-cover irrep matrices
    idxs = np.hstack([np.where([np.allclose(gg, g) for gg in oh]) for g in little])
    idxs = np.hstack([idxs, idxs+48]).squeeze()
    Dmumu_pion = np.vstack(2*[mhi.make_irrep_from_group(oh)['A1m']])
    Dmumu_pion = Dmumu_pion[idxs]

    # Combine momentum-orbit rep and particle-spin irrep matrices
    Dmm_momspin = mhi.make_momentum_spin_rep(Dmm, Dmumu_nucleon, Dmumu_pion)

    # Make double_cover irrep matrices
    Dmumu_double = mhi.make_irrep_from_groupD(little)

    ref = read_mathematica(fname_rep, Dmm_momspin.shape)

    assert np.allclose(Dmm_momspin, ref),\
        f"Trouble with {os.path.split(fname_rep)[-1]}"
    print("Success:", os.path.split(fname_rep)[-1])

    proj = mhi.project_basis(Dmm_momspin, Dmumu_double)
    table = make_table(proj)
    ref = read_mathematica(fname_basis, table.shape)

    assert np.allclose(table, ref), f"Error: trouble with {fname_basis}"
    print("Success:", os.path.split(fname_basis)[-1])


def test_np(momenta, fname_rep, fname_basis):
    """
    Tests "neutron-proton" tables.

    Verifies construction of momentum-spin-orbit representation matrices as
    well as the change-of-basis coefficients against tabulated reference data
    for two distinguishable spin-1/2 particles.

    Parameters
    ----------
    momenta : (nmomenta, 3) or (3, ) array_like
        The ordered momenta, with shape.
    fname_rep : str
        Path to the input file containing the data for the momentum orbit
        representation.
    fname_basis : str
        Path to the input file containing the data for the change-of-basis
        coefficients

    Returns
    -------
    None

    """
    oh = mhi.make_oh()
    little, _ = mhi.make_little_and_stabilizer(momenta, oh)
    little_double = mhi.make_spinorial_little_group(little)
    orbit = mhi.make_momentum_orbit(momenta, little)

    # Make momentum-orbit representation matrices
    Dmm = mhi.make_momentum_orbit_rep(orbit, little)

    # Make nucleon irrep matrices
    basis = basis_functions.basis_spinors['nucleon']['nucleon']
    Dmumu_nucleon = mhi.make_irrep_spinor(basis, little_double)

    # Combine momentum-orbit rep and particle-spin irrep matrices
    Dmm_momspin = mhi.make_momentum_spin_rep(Dmm, Dmumu_nucleon, Dmumu_nucleon)

    ref = read_mathematica(fname_rep, Dmm_momspin.shape)
    assert np.allclose(Dmm_momspin, ref),\
          f"Trouble with {os.path.split(fname_rep)[-1]}"
    print("Success:", os.path.split(fname_rep)[-1])

    Dmumu_double = mhi.make_irrep_from_groupD(little)
    proj = mhi.project_basis(Dmm_momspin, Dmumu_double)
    table = make_table(proj)
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

if __name__ == "__main__":
    main()
