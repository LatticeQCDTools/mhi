"""
Command-line script to check construction of "bosonic rotations" and to
verify irrep conventions against those appearing in the literature in
    C. Morningstar et al., Phys.Rev.D 88 (2013) 1, 014511
    "Extended hadron and two-hadron operators of definite momentum for
    spectrum calculations in lattice QCD"
    [https://arxiv.org/pdf/1303.6816.pdf]
"""
import numpy as np
from mhi import mhi

def main():
    """
    Main function running the tests for checking construction of rotations as
    well as consistency of irrep matrices against the conventions in the
    literature from Morningstar et al.
    """
    oh = mhi.make_oh()
    check_rotations()
    check_oh(oh)
    check_c4v(oh)
    check_c3v(oh)
    check_c2v(oh)


def check_rotations():
    """
    Checks routines for constructing "bosonic" rotation matrices against the
    expected behavior acting on reference vectors.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    c2x = mhi.rotation(np.pi, 1)
    c2y = mhi.rotation(np.pi, 2)
    c2z = mhi.rotation(np.pi, 3)
    c4x = mhi.rotation(np.pi/2, 1)
    c4y = mhi.rotation(np.pi/2, 2)
    c4z = mhi.rotation(np.pi/2, 3)
    c4xi = mhi.rotation(-np.pi/2, 1)
    c4yi = mhi.rotation(-np.pi/2, 2)
    c4zi = mhi.rotation(-np.pi/2, 3)

    # Verify "(group element) @ vec_reference = vec_expected"
    # checks = {vec_reference : [(group_elt1, vec_expected1), ...] }
    checks = {
        (0,0,1) : [
            (c2x,  [0, 0, -1]),
            (c4y,  [1, 0, 0]),
            (c4yi, [-1, 0, 0]),
            (c4x,  [0, -1, 0]),
            (c4xi, [0, 1, 0]),],
        (0,1,1) : [
            (c2x,       [0, -1, -1]),
            (c4xi,      [0, 1, -1]),
            (c4x,       [0, -1, 1]),
            (c4zi,      [1, 0, 1]),
            (c2x @ c4z, [-1, 0, -1]),
            (c2y @ c4z, [1, 0, -1]),
            (c4z,       [-1, 0, 1]),
            (c4y,       [1, 1, 0]),
            (c2z @ c4y, [-1, -1, 0]),
            (c4y @ c2z, [1, -1, 0]),
            (c4yi,      [-1, 1, 0]),],
        (1,1,1) : [
            (c4y, [1, 1, -1]),
            (c4x, [1, -1, 1]),
            (c2x, [1, -1, -1]),
            (c4z, [-1, 1, 1]),
            (c2y, [-1, 1, -1]),
            (c2z, [-1, -1, 1]),
            (c2z @ c4y, [-1, -1, -1]),],
    }
    for vec_reference, pairs in checks.items():
        for arr, vec_expected in pairs:
            assert np.allclose(arr @ vec_reference, vec_expected)
        print("Success:", vec_reference)


def locate(element, group):
    """
    Locates the index of the specified element within the group.

    Parameters
    ----------
    element : (n, n) ndarray
        The group element to be found
    group : (|G|, n, n) ndarray
        The group matrices

    Returns
    -------
    idx : int
        The location of the group element within the group
    """
    return np.argwhere([np.allclose(g, element) for g in group]).item()


def reverse(arr):
    """
    Reverses the order of the rows and columns of the array, corresponding
    to reversing the order of the associated basis vectors.

    Parameters
    ----------
    arr : (n, n) ndarray
        The array

    Returns
    -------
    new_arr : (n, n) ndarray
        The array with all rows and columns reversed
    """
    return arr[::-1, ::-1]


def check_oh(oh):
    """
    Checks reference values for representation matrices of the single- and
    double-valued irreps of Oh for zero-momentum operators.

    Parameters
    ----------
    oh : (48, 3, 3)
        The group matrices for Oh
    ohd : (96, 3, 3)
        The group matrices fo OhD

    Returns
    -------
    None

    Notes
    -----
    Comparisons made against reference values given in Table XI of
    [https://arxiv.org/pdf/1303.6816.pdf]
    """
    dirac = mhi.DiracPauli()
    ohd = mhi.make_spinorial_little_group(oh)
    idx1 = locate(dirac.rotation(np.pi/2, 2), ohd)
    idx2 = locate(dirac.rotation(np.pi/2, 3), ohd)
    idxp = locate(dirac.g4, ohd)
    Dmumu = mhi.make_irrep_from_groupD(oh)

    # A1p irrep
    assert np.allclose(Dmumu['A1p'][idx1], np.array([[1]]))
    assert np.allclose(Dmumu['A1p'][idx2], np.array([[1]]))

    # A2p irrep
    assert np.allclose(Dmumu['A2p'][idx1], np.array([[-1]]))
    assert np.allclose(Dmumu['A2p'][idx2], np.array([[-1]]))

    # Ep irrep
    assert np.allclose(reverse(Dmumu['Ep'][idx1]),
                       0.5 * np.array([[1, np.sqrt(3)],[np.sqrt(3), -1]]))
    assert np.allclose(reverse(Dmumu['Ep'][idx2]),
                       np.array([[-1, 0],[0, 1]]))

    # G1p irrep
    assert np.allclose(Dmumu['G1p'][idx1], np.array([[1,-1],[1,1]]) / np.sqrt(2))
    assert np.allclose(Dmumu['G1p'][idx2], np.array([[1-1j,0],[0,1+1j]]) / np.sqrt(2))

    # G2p irrep
    assert np.allclose(Dmumu['G2p'][idx1], -np.array([[1,-1],[1,1]]) / np.sqrt(2))
    assert np.allclose(Dmumu['G2p'][idx2], -np.array([[1-1j,0],[0,1+1j]]) / np.sqrt(2))

    # Hp irrep
    ref_c4y = np.array([
        [1, -np.sqrt(3), np.sqrt(3), -1],
        [np.sqrt(3), -1, -1, np.sqrt(3)],
        [np.sqrt(3), +1, -1,-np.sqrt(3)],
        [1, np.sqrt(3), np.sqrt(3), 1],]) / (2*np.sqrt(2))
    ref_c4z = np.array([
        [-1-1j, 0, 0, 0],
        [0, 1-1j, 0, 0],
        [0, 0, 1+1j, 0],
        [0, 0, 0, -1+1j],]) / (np.sqrt(2))
    assert np.allclose(Dmumu["Hp"][idx1], ref_c4y)
    assert np.allclose(Dmumu["Hp"][idx2], ref_c4z)

    # parity matrices in all irreps
    for irrep, matrices in Dmumu.items():
        sign = +1 if irrep.endswith("p") else -1
        dim = matrices.shape[1]
        assert np.allclose(matrices[idxp], sign*np.eye(dim))

    print("Success: Oh")


def check_c4v(oh):
    """
    Checks reference values for representation matrices of the single- and
    double-valued irreps of the subgroup C4v of Oh for momentum in the
    direction (0, 0, 1).

    Parameters
    ----------
    oh : (48, 3, 3)
        The group matrices for Oh

    Returns
    -------
    None

    Notes
    -----
    Comparisons made against reference values given in Table XII of
    [https://arxiv.org/pdf/1303.6816.pdf]
    """
    little = mhi.make_stabilizer(momenta=[0,0,1], group=oh)
    little_double = mhi.make_spinorial_little_group(little)
    dirac = mhi.DiracPauli()
    idx1 = locate(dirac.rotation(np.pi/2, 3), little_double)
    idx2 = locate( dirac.g4 @ dirac.rotation(np.pi, 2), little_double)
    Dmumu = mhi.make_irrep_from_groupD(little)

    # A1 irrep
    assert np.allclose(Dmumu['A1'][idx1], np.eye(1))
    assert np.allclose(Dmumu['A1'][idx2], np.eye(1))

    # A2 irrep
    assert np.allclose(Dmumu['A2'][idx1], np.eye(1))
    assert np.allclose(Dmumu['A2'][idx2], -np.eye(1))

    # B1 irrep
    assert np.allclose(Dmumu['B1'][idx1], -np.eye(1))
    assert np.allclose(Dmumu['B1'][idx2], np.eye(1))

    # B2 irrep
    assert np.allclose(Dmumu['B2'][idx1], -np.eye(1))
    assert np.allclose(Dmumu['B2'][idx2], -np.eye(1))

    # E irrep
    assert np.allclose(Dmumu['E'][idx1], np.array([[0, -1], [1, 0]]))
    assert np.allclose(Dmumu['E'][idx2], np.array([[1, 0], [0, -1]]))

    # G1 irrep
    assert np.allclose(Dmumu['G1'][idx1], np.array([[1-1j, 0],[0, 1+1j]])/np.sqrt(2))
    assert np.allclose(Dmumu['G1'][idx2], np.array([[0, -1],[1, 0]]))

    # G2 irrep
    assert np.allclose(Dmumu['G2'][idx1], -np.array([[1-1j, 0],[0, 1+1j]])/np.sqrt(2))
    assert np.allclose(Dmumu['G2'][idx2], np.array([[0, -1],[1, 0]]))

    print("Success: C4v")


def check_c3v(oh):
    """
    Checks reference values for representation matrices of the single- and
    double-valued irreps of the subgroup C3v of Oh for momentum in the
    direction (1, 1, 1).

    Parameters
    ----------
    oh : (48, 3, 3)
        The group matrices for Oh

    Returns
    -------
    None

    Notes
    -----
    Comparisons made against reference values given in Table XIV of
    [https://arxiv.org/pdf/1303.6816.pdf]
    """
    dirac = mhi.DiracPauli()
    c2xd = dirac.rotation(np.pi, 1)
    c4yd = dirac.rotation(np.pi/2, 2)
    c4zd = dirac.rotation(np.pi/2, 3)

    # C3v Subgroup
    little = mhi.make_stabilizer(momenta=[1,1,1], group=oh)
    little_double = mhi.make_spinorial_little_group(little)
    Dmumu = mhi.make_irrep_from_groupD(little)
    idx1 = locate(c4yd@c4zd, little_double)
    idx2 = locate(dirac.g4@c2xd@c4zd, little_double)

    # A1 irrep
    assert np.allclose(Dmumu['A1'][idx1], np.eye(1))
    assert np.allclose(Dmumu['A1'][idx2], np.eye(1))

    # A2 irrep
    assert np.allclose(Dmumu['A2'][idx1], np.eye(1))
    assert np.allclose(Dmumu['A2'][idx2], -np.eye(1))

    # E irrep
    assert np.allclose(reverse(Dmumu['E'][idx1]),
                       0.5*np.array([[-1, np.sqrt(3)], [-np.sqrt(3), -1]]))
    assert np.allclose(reverse(Dmumu['E'][idx2]),
                       np.array([[-1, 0], [0, 1]]))

    # F1 irrep
    assert np.allclose(Dmumu['F1'][idx1], -np.eye(1))
    assert np.allclose(Dmumu['F1'][idx2], 1j*np.eye(1))

    # F2 irrep
    assert np.allclose(Dmumu['F2'][idx1], -np.eye(1))
    assert np.allclose(Dmumu['F2'][idx2], -1j*np.eye(1))

    # G irrep
    assert np.allclose(Dmumu['G'][idx1], 0.5*np.array([[1-1j, -1-1j],[1-1j, 1+1j]]))
    assert np.allclose(Dmumu['G'][idx2], np.array([[0, 1-1j],[-1-1j, 0]])/np.sqrt(2))

    print("Success: C3v")


def check_c2v(oh):
    """
    Checks reference values for representation matrices of the single- and
    double-valued irreps of the subgroup C2v of Oh for momentum in the
    direction (0, 1, 1).

    Parameters
    ----------
    oh : (48, 3, 3)
        The group matrices for Oh

    Returns
    -------
    None

    Notes
    -----
    Comparisons made against reference values given in Table XIII of
    [https://arxiv.org/pdf/1303.6816.pdf]
    """
    dirac = mhi.DiracPauli()
    c2yd = dirac.rotation(np.pi, 2)
    c2zd = dirac.rotation(np.pi, 3)
    c4xd = dirac.rotation(np.pi/2, 1)

    # C2v Subgroup
    little = mhi.make_stabilizer(momenta=[0,1,1], group=oh)
    little_double = mhi.make_spinorial_little_group(little)
    idx1 = locate(c2zd@c4xd, little_double)
    idx2 = locate(dirac.g4@c2yd@c4xd, little_double)
    Dmumu = mhi.make_irrep_from_groupD(little)

    # A1 irrep
    assert np.allclose(Dmumu['A1'][idx1], np.eye(1))
    assert np.allclose(Dmumu['A1'][idx2], np.eye(1))

    # A2 irrep
    assert np.allclose(Dmumu['A2'][idx1], np.eye(1))
    assert np.allclose(Dmumu['A2'][idx2], -np.eye(1))

    # B1 irrep
    assert np.allclose(Dmumu['B1'][idx1], -np.eye(1))
    assert np.allclose(Dmumu['B1'][idx2], np.eye(1))

    # B2 irrep
    assert np.allclose(Dmumu['B2'][idx1], -np.eye(1))
    assert np.allclose(Dmumu['B2'][idx2], -np.eye(1))

    # G irrep
    assert np.allclose(Dmumu['G'][idx1], np.array([[-1j, -1],[1, 1j]])/np.sqrt(2))
    assert np.allclose(Dmumu['G'][idx2], np.array([[1j, -1],[1, -1j]])/np.sqrt(2))

    print("Success: C2v")


if __name__ == '__main__':
    main()
