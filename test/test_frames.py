"""Frame-orientation tests for mhi.mhi: little-group ordering, double-cover lifts, and the change-of-basis
matrices in every frame d with |d_i| <= 2.

Background (fixed 2026-10): element j of a frame's little group must be g.C_j.g^-1 with C_j the canonical
element and g the orienting element from find_subgroup_isomorphism, and its double cover must be
g^.C^_j.g^^-1 with g^ an OhD lift of g.  Stock MHI used little[perm] (correct only when perm is an
involution; wrong in the off-diagonal C3v frames such as (-1,1,1)) and looked up each spinor lift by its
position in OhD (wrong sign for some elements in e.g. d = (1,0,0), (2,0,0), (-1,1,1)).  The first bug broke
every system there; the second only systems with an odd number of half-integer spins (e.g. N pi), and in
some frames silently swapped labels (F1 <-> F2) instead of failing.

Run with ``pytest test/`` from the repository root after ``pip install -e ".[test]"``
"""
import itertools

import numpy as np
import pytest

from mhi import mhi as M

OH = M.make_oh()
OHD = M.make_ohd()
FRAMES = [d for d in itertools.product(range(-2, 3), repeat=3) if any(d)]


def frame_groups(d):
    """(little, canonical, isomorphism) exactly as mhi.mhi builds them."""
    momenta = np.array([np.zeros(3, int), np.array(d)])
    little, _ = M.make_little_and_stabilizer(momenta, group=OH)
    canonical = M.make_canonical_stabilizer(M.identify_stabilizer(little), group=OH)
    iso = M.find_subgroup_isomorphism(OH, canonical, little)
    return little, canonical, iso


def lift(g):
    return OHD[[np.allclose(elt, g) for elt in OH].index(True)]


def homomorphism_violation(group, irrep):
    """max |D(a) D(b) - D(ab)| over the group (elements matched by value)."""
    index = {tuple(np.round(x, 8).ravel() + 0.0): i for i, x in enumerate(group)}
    worst = 0.0
    for a in range(len(group)):
        for b in range(len(group)):
            c = index[tuple(np.round(group[a] @ group[b], 8).ravel() + 0.0)]
            worst = max(worst, np.abs(irrep[a] @ irrep[b] - irrep[c]).max())
    return worst


# ---------------------------------------------------------------------------------------------------
# group-level properties
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", FRAMES, ids=str)
def test_isomorphism_contract(d):
    """find_subgroup_isomorphism: g.C[perm[i]].g^-1 = little[i], and argsort(perm) = perm^-1."""
    little, canonical, iso = frame_groups(d)
    conj = M.conjugate_group(iso.g, canonical)
    assert np.allclose(conj[iso.perm], little)
    assert np.array_equal(iso.perm[np.argsort(iso.perm)], np.arange(len(iso.perm)))
    # the canonical-order frame group, by reordering and by conjugation, bitwise identical
    assert little[np.argsort(iso.perm)].tobytes() == conj.tobytes()


@pytest.mark.parametrize("d", FRAMES, ids=str)
def test_single_cover_irreps_are_representations_of_the_frame_group(d):
    _, canonical, iso = frame_groups(d)
    frame = M.conjugate_group(iso.g, canonical)
    for name, D in M.make_irrep_from_group(canonical).items():
        assert homomorphism_violation(frame, D) < 1e-12, name


@pytest.mark.parametrize("d", FRAMES, ids=str)
def test_double_cover_irreps_are_representations_of_the_frame_group(d):
    """The spinorial irreps tabulated on the canonical double group must be a representation of the frame's
    double group in the same element order, g^.C^_j.g^^-1."""
    _, canonical, iso = frame_groups(d)
    g_hat = lift(iso.g)
    frame_double = M.conjugate_group(g_hat, M.make_spinorial_little_group(canonical))
    for name, D in M.make_irrep_from_groupD(canonical).items():
        assert homomorphism_violation(frame_double, D) < 1e-12, name


def test_conjugate_inverts_unitary_elements():
    """conjugate(g, h) = g.h.g^-1 for the complex unitary OhD elements, not only for real orthogonal Oh."""
    rng = np.random.default_rng(0)
    for i in rng.choice(len(OHD), 12, replace=False):
        g, h = OHD[i], OHD[(i + 7) % len(OHD)]
        assert np.allclose(M.conjugate(g, h), g @ h @ np.linalg.inv(g))
    for g in OH:   # real orthogonal: unchanged, bitwise
        assert M.conjugate(g, OH[5]).tobytes() == (g @ OH[5] @ g.T).tobytes()


def test_canonicalize_makes_equal_elements_hash_equal():
    a = np.array([[0.0, 1.0], [-1.0, 0.0]])
    b = np.array([[-0.0, 1.0 - 1e-16], [-1.0, 0.0]])
    assert M.force_hash(a) != M.force_hash(b)              # raw bytes differ
    assert M.force_hash(M.canonicalize(a)) == M.force_hash(M.canonicalize(b))
    z = np.array([1j * 0.0, -0.0 + 0j])
    assert M.force_hash(M.canonicalize(z)) == M.force_hash(M.canonicalize(np.zeros(2, complex)))


# ---------------------------------------------------------------------------------------------------
# full decompositions: pi pi (identical bosons), NN (identical fermions), N pi (odd number of fermions)
# ---------------------------------------------------------------------------------------------------

SYSTEMS = {
    "pipi": (None, lambda: M.symmetrizer([0, 1], signed=0, n=2)),
    "NN": (["G1p", "G1p"], lambda: M.symmetrizer([0, 1], signed=1, n=2)),
    "Npi": (["G1p", "A1m"], lambda: None),
}


def check_decomposition(momenta, spins, sym):
    res = M.mhi(np.asarray(momenta), spins, sym())       # mhi's own asserts run inside apply_schur
    if not res.decomp:
        return
    U = np.vstack(list(res.decomp.values()))
    assert np.allclose(U @ U.conj().T, np.eye(len(U)), atol=1e-10), "rows not orthonormal"
    for (irrep, _), Ub in res.decomp.items():           # MHI's convention D = U* Dmm U^T
        D = np.einsum("ia,rab,jb->rij", Ub.conj(), res.Dmm, Ub)
        assert np.allclose(D, res.Dmumu[irrep], atol=1e-10), f"{irrep} not block diagonal"


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("d", FRAMES, ids=str)
def test_decomposition_every_frame(d, system):
    spins, sym = SYSTEMS[system]
    n1 = np.array([1, 0, 0])
    check_decomposition([n1, np.array(d) - n1], spins, sym)


# frames that failed before the fixes (kept explicit so a regression names the case)
@pytest.mark.parametrize("d,system", [
    ((-1, 1, 1), "pipi"), ((1, -1, 1), "pipi"), ((1, 1, -1), "pipi"),     # little[perm] ordering
    ((-1, 1, 1), "NN"), ((1, -1, 1), "NN"),
    ((1, 0, 0), "Npi"), ((2, 0, 0), "Npi"), ((-1, 1, 1), "Npi"),         # spinor lift signs
    ((-2, 0, 2), "Npi"), ((-2, 0, 0), "Npi"),
], ids=str)
def test_previous_failures(d, system):
    spins, sym = SYSTEMS[system]
    for n1 in ((0, 0, 0), (1, 0, 0), (0, 1, 1)):
        n1 = np.array(n1)
        check_decomposition([n1, np.array(d) - n1], spins, sym)


def test_no_silent_label_swap_c3v():
    """At d = (1,1,-1) the position-based spinor lift differs from g^.C^.g^^-1 by a sign pattern that is a
    1-d character, so the old code passed every assert but labelled the N pi F1 block as F2.  The F1/F2
    blocks must be the same subspaces as in the frame (1,1,1) case transported by the orienting element."""
    _, canonical, iso = frame_groups((1, 1, -1))
    g_hat = lift(iso.g)
    frame_double = M.conjugate_group(g_hat, M.make_spinorial_little_group(canonical))
    lifted = M.make_spinorial_little_group(M.conjugate_group(iso.g, canonical))
    signs = np.sign(np.real(np.einsum("rij,rij->r", lifted.conj(), frame_double)))
    # the position-based lift is NOT the conjugated one here (the condition that used to mislabel) ...
    assert not np.all(signs > 0)
    # ... and mhi.mhi now uses the conjugated one: its N pi blocks are labelled consistently
    res = M.mhi(np.array([[1, 0, 0], [0, 1, -1]]), ["G1p", "A1m"], None)
    for (irrep, _), Ub in res.decomp.items():
        D = np.einsum("ia,rab,jb->rij", Ub.conj(), res.Dmm, Ub)
        assert np.allclose(D, res.Dmumu[irrep], atol=1e-10), irrep


# ---------------------------------------------------------------------------------------------------
# the original script-style reference tests (test_mhi.py at the repository root)
# ---------------------------------------------------------------------------------------------------

def test_reference_script(monkeypatch):
    import pathlib
    import runpy
    root = pathlib.Path(__file__).resolve().parents[1]
    if not (root / "test_mhi.py").exists() or not (root / "test" / "data").exists():
        pytest.skip("test_mhi.py / test/data not present")
    monkeypatch.chdir(root)                              # the script reads ./test/data
    module = runpy.run_path(str(root / "test_mhi.py"))
    module["main"]()
