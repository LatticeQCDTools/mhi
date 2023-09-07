"""
MHI -- "Multi-Hadron Interpolators"

Module for constructing block-diagonalization / change-of-basis matrices to map
products of N local plane-wave operators into irreps of the cubic group.
Includes appropriate generalizations for spin, identical-particle exchange
symmetry, and isospin (TODO!).

Authors:
William Detmold, William I. Jay, Gurtej Kanwar, Phiala E. Shanahan, and Michael L. Wagman

* Check for TODO flag in the docs
* Rerun tests after lastest refactoring
* Add main function giving support as a command-line utility
* Making a few wrapper/driver classes
* Add setup.py file for installation using pip
"""

import functools
import itertools
from collections import namedtuple
from hashlib import sha256
import sys
import os
import string
import numpy as np
import sympy
import yaml
import h5py
from . import basis_functions

####################
# Tensor utilities #
####################

def levi_civita(dim):
    """
    The totally antisymmetric Levi-Civita tensor in arbitrary dimensions.

    Parameters
    ----------
    dim : int
        The number of dimensions.

    Returns
    -------
    arr : ndarray
        The Levi-Civita tensor.

    Notes
    -----
    Implementation from StackOverflow user JGibs:
    https://stackoverflow.com/questions/59740966/levi-civita-tensor-in-numpy
    """
    arr = np.zeros(tuple(dim for _ in range(dim)), dtype=int)
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=int)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x] = int(np.linalg.det(mat))
    return arr

def unique_permutations(seq):
    """
    Yield only unique permutations of seq in an efficient way.

    Parameters
    ----------
    seq: array_like
        The elements to be permuted

    Returns
    -------
    seq: array_like
        The permuated sequences

    Examples
    --------
    >>> for perm in unique_permutations([1, 1, 2]):
    >>>     print(perm)
    [1, 1, 2]
    [1, 2, 1]
    [2, 1, 1]
    >>> for perm in unique_permutations(['a', 'a', 'a', 'b', 'b']):
    >>>     print(perm)
    ['a', 'a', 'a', 'b', 'b']
    ['a', 'a', 'b', 'a', 'b']
    ['a', 'a', 'b', 'b', 'a']
    ['a', 'b', 'a', 'a', 'b']
    ['a', 'b', 'a', 'b', 'a']
    ['a', 'b', 'b', 'a', 'a']
    ['b', 'a', 'a', 'a', 'b']
    ['b', 'a', 'a', 'b', 'a']
    ['b', 'a', 'b', 'a', 'a']
    ['b', 'b', 'a', 'a', 'a']

    Notes
    -----
    A python implementation of Knuth's "Algorithm L", also known from the
    std::next_permutation function of C++, and as the permutation algorithm
    of Narayana Pandita.

    Code taken from a post by StackOverflow user Lauritz V. Thaulow:
    https://stackoverflow.com/questions/12836385/how-can-i-interleave-or-create-unique-permutations-of-two-strings-without-recur/12837695#12837695
    """

    # Precalculate the indices we'll be iterating over for speed
    i_indices = list(range(len(seq) - 1, -1, -1))
    k_indices = i_indices[1:]

    # The algorithm specifies to start with a sorted version
    seq = sorted(seq)

    while True:
        yield seq

        # Working backwards from the last-but-one index,           k
        # we find the index of the first decrease in value.  0 0 1 0 1 1 1 0
        for k in k_indices:
            if seq[k] < seq[k + 1]:
                break
        else:
            # Introducing the slightly unknown python for-else syntax:
            # else is executed only if the break statement was never reached.
            # If this is the case, seq is weakly decreasing, and we're done.
            return

        # Get item from sequence only once, for speed
        k_val = seq[k]

        # Working backwards starting with the last item,           k     i
        # find the first one greater than the one at k       0 0 1 0 1 1 1 0
        for i in i_indices:
            if k_val < seq[i]:
                break

        # Swap them in the most efficient way
        (seq[k], seq[i]) = (seq[i], seq[k])                #       k     i
                                                           # 0 0 1 1 1 1 0 0

        # Reverse the part after but not                           k
        # including k, also efficiently.                     0 0 1 1 0 0 1 1
        seq[k + 1:] = seq[-1:k:-1]

def symmetrize(arr):
    """
    Computes the completely symmetrized version of the input array.

    Parameters
    ----------
    arr : array_like
        A tensor of arbitrary rank.

    Returns
    -------
    arr_sym : ndarray
        The completely symmetrized version of the input tensor.

    Notes
    -----
    In index notation symmetrization is the map which sends
    T_{ij...n} to  T_{(ij...n)}, where parantheses denote symmetrization.
    For instance, the symmetrization of a three-index tensor T_{ijk} is
    T_{(ijk)} = (T_{ijk} + T_{ikj} + T_{jik} + T_{jki} + T_{kij} + T_{kji})/6
    """
    rank = len(arr.shape)
    dim = np.unique(arr.shape).item()
    # Run over unique index combinations
    # For instance, {(1,0,0), (0,1,0), (0,0,1)} should be identified
    arr_sym = np.zeros(arr.shape, dtype=complex)
    for idxs in itertools.combinations_with_replacement(range(dim), r=rank):
        # Run over permuations of a given index combination
        # For instance, (1,0,0) --> [(1,0,0), (0,1,0), (0,0,1)]
        perms = [tuple(p) for p in unique_permutations(idxs)]
        sym = sum(arr[p] for p in perms) / len(perms)
        for perm in perms:
            arr_sym[perm] = sym
    return arr_sym

def tensor_product(a, b):
    """
    Computes the tensor product between tensors a and b.

    Parameters
    ----------
    a : array_like
    b : array_like

    Returns
    -------
    tensor : array_like
        The tensor product of arrays "a" and "b".

    Notes
    -----
    In index notation, this fuction computes
    tensor[i,j,...k,r,s,...t] = a[i,j,...k]*b[r,s,...,t].
    """
    return np.tensordot(a, b, axes=0)

def tensor_nfold(*tensors):
    """
    Computes the n-fold tensor product between all the tensors in a list.

    Parameters
    ----------
    tensors : (variable number of) ndarray
        The input tensors for the product.

    Returns
    -------
    tensor : ndarray
        The product of all the tensors.

    Notes
    -----
    Suppose the input tensors are {a[i], b[j,k], c[l,m,n]}.
    This function computes the product
    tensor[i,j,k,l,m,n] = a[i]*b[j,k]*c[l,m,n]
    """
    return functools.reduce(tensor_product, tensors)


def decompose(monomial):
    """
    Decomposes a monomial of the form c * x**nx * y**ny * z**nz
    into a coefficient "c" and a triplet of exponents "(nx, ny, nz)".

    Parameters
    ----------
    monomial : sympy.core.mul.Mul
        A monomial in the variables {'x', 'y', 'z'}

    Returns
    -------
    (coeff, exponents) : (complex, tuple)
        The coefficient and exponents (nx, ny, nz) of the monomial
    """
    X, Y, Z = sympy.symbols('x y z')
    nx = sympy.degree(monomial, X)
    ny = sympy.degree(monomial, Y)
    nz = sympy.degree(monomial, Z)
    coeff = complex(monomial.coeff(X**nx * Y**ny * Z**nz))
    return coeff, (nx, ny, nz)


def compute_polarization(monomial):
    """
    Computes the polarization tensor of a given monomial.

    Parameters
    ----------
    monomial : sympy.core.mul.Mul
        A monomial in the variables {'x', 'y', 'z'}

    Returns
    -------
    polarization_tensor : ndarray
        The polarization, a totally symmetric tensor
    """
    coeff, (nx, ny, nz) = decompose(monomial)
    x, y, z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    args = nx*[x] + ny*[y] + nz*[z]
    return coeff * symmetrize(tensor_nfold(*args))


def contract_across(arr, vec):
    """
    Conctract a vector full across all the free indices, yielding a scalar.

    Parameters
    ----------
    arr : array_like
        Tensor of generic rank
    vec : array_like
        The vector to contract against all the free indices

    Returns
    -------
    scalar : float or complex
        The product c = vec[i]*vec[j]*...*vec[n]*arr[i,j,...,n]
    """
    rank = len(arr.shape)
    while rank > 0:
        arr = np.tensordot(vec, arr, axes=1)
        rank = len(arr.shape)
    return arr.item()


def compute_restitution(polarization):
    """
    Computes the monomial restitution of a polarization tensor.

    Parameters
    ----------
    polarization : array_like
        The (completely symmetric) polarization tensor

    Returns
    -------
    restitution : monomial (sympy.core.mul.Mul)
        The monomial resulting from contracting the vector [x,y,z] across all
        the free indices of the polarization tensor.
    """
    return contract_across(polarization, vec=np.array(sympy.symbols('x y z')))


def polarize(fcn):
    """
    Computes the polarization tensor associated with a basis function.

    Parameters
    ----------
    fcn : sympy basis function
        The basis function, a sum of monomials in {'x', 'y', 'z'}.

    Returns
    -------
    arr : ndarray
        The polarization tensor of rank d, where d is the degree of the
        monomials within the basis function.
    """
    fcn = fcn.expand()
    if not isinstance(fcn, (sympy.Mul, sympy.Pow, sympy.Add)):
        raise ValueError("Bad type", type(fcn))

    if isinstance(fcn, (sympy.Mul, sympy.Pow)):
        # fcn is already a monomial
        arr = compute_polarization(fcn)
    else:
        # fcn is generically a polynomial, so run through monomials in the sum
        arr = 0
        summands = fcn.expand().args
        for monomial in summands:
            arr = arr + compute_polarization(monomial)
    return arr


def transform(arr, group_element, verbose=False):
    """
    Computes the "rotation" transformation of a tensor of arbitrary rank,
    A[a,b,...,c] --> R[a,x] R[b,y] ... R[c,z] A[x,y,...,z].

    Parameters
    ----------
    arr : (M, M, ..., M) array_like
        The tensor to transform.
    group_element : (M, M) array_like
        The group element applying the transformation.
    verbose : bool
        Whether to print the indices used in Einstein summation notation.

    Returns
    -------
    arr_transformed : ndarray
        The transformed tensor.
    """
    rank = len(arr.shape)
    # Assemble indices Einstein summation notation.
    # For example, for transforming a rank-3 tensor:
    # R[ad]*R[be]*R[cf]*arr[def] --> ad,be,cf,def
    if 2*rank > len(string.ascii_lowercase):
        raise ValueError("Rank too large for current implementation.")
    idxs1 = string.ascii_lowercase[:rank]
    idxs2 = string.ascii_lowercase[rank:2*rank]
    subscripts = ",".join([
        ",".join([f"{i1}{i2}" for i1, i2 in zip(idxs1, idxs2)]),
        idxs2
    ])
    if verbose:
        print(subscripts)
    args = rank * [group_element] + [arr]  # einsum eats a list of tensors
    return np.einsum(subscripts, *args, optimize='greedy')

def tensor_inner(a, b, verbose=False):
    """
    Computes the inner product between two tensors a[i,j,...,k]*b[i,j,...,k].

    Parameters
    ----------
    a : (M, M, ... , M) array_like
    b : (M, M, ... , M) array_like

    Returns
    -------
    c : float or complex
        The result of the inner product across all free indices
    """
    rank, rank2 = len(a.shape), len(b.shape)
    if rank != rank2:
        raise ValueError("Incommensurate ranks", rank, rank2)
    if rank > len(string.ascii_lowercase):
        raise ValueError("Rank too large for current implementation.")
    idxs = string.ascii_lowercase[:rank]
    subscripts = f"{idxs},{idxs}"
    if verbose:
        print(subscripts)
    return np.einsum(subscripts, np.conjugate(a), b, optimize='greedy')

def make_tensor_product_space(dims):
    """
    Makes a tensor product space with the specified dimensions.

    Parameters
    ----------
    dims : list
        Integers specifying the dimensions

    Returns
    -------
    tensor_product_space : ndarray

    Notes
    -----
    These spaces arise naturally in the construction of combined "momentum-
    spin" orbits below. For the portion of the orbit associated with spin,
    it's assumed that the spin has already been decomposed into irreps of
    known dimensions. By definition of the irrep space, group transformation
    simply permute the basis elements of the irrep space.

    Examples
    --------
    >>> arr = make_tensor_product_space([1,2,3])
    >>> print(arr.shape)
    (4, 2)
    >>> print(arr)
    array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 2]])
    """
    return np.array(list(idxs for idxs in itertools.product(*[range(dim) for dim in dims])))


###########
# Spinors #
###########

class DiracPauli:
    """
    Wrapper for Dirac matrices in the Dirac-Pauli basis.
    """
    def __init__(self, verbose=False):
        self.g1 = np.array(
            [[0, 0, 0, -1j],
            [0, 0, -1j, 0],
            [0, 1j, 0, 0],
            [1j, 0, 0, 0]])
        self.g2 = np.array(
            [[0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0]])
        self.g3 = np.array(
            [[0, 0, -1j, 0],
            [0, 0, 0, 1j],
            [1j, 0, 0, 0],
            [0, -1j, 0, 0]])
        self.g4 = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]])
        self.g5 = self.g1 @ self.g2 @ self.g3 @ self.g4
        self.eps3 = levi_civita(3)
        # Minkowski metric
        self.eta = np.block([
            [-np.eye(3), np.zeros((3,1))],
            [np.zeros((1,3)), np.eye(1)]])
        test_clifford([self.g1, self.g2, self.g3, self.g4], self.eta, verbose)
        test_gamma5([self.g1, self.g2, self.g3, self.g4], self.g5, verbose)


def get_nprod(j):
    """
    Gets the number of spin-1/2 copies present for a given half-integer j.

    For example, 3/2 = 1/2 \otimes 1/2 \otimes 1/2 --> 3 copies

    Parameters
    ----------
    j : int or float
        The total spin.

    Returns
    -------
    n : int
        The number of spin-1/2 copies present.
    """
    nprod = 2*j
    assert isinstance(nprod, int) or nprod.is_integer(),\
        "Error: total j should be half-integer."
    return int(nprod)


def make_spinor_array(spinor):
    """
    Builds a normalized "state" with total (j, jz, parity) inside a suitable
    tensor product space of spin-1/2 states.
    Parameters
    ----------
    spinor : SpinorTuple
        The namedtuple specifying (j, jz, parity)

    Returns
    -------
    vec : ndarray
        The normalized vector of length 4**(2*j)
    """
    nprod = get_nprod(spinor.j)
    assert spinor.parity in (-1, 1), "Error: parity must be +/- 1"

    # The basic spin-1/2 space is four dimensional, depending on the choice of
    # jz and parity. Represent the full space as a flattened tensor product of
    # (jz, parity) states
    spin_half_basis = [(+0.5, +1), (-0.5, +1), (+0.5, -1), (-0.5, -1)]
    basis = np.array(
        list(idxs for idxs in itertools.product(spin_half_basis, repeat=nprod))
    )

    # Find desired total jz: sum the constituent jz values for each state
    jz_mask = (np.sum(basis[:,:,0], axis=1) == spinor.jz)  # (jz, parity) -> jz

    # Find desired total parity. Note that:
    # all +1 --> positive
    # all -1 --> negative
    # else   --> mixed parity
    parity_sum = np.sum(basis[:,:,1], axis=1)  # (jz, parity) -> parity
    parity_mask = (parity_sum == spinor.parity*nprod)

    # Build normalized state with (j, jz, parity) in the tensor product space
    vec = np.zeros(len(basis))
    vec[jz_mask & parity_mask] = 1
    vec /= np.linalg.norm(vec)
    return vec


def make_spinor_array_lincombo(basis):
    """
    TODO: write docs
    """
    result = []
    for state in basis:
        result.append(
            np.sum([c*make_spinor_array(ket) for c, ket in state], axis=0))
    return np.array(result)


########################
# General group theory #
########################

def make_oh():
    """
    Constructs a presentation of the cubic group Oh with a standardized
    ordering of group elements.

    Parameters
    ----------
    None

    Returns
    -------
    group : (48, 3, 3) ndarray
    """
    # Reflections
    rx = np.diag([-1, 1, 1])
    ry = np.diag([1,-1, 1])
    rz = np.diag([1, 1, -1])
    id3 = np.eye(3)
    refls = [id3, rz, ry, ry@rz, rx, rx@rz, rx@ry, rx@ry@rz]

    # Permutations
    pxy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    pyz = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    pzy = pyz
    pzx = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    pxz = pzx
    perms = [id3, pxy, pyz, pxz, pxy@pyz, pxz@pzy]

    return np.array([r@p for r, p in itertools.product(refls, perms)])


def make_ohd():
    """
    Constructs a presentation of the "spinorial" double cover Oh^D of the cubic
    group ordering of group elements.

    Parameters
    ----------
    None

    Returns
    -------
    group : (96, 4, 4) ndarray
    """
    dirac = DiracPauli()

    id4 = np.eye(4)
    invs = [id4, -id4]  # Note: 2*pi rotation != id in OhD

    # Reflections
    Rx = dirac.g5 @ dirac.g1
    Ry = dirac.g5 @ dirac.g2
    Rz = dirac.g5 @ dirac.g3
    refls = [id4, Rz, Ry, Ry@Rz, Rx, Rx@Rz, Rx@Ry, Rx@Ry@Rz]

    # Permutations
    Pxy = dirac.g5 @ (dirac.g1 - dirac.g2) / np.sqrt(2)
    Pyz = dirac.g5 @ (dirac.g2 - dirac.g3) / np.sqrt(2)
    Pzx = dirac.g5 @ (dirac.g3 - dirac.g1) / np.sqrt(2)
    perms = [id4, Pxy, Pyz, Pzx, Pxy @ Pyz, np.linalg.inv(Pxy @ Pyz)]

    # Present group as a flattened tensor product elements
    ohd = np.array(
        [i@r@p for i, r, p in itertools.product(invs, refls, perms)])
    assert len(ohd) == 96, f"Unexpected group size {len(ohd)}"
    return ohd

def make_spinorial_little_group(little):
    """
    Makes the "spinorial" little group associated with the double cover OhD,
    given a little group G in Oh.

    Parameters
    ----------
    little : (|G|, 3, 3) array_like
        The little group G.

    Returns
    -------
    group : (2*|G|, 4, 4) ndarray
        The double-cover little group.
    """
    oh_group = make_oh()
    ohd_group = make_ohd()
    # Compute locations of little-group elements within Oh
    idxs = np.hstack(
        [np.where([np.allclose(gg, g) for gg in oh_group]) for g in little])
    idxs = np.hstack([idxs, idxs+48])
    return ohd_group[idxs].squeeze()


def make_stabilizer(momenta, group):
    """
    Constructs the stabilizer subgroup of an ordered set of momenta.
    The stabilizer group is the subgroup that leaves the ordered set invariant.

    Parameters
    ----------
    momenta : (nmomenta, 3) or (3, ) array_like
        The ordered momenta, with shape.
    group : (|G|, 3, 3) array_like
        The group, with shape.

    Returns
    -------
    stabilizer : (|H|, 3, 3) ndarray
        The stabilizer group H.
    """
    momenta = np.array(momenta)
    if len(momenta.shape) == 1:
        stab = np.array([g for g in group if np.allclose(g@momenta, momenta)])
    else:
        stab = np.array(
            [g for g in group if
                np.allclose(np.einsum("ab,ib->ia", g, momenta),
                            momenta)
            ]
        )
    return stab

def identify_stabilizer(stabilizer):
    """
    Identifies the name of the stabilizer group "H" by checking its order.

    Parameters
    ----------
    stabilizer : (|H|, 3, 3) array_like
        The stabilizer group.

    Returns
    -------
    name : str
        The name of the stabilizer group.
    """
    dim = len(stabilizer)
    if dim == 2:
        check = np.sort([np.trace(np.abs(h)) for h in stabilizer])
        if np.allclose(check, [3, 3]):
            return 'C2R'
        if np.allclose(check, [1, 3]):
            return 'C2P'
        raise ValueError("Unable to identify stabilizer group.")
    dims = {
        48: "Oh",
        8: "C4v",
        6: "C3v",
        4: "C2v",
        # 2: handled separately above for C2P and C2R
        1: "C1",
    }
    return dims[dim]

def make_little_and_stabilizer(momenta, group):
    """
    Computes the little group and stabilizer group of the ordered set of momenta.

    Parameters
    ----------
    momenta : (nmomenta, 3) array_like
        The momenta, with shape
    group : (|G|, 3, 3) array_like
        The group, with shape

    Returns
    -------
    groups : tuple = (ndarray, ndarray)
        A pair of groups arranged as ("little group", "stabilizer group")

    Notes
    -----
    A word on physics naming conventions.
    Let G be a group, let momenta be a set of ordered momenta, and let
    ktot be the total momentum (i.e., the sum of the momenta).
    The "little group" is the subgroup of G that leaves ktot invariant.
    The "stabilizer group" is the subgroup of G that leaves momenta invariant.
    """
    ktot = np.sum(momenta, axis=0)
    assert len(ktot) == 3, "Error: expected 3-vector for total momentum."
    little = make_stabilizer(ktot, group)
    stab = make_stabilizer(momenta, little)
    return (little, stab)


######################
# Orbit construction #
######################

class HashableArray(np.ndarray):
    """
    Minimal wrapper to make arrays hashable based on their value at
    initialization. Basically copied from the docs:
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """
    def __new__(cls, dset):
        obj = np.asarray(dset).view(cls)
        return obj

    def __init__(self, arr):
        self._arr = arr
        self._hash = int(sha256(arr.view(np.uint8)).hexdigest(), 16)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.attrs = getattr(obj, 'attrs', None)

    def __eq__(self, other):
        return np.all(self._arr == other)

    def __hash__(self):
        return self._hash


class MomentumSpinOrbitElement:
    """
    Wrapper for easy manipulations involving elements of momentum-spin orbits.
    """
    def __init__(self, momenta, spins):
        self.momenta = momenta
        self.spins = spins

    def __eq__(self, other):
        return np.allclose(self.momenta, other.momenta) and\
            np.allclose(self.spins, other.spins)

    def __getitem__(self, perm):
        return MomentumSpinOrbitElement(self.momenta[perm,:], self.spins[perm])

    def __str__(self):
        return f"MomentumSpinOrbitElement(momenta={self.momenta}, spins={self.spins})"

    def __repr__(self):
        return self.__str__()


SpinShellTuple = namedtuple("SpinShellTuple", ['momenta', 'spins'])


def make_momentum_orbit(momenta, group, exchange_group=None):
    """
    Computes the orbit of an ordered set of vectors under a group action.

    Parameters
    ----------
    momenta : (nmomenta, 3) array_like
        The ordered momenta.
    group : (|G|, 3, 3) array_like
        The group matrices
    exchange_group : TODO
        <TODO: description of exchange group>

    Returns
    -------
    orbit : list
        The matrices corresponding to ordered momenta in the orbit

    Notes
    -----
    In abstract algebra, an orbit is often considered as an unordered set.
    For numerical work, it is useful to work with orbits ordered in a standard
    way. The convention in the present work is that the orbit inherits its
    order from that of the group elements acting on the original starting
    vector (or ordered list of vectors.) When a given (set of) vector(s) arises
    more than once, the first instance defines the location of the vector(s)
    within the orbit.
    """
    assert momenta.shape[1] == group[0].shape[1], (
        "Incommensurate shape for product g@momenta.T, "
        f"g={group[0].shape} momenta={momenta.shape}")
    if exchange_group is None:
        exchange_group = [np.arange(len(momenta))]
    shell = []
    for group_element in group:
        # Note: g acts on each momentum vector within "momenta".
        momenta_new = np.einsum("ab,ib->ia", group_element, momenta)
        for permutation in exchange_group:
            arr = HashableArray(momenta_new[permutation])
            if arr not in shell:
                shell.append(arr)
    return np.array(shell)


def make_momentum_spin_orbit(momenta, spin_dims, group, exchange=None):
    """
    Computes the momentum-spin orbit, i.e., the tensor product of the
    momentum orbit and the associated spinor orbit.

    Parameters
    ----------
    momenta : (nmomenta, 3) array_like
        The ordered momenta
    spin_dims : list of ints
        The dimensions of the spinor spaces.
    group : (|G|, 3, 3) array_like
        The group matrices.
    exchange : TODO
        <TODO: description of exchange>

    Return
    ------
    spin_shell : ndarray
        The flattened tensor product.

    Note
    ----
    Consider a spinor transforming in an N-dimensional irrep.
    By definition, the different basis vectors for the irrep transform
    into each other under the action of the group.
    Thus, the spinorial part of the shell is just the tensor product
    of all the individual spinor spaces.
    """
    orbit = make_momentum_orbit(momenta, group, exchange)
    spin_space = make_tensor_product_space(spin_dims)
    # Compute flattened tensor product
    spin_shell = np.zeros(len(orbit)*len(spin_space), dtype=object)
    for idx, (mom, spins) in enumerate(itertools.product(orbit, spin_space)):
        spin_shell[idx] = SpinShellTuple(mom, spins)
    return spin_shell


def compute_identical_particle_projector(momentum_spin_orbit,
                                         signed_permutations,
                                         verbose=False):
    """
    Computes the identicical-particle projection matrix.

    Parameters
    ----------
    momentum_spin_orbit : array_like
        <TODO description of the array shape for momentum_spin_orbit>
    signed_permutations : TODO
        <TODO description of signed permutations>
    verbose : bool
        Whether or not to print extra diagnostic information.

    Returns
    -------
    proj : ndarray
        projection matrix <TODO describe expected shape>
    """
    dim = len(momentum_spin_orbit)
    seen = [False] * dim
    proj = np.zeros((dim, dim))

    # Repackage (momenta, spins) elements for later convenience
    momentum_spin_orbit = np.array(
        [MomentumSpinOrbitElement(*elt) for elt in momentum_spin_orbit],
        dtype=object)

    # Compute the eigenvectors
    for idx_initial, initial in enumerate(momentum_spin_orbit):
        if seen[idx_initial]:
            # Skip eigenvectors that have already been computed
            continue

        vec = np.zeros(dim)
        for sign, perm in signed_permutations:
            final = initial[perm]

            # Grab location associated with the "final state"
            for idx_final, elt in enumerate(momentum_spin_orbit):  # for-else
                if final == elt:
                    # Distinct eigenvectors will contain disjoint sets of
                    # nonzero entries. It suffices to construct the eigenvector
                    # containing the jth entry once.
                    seen[idx_final] = True
                    vec[idx_final] += sign
                    break
            else:
                raise ValueError((
                    f"Unable to locate {final.momenta}, {final.spins} within "
                    "the momentum-spin orbit. Please check that the specified "
                    "particles are really identical"
                ))
        if np.linalg.norm(vec) == 0:
            continue
        vec /= np.linalg.norm(vec)
        if verbose:
            print("Found another eigenvector")
        proj += np.einsum("i,j->ij", vec, vec)

    assert np.allclose(proj @ proj, proj), "Error: projector not idempotent"
    return proj


#################################
# Orbit-representation matrices #
#################################


def make_momentum_orbit_rep_matrix(orbit, group_element):
    """
    Computes the momentum-orbit representation matrix associated with the
    action of a group element on an orbit of vectors.

    Parameters
    ----------
    orbit : list
        The orbit of an ordered set of vectors under the group
    group_element : array_like
        The group element which will act on all the elements of the orbit

    Returns
    -------
    permutation : (|O|, |O|) ndarray
        The representation matrix, which happens to be a permutation matrix.

    Notes
    -----
    A group acts on a vector to generate an orbit O. When the group then acts
    on the orbit, the result is a permutation of the vectors in the orbit.
    The permutation can be represented as a square matrix of size |O|x|O|,
    where |O| is the size of the orbit. The full set of these matrices (with
    one for each group element) is itself a group representation.
    """
    dim = len(orbit)
    permutation = np.zeros((dim, dim), dtype=int)
    # Note: the group element acts on each momentum vector within "momenta"
    permuted_orbit =\
        [np.einsum("ab,ib->ia", group_element, momenta) for momenta in orbit]
    for i, vec_i in enumerate(orbit):
        for j, vec_j in enumerate(permuted_orbit):
            permutation[i,j] = int(np.allclose(vec_i-vec_j, 0))
    return permutation


def make_momentum_orbit_rep(orbit, group):
    """
    Computes the representation "Dmm" of a group G acting on an orbit O.
    The representation is the set of matrices "D_{m,m'}(g)".

    Parameters
    ----------
    orbit : list
        The orbit of an ordered set of vectors under the group
    group : (|G|, 3, 3) array_like
        The group matrices.

    Returns
    -------
    representation : (|G|, |O|, |O|) ndarray
        The momentum-representation matrices
    """
    return np.array([make_momentum_orbit_rep_matrix(orbit, g) for g in group])


def make_momentum_spin_rep(Dmm, *Dspin):
    """
    Computes the combined momentum-spin representation matrices.

    Parameters
    ----------
    Dmm : (|G|,|O|,|O|) ndarray
        The momentum-representation matrices.
    *Dspin : (|G^D|, |irrep|, |irrep|) ndarray(s)
        The spin irrep matrices.

    Returns
    -------
    Dmm_spin : (|G^D|, dim_total, dim_total) ndarray
        The combined momomentum-spin representation matrices, where the total
        dimension is given by dim_total = |O|x|irrep1|x|irrep2|x...x|irrepN|.
        As should be expected, the proudct includes all the representations
        appearing in the list of "Dspin" matrices.
    """
    dim_single = Dmm.shape[0]
    dim_double = Dspin[0].shape[0]
    result = []
    for idx in range(dim_double):
        tensors = [Dmm[idx%dim_single]] + [Dmumu[idx] for Dmumu in Dspin]
        result.append(functools.reduce(np.kron, tensors))
    return np.array(result)


##########################
# Bosonic irrep matrices #
##########################


def make_irrep_matrix(polarizations, group_element):
    """
    Computes the irrep matrix associated with a group element using the
    algebraic method in terms of polarization tensors.

    Parameters
    ----------
    polarizations : list
        The polarization tensors specifying the basis functions for the irrep
    group_element : (3, 3) array_like
        The group element.

    Returns
    -------
    irrep_matrix : (|irrep|, |irrep|) ndarray
        The irrep matrix.
    """
    dim = len(polarizations)
    arr = np.zeros((dim, dim), dtype=complex)
    for i, j in itertools.product(range(dim), repeat=2):
        arr[i, j] = tensor_inner(
            polarizations[i],
            transform(polarizations[j], group_element)
        )
    return arr


def make_irrep(polarizations, group):
    """
    Computes the irrep matrices D_{mu, mu'}(g).

    Parameters
    ----------
    polarizations : list
        The polarization tensors specifying the basis functions for the irrep.
    group: (|G|, 3, 3) array_like
        The group matrices.

    Returns
    -------
    irrep : (|G|, |irrep|, |irrep|) ndarray
        The irrep matrices.
    """
    return np.array([make_irrep_matrix(polarizations, g) for g in group])

############################
# Fermionic irrep matrices #
############################

def make_irrep_matrix_spinor(irrep_basis, group_element):
    """
    Computes the spinorial irrep matrix associated with a group element,
    given a basis for the irrep.

    Parameters
    ----------
    irrep_basis : list
        The spinor basis spanning the irrep
    group_element : (4, 4) ndarray
        The "spinorial" group element.

    Returns
    -------
    irrep_matrix : (|irrep|, |irrep|) ndarray
        The irrep matrix.

    Notes
    -----
    Each basis state should be a list of tuples "(coefficient, SpinorTuple)"
    """
    assert group_element.shape == (4,4), "group element should act on spinors"

    # Grab total j, making sure that it is specified consistently for all states
    j = np.unique([psi.j for _, psi in irrep_basis[0]]).item()
    nprod = get_nprod(j)
    cgs = make_spinor_array_lincombo(irrep_basis)
    assert np.allclose(cgs.conj() @ cgs.T, np.eye(len(irrep_basis))),\
        f"Problem with state normalization {cgs.shape} {len(irrep_basis)}"

    # Construct representation D(g) of g acting on the tensor product space
    irrep_mat = functools.reduce(np.kron, nprod*[group_element])

    # Compute matrix of overlaps M_{a,b} <a|D(g)|b>,
    # indexed by the states a, b in the basis of the tensor product space
    return cgs.conj() @ irrep_mat @ cgs.T


def make_irrep_spinor(basis, group):
    """
    Computes the spinorial irrep matrices D_{mu, mu'}(g).

    Parameters
    ----------
    basis : list
        The spinor basis spanning the irrep
    group : (|G|, 4, 4) list or array_like
        The group matrices.

    Returns
    -------
    irrep : (|G|, |irrep|, |irrep|) ndarray
        The irrep matrices.
    """
    return np.array([make_irrep_matrix_spinor(basis, g) for g in group])


def make_irrep_from_group(little_group):
    """
    Computes the irrep matrices associated with a particular little group.

    Parameters
    ----------
    little_group : (|G|, 3, 3) ndarray
        The matrices for the little group G.

    Returns
    -------
    Dmumu : dict
        The irrep matrices as a dict. The keys give the name of the irrep.
        The values contain the irrep matrices themselves, each with shape
        (|G|, |irrep|, |irrep|).
    """
    little_group_name = identify_stabilizer(little_group)

    # Compute normalized polarizations
    basis_arrs = {}
    for irrep_name in basis_functions.basis_fcns[little_group_name]:
        for fcn in basis_functions.basis_fcns[little_group_name][irrep_name]:
            arr = polarize(fcn)
            arr = arr / np.linalg.norm(arr)
            if irrep_name not in basis_arrs:
                basis_arrs[irrep_name] = [arr]
            else:
                basis_arrs[irrep_name].append(arr)

    # Compute irrep matrices
    Dmumu = {}
    for irrep_name, polarizations in basis_arrs.items():
        Dmumu[irrep_name] = make_irrep(polarizations, little_group)
    return Dmumu


def make_irrep_from_groupD(little_group):
    """
    Computes double-cover irrep matrices associated with a given little group.

    Parameters
    ----------
    little_group : (|G|, 3, 3) ndarray
        The matrices for the little group G.

    Returns
    -------
    Dmumu_double : dict
        The irrep matrices as a dict. The keys give the name of the irrep.
        The values contain the irrep matrices themselves, each with shape
        (|G^D|, |irrep|, |irrep|).
    """
    little_name = identify_stabilizer(little_group)
    little_double = make_spinorial_little_group(little_group)

    # Spinorial irreps
    basis_ohd = basis_functions.basis_spinors[little_name]
    Dmumu_double = {}
    for irrep_name, basis in basis_ohd.items():
        Dmumu_double[irrep_name] = make_irrep_spinor(basis, little_double)

    # Bosonic irreps -- double-cover irreps from standard single-cover irreps
    Dmumu_single = make_irrep_from_group(little_group)
    for irrep_name, irrep_mats in Dmumu_single.items():
        Dmumu_double[irrep_name] = np.vstack([irrep_mats, irrep_mats])
    return Dmumu_double


########################################################
# Block diagonalization / change-of-basis coefficients #
########################################################


def compute_lowering_coefficients(irrep_matrices):
    """
    Computes lowering coefficients for moving between rows of an irrep.

    Parameters
    ----------
    irrep_matrices : (|G|, |irrep|, |irrep|) array_like
        The irrep matrices.

    Returns
    -------
    coeffs : (|G|,) np.ndarray
        The lowering coefficients.

    Notes
    -----
    The linear algebra problem is to find "lowering coefficients" c[i] such
    that c[i] D[i,a,b] = L[a,b] (with sum over group index i implied),
    where D[i,a,b] are irrep matrices and L[a,b] is lowering operator. By
    flattening the irrep indices {a,b}, the problem can be cast as a linear
    system of the form A.x=b. This system is generically underdetermined. The
    current implementation singles out a particular solution using the
    Moore-Penrose pseudo-inverse.
    """
    assert irrep_matrices.shape[1] == irrep_matrices.shape[2],\
        "Representation matrices must be square"
    dim_group, dim_rep, _ = irrep_matrices.shape
    irrep_matrices_flat = irrep_matrices.reshape(dim_group, dim_rep**2)

    # Lowering operators
    lower = np.eye(dim_rep, k=-1)
    lower_flat = lower.reshape(dim_rep**2)

    # Solve the flattened problem
    coeffs = np.linalg.pinv(irrep_matrices_flat.T) @ lower_flat
    assert np.allclose(np.einsum("i,iab", coeffs, irrep_matrices), lower),\
        "Failed to construct lowering coefficients."
    return coeffs


def orth(arr):
    """
    Computes an orthonormal basis for the row space.

    This implementation constructs the basis for the row space using the
    Gram-Schmidt algorithm applied to the rows of the input array.

    Parameters
    ----------
    arr : array_like

    Returns
    -------
    basis : ndarray
        An orthonormal basis for the row space, with each row corresponding to
        a basis element.
    """
    basis = []
    for vec in arr:
        if np.isclose(np.linalg.norm(vec), 0):
            continue
        new_vec = project(vec, basis, "perpendicular")
        norm = np.linalg.norm(new_vec)
        if np.isclose(norm, 0):
            continue
        new_vec = new_vec / np.linalg.norm(norm)
        basis.append(new_vec)
    if len(basis) == 0:
        raise ValueError("Failed to construct orthogonal basis")
    return np.array(basis)


def project(vector, basis, direction):
    """
    Computes the paralell or perpendicular projection of a vector with respect
    to the space spanned by a basis.

    Parameters
    ----------
    vector : array_like
    basis : array_like
        Vectors specifying the basis, with each row corresponding to a vector.
    direction : {'parallel', 'perpendicular'}
        Whether to compute the parallel or perpendicular projection.

    Returns
    -------
    vector_new : np.array
        The projected vector.
    """
    nbasis = len(basis)
    if not nbasis:
        # Bail out if the basis is empty
        return np.array(vector)

    if direction not in ('parallel', 'perpendicular'):
        raise ValueError((
            f"Unrecongnized direction '{direction}'."
            "Please specify 'parallel' or 'perpendicular'.")
        )

    vector = np.array(vector)
    basis = np.array(basis)

    # Check sizes
    _, ncols = basis.shape
    dim, = vector.shape
    if ncols != dim:
        raise f"Incomensurate shapes, {basis.shape}, {vector.shape}"

    # Compute projections
    v_parallel = np.zeros(vector.shape)
    for w_vec in basis:
        v_parallel = v_parallel + w_vec*np.dot(np.conjugate(w_vec), vector)
    if direction == 'perpendicular':
        return vector - v_parallel
    return v_parallel  # parallel


def project_basis(Dmm, Dmumu, verbose=False):
    """
    Computes the block diagonalization matrix / change-of-basis coefficients.

    Parameters
    ----------
    Dmm : (|G|, |O|, |O|) array_like
        Momentum-(spin)-representation matrices
    Dmumu : dict
        The irrep matrices. The keys give the name of the irrep. The values
        contain the group irrep matrices, with shape (|G|, |irrep|, |irrep|).
    Returns
    -------
    projector : dict
        The block diagonalization matrix / change-of-basis coefficients.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the coefficients, each of shape
        (|irrep|, |O|).
    """
    projector = {}
    for irrep_name in Dmumu:
        _, dim, _  = Dmumu[irrep_name].shape

        # Apply the Wonderful orthogonality theorem, i.e., Schur's lemma
        # Lowering moves from the top of the irrep down the matrix, so start
        # explicitly with row 0.
        f_projector = np.einsum("i,ijk->jk",
                                Dmumu[irrep_name][:,0,0].conj(),  # mu = 0
                                Dmm)
        if np.all(np.isclose(f_projector, 0)):
            # The present irrep doesn't appear in the decompostion. Carry on.
            continue

        # Construct the lowering operator...
        if dim > 1: # ... but only if the irrep has several rows.
            try:
                coeffs = compute_lowering_coefficients(Dmumu[irrep_name])
            except Exception as err:
                print(f"Failed to locate lowering coefficients: {irrep_name}")
                raise err
            lowering_op = np.einsum("i,iab", coeffs, Dmm)

        # Count the number of degenerate copies
        rank = np.linalg.matrix_rank(f_projector)

        # The following commented-out line works and is more idiomatic.
        # However, the method differs from the Gram-Schmidt algorithm
        # described in the paper.
        # basis = scipy.linalg.orth(f_projector).T
        basis = orth(f_projector.T)
        assert len(basis) == rank, (
            "Failure: expected len(basis)=rank. "
            f"Found len(basis)={len(basis)} and rank={rank}.")
        if verbose:
            print(f"Located irrep {irrep_name} with degeneracy {rank}.")

        # Compute coefficients for each degenerate copy
        for kappa in range(rank):
            vecs = [basis[kappa]]
            # Compute each row using lowering operators
            for _ in range(dim-1):
                new_vec = lowering_op @ vecs[-1]
                if np.isclose(np.linalg.norm(new_vec), 0):
                    raise ValueError("Zero vector encountered while lowering")
                vecs.append(new_vec)
            vecs = rephase(np.array(vecs), irrep_name)
            projector[(irrep_name, kappa)] = vecs

    # Check results for consistency
    test_row_orthogonality(projector, verbose=verbose)
    test_degenerate_orthogonality(projector, verbose=verbose)
    test_block_diagonalization(Dmm, Dmumu, projector, verbose=verbose)
    return projector


def rephase(arr, irrep):
    """
    Applies a phase convention to block diagonalization / change-of-basis
    coefficieints.

    Parameters
    ----------
    arr : (|irrep|, |O|) array_like
        Table specifiying the coefficients.

    Returns
    -------
    arr : (|irrep|, |O|) ndarray
        The table with appropriate phases applied.

    Notes
    -----
    The phase convention is as follows:
    * For a generic irrep, the phase is chosen such that the first nonzero
      entry of the first row ("mu=1") is real and positive.
    * For the irreps T2p and T2m only, the phase is chosen such that the second
      row ("mu=2") is purely imaginary with a negative imaginary part. This
      choice matches the basis-vector conventions of Basak et al., where a
      particular combination of spheric harmonics (Y_2^2 - Y_2^{-2}) is used as
      the "mu=2" basis vector for T2.

    References
    ----------
    [1] S. Basak et al., "Clebsch-Gordan construction of lattice interpolating
        fields for excited baryons", Phys. Rev. D 72, 074501 (2005),
        [arXiv:hep-lat/0508018].
    """
    # Special convention to match Basak et al.
    if irrep in ("T2p", "T2m"):
        vec = arr[1]
        if np.isclose(np.linalg.norm(vec), 0):
            raise ValueError(f"Zero vector found while rephasing {irrep}.")
        idx = np.min(np.nonzero(~np.isclose(vec, 0)))
        phase = np.exp(1j*(-0.5*np.pi - np.angle(vec[idx])))
        tmp = vec[idx] * phase
        phi = np.angle(tmp, deg=True) % 360
        assert phi == 270, f"Bad angle, phi={phi} deg."
        assert np.isclose(np.abs(vec[idx]), np.abs(tmp)), "Bad length."

    # Convention for generic irreps
    else:
        vec = arr[0]
        if np.isclose(np.linalg.norm(vec), 0):
            raise ValueError(f"Zero vector found while rephasing {irrep}.")
        idx = np.min(np.nonzero(~np.isclose(vec, 0)))
        phase = np.exp(-1j*np.angle(vec[idx]))

    return phase * arr


###################
# Driver function #
###################


def mhi(momenta, particles, verbose=False):
    """
    General-purpose driver function for projection.

    Parameters
    ----------
    momenta : (nparticles, 3) or (3, ) array_like
        The ordered momenta, with shape.
    particles : (nparticles, 2) array_like
        The particles specified as rows of the form ['particle_name', 'irrep']

    Returns
    -------
    proj : dict
        The block-diagonalization/change-of-basis matrices
    """
    oh = make_oh()

    little, _ = make_little_and_stabilizer(momenta, oh)
    little_double = make_spinorial_little_group(little)

    Dmumu = make_irrep_from_group(little)
    Dmumu_double = make_irrep_from_groupD(little)

    orbit = make_momentum_orbit(momenta, little)
    Dmm = make_momentum_orbit_rep(orbit, little)

    Dspin = []
    for particle, irrep in particles:
        if particle in ('nucleon', 'proton', 'neutron'):
            if irrep != 'G1p':
                raise ValueError(f"Expected {particle} in G1p irrep, found {irrep}.")
            basis = basis_functions.basis_spinors['nucleon']['nucleon']
            Dspin.append(make_irrep_spinor(basis, little_double))
        elif particle in ('pion', 'pi', 'pi+', 'pi-', 'pi0', 'kaon', 'K', 'K+', 'K-', 'K0'):
            if irrep != 'A1m':
                raise ValueError(f"Expected {particle} in A1m irrep, found {irrep}.")
            Dspin.append(Dmumu_double[irrep])
        else:
            raise ValueError(f"Unexpected particle {particle}")

    # Combine momentum-orbit rep and particle-spin irrep matrices
    Dmm_momspin = make_momentum_spin_rep(Dmm, *Dspin)

    # proj = project_basis(Dmm, Dmumu, verbose=True)
    proj = project_basis(Dmm_momspin, Dmumu_double, verbose)
    return proj



##################
# Test functions #
##################


def test_stabilizer(verbose=False):
    """
    Test identification of the stabilizer group by comparing to known results.
    """
    oh_group = make_oh()
    known_results = {
        "Oh": [0,0,0],
        "C4v": [1,0,0],
        "C3v": [1,1,1],
        "C2v": [1,1,0],
        "C2R": [1,2,0],
        "C2P": [1,1,2],
        "C1": [1,2,3],
    }
    for name, ktot in known_results.items():
        stabilizer = make_stabilizer(ktot, oh_group)
        assert identify_stabilizer(stabilizer) == name,\
            f"Error: Misidentified stabilizer for {ktot}"

    known_results = {
        ("C4v", "C2R"): np.array([[0,1,2], [0,-1,0]]),
        ("C4v", "C2P"): np.array([[1,1,2], [-1,-1,0]]),
    }
    for known_names, momenta in known_results.items():
        little, stab = make_little_and_stabilizer(momenta, oh_group)
        names = identify_stabilizer(little), identify_stabilizer(stab)
        assert names == known_names,\
            f"Error: Misidentified little group and stabilizer for {momenta}"

    if verbose:
        print("Success: reference stabilizer groups identified.")


def anticommutator(arr1, arr2):
    """
    Computes the commutator of two matrices.

    Parameters
    ----------
    arr1 : array_like
    arr2 : array_like

    Returns
    -------
    arr : ndarray
        The commutator [arr1, arr2]
    """
    return arr1 @ arr2 + arr2 @ arr1


def test_clifford(gamma, eta, verbose=False):
    """
    Tests the Clifford-algrebra condition,
    {gamma[mu], gamma[nu]} = 2*Id*eta[mu,nu].

    Parameters
    ----------
    gamma : (4, ) list or (4, 4, 4) array_like
        The gamma matrices gamma[i], i=0,1,2,3.
    eta : (4,4)
        The metric.
    verbose : bool
        Whether to print additional information about successful tests.

    Results
    -------
    None
    """
    id4 = np.eye(4)
    test = np.zeros((4,4), dtype=bool)
    for mu in [0,1,2,3]:
        for nu in [0,1,2,3]:
            # Clifford-algebra condition
            test[mu, nu] = np.all(np.sum(
                anticommutator(gamma[mu], gamma[mu]) == 2*eta[mu,nu]*id4
            ))
    assert np.all(test), "Clifford algebra not satisfied"
    if verbose:
        print("Success: Clifford algebra satisfied.")


def test_gamma5(gamma, gamma5, verbose):
    """
    Tests anticommutation of gamma5, {gamma[mu], gamma5} = 0.

    Parameters
    ----------
    gamma : (4, ) list or (4, 4, 4) array_like
        The gamma matrices gamma[i], i=0,1,2,3.
    gamma5 : (4,4)
        The matrix gamma5
    verbose : bool
        Whether to print additional information about successful tests.

    Results
    -------
    None
    """
    zero4 = np.zeros((4, 4))
    for mu in range(4):
        assert np.all(anticommutator(gamma[mu], gamma5) == zero4),\
            f"gamma_{mu} does not anticommute with gamma5"
    if verbose:
        print("Success: gamma5 anticommutes.")


def test_row_orthogonality(projector, verbose=False):
    """
    Tests the row orthogonality of tables of block-diagonalization /
    change-of-basis coefficients, within each table for a given irrep.

    Parameters
    ----------
    projector : dict
        The block diagonalization matrix / change-of-basis coefficients.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the coefficients, each of shape
        (|irrep|, |O|).
    verbose : bool
        Whether to print additional information about successful tests.

    Returns
    -------
    None
    """
    for irrep_name, kappa in projector:
        arr = projector[(irrep_name, kappa)]
        dim = arr.shape[0]
        assert np.allclose(arr.conj() @ arr.T, np.eye(dim)),\
            f"Failure of row orthogonality {irrep_name} {kappa}"
        if verbose:
            print(f"Success: orthogonal rows for {irrep_name}, kappa={kappa}")


def count_degeneracy(projector):
    """
    Counts the degeneracy of irreps within the tables of block-diagnoalization/
    change-of-basis coefficients.

    Parameters
    ----------
    projector : dict
        The block diagonalization matrix / change-of-basis coefficients.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the coefficients, each of shape
        (|irrep|, |O|).

    Returns
    -------
    degeneracy : dict
        The degeneracies, with keys corresponding to the irrep names and values
        to the counts.
    """
    degeneracy = {}
    for irrep_name, _ in projector:
        if irrep_name in degeneracy:
            degeneracy[irrep_name] += 1
        else:
            degeneracy[irrep_name] = 1
    return degeneracy


def test_degenerate_orthogonality(projector, verbose=False):
    """
    Tests that tables corresponding to degenerate irreps are orthogonal.

    Parameters
    ----------
    projector : dict
        The block diagonalization matrix / change-of-basis coefficients.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the coefficients, each of shape
        (|irrep|, |O|).
    verbose : bool
        Whether to print additional information about successful tests.

    Returns
    -------
    None
    """
    degeneracy = count_degeneracy(projector)
    for irrep_name, rank in degeneracy.items():
        if rank == 1:
            continue
        for k1, k2 in itertools.product(range(rank), repeat=2):
            if k1 == k2:
                continue
            arr1 = projector[(irrep_name, k1)]
            arr2 = projector[(irrep_name, k2)]
            dim = arr1.shape[0]
            assert np.allclose(arr1.conj() @ arr2.T, np.zeros(dim)),\
                f"Failure of orthogonality {irrep_name} {k1} {k2}"
    if verbose:
        print("Success: all irreps orthogonal")


def test_block_diagonalization(Dmm, Dmumu, projector, verbose=False):
    """
    Tests the block diagonalization property of the change-of-basis
    projection matrices which, schematically, reads: "Dmumu = F^* Dmm F"
    in terms of the momentum-shell representation "Dmm", the projector "F",
    and the block-diagonal irrep matrix "Dmumu".

    Parameters
    ----------
    Dmm : (|G|,|O|,|O|) ndarray
        The momentum-representation matrices.
    Dmumu : dict
        The irrep matrices as a dict. The keys give the name of the irrep.
        The values contain the irrep matrices themselves, each with shape
        (|G|, |irrep|, |irrep|).
    projector : dict
        The block diagonalization matrix / change-of-basis coefficients.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the coefficients, each of shape
        (|irrep|, |O|).
    verbose : bool
        Whether to print additional information about successful tests.

    Returns
    -------
    None
    """
    for (irrep, _), F in projector.items():
        test = np.allclose(
            Dmumu[irrep],
            np.einsum("ia,rab,jb->rij", F.conj(), Dmm, F))
        assert test, f"Failure: block diagonalization {irrep}"
    if verbose:
        print("Success: block diagonalization using projection matrices")


def write_hdf5(h5fname, result_dict):
    """
    Writes block-diagonalization / change-of-basis coefficients to HDF5.

    Parameters
    ----------
    h5fname : str
        The name of the output file
    result_dict : dict
        The block-diagonalization / change of basis matrices, where each key
        is a tuple the form (irrep_name, degeneracy) and each key is an ndarray
        with the coefficients.

    Returns
    -------
    None

    """
    with h5py.File(h5fname, mode='a') as h5file:
        for (irrep_name, degeneracy), data in result_dict.items():
            dpath = f'data/{irrep_name}/{degeneracy}'
            if dpath not in h5file:
                _ = h5file.create_dataset(name=dpath, data=data,
                                          compression='gzip', shuffle=True)
            else:
                existing = h5file[dpath]  # load existing data
                existing[...] = data      # assigned new values to data
                assert np.allclose(h5file[dpath], data),\
                    f"Error updated values for ({irrep_name}, {degeneracy})."
            h5file.flush()


def read_hdf5(h5fname):
    """
    Reads saved block-diagonalization / change-of-basis coefficients from HDF5.

    Parameters
    ----------
    h5fname : str
        The name of HDF5 file to read

    Returns
    -------
    result_dict : dict
        The block-diagonalization / change of basis matrices, where each key
        is a tuple the form (irrep_name, degeneracy) and each key is an ndarray
        with the coefficients.

    """
    result_dict = {}
    with h5py.File(h5fname, 'r') as ifile:
        for irrep_name in ifile['data']:
            for degeneracy in ifile['data'][irrep_name]:
                print(irrep_name, degeneracy)
                result_dict[(irrep_name, int(degeneracy))] =\
                    np.array(ifile['data'][irrep_name][degeneracy][:])
    return result_dict
