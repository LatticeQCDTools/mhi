"""
MHI -- "Multi-Hadron Interpolators"

Module for constructing block-diagonalization / change-of-basis matrices to map
products of N local plane-wave operators into irreps of the cubic group.
Includes appropriate generalizations for spin and internal symmetries.
"""

import functools
import itertools
from collections import namedtuple
from hashlib import sha256
import re
import sys
import os
import pathlib
import string
from operator import itemgetter
import numpy as np
import sympy
from scipy.linalg import expm
import yaml
import h5py
from . import basis_functions

class WeightedPermutation(namedtuple('WeightedPermutation', ['weight', 'perm'])):
    """A complex scalar weight times a permutation group element.

    Attributes
    ----------
    weight : complex or float
        Scalar weight multiplying the permutation.
    perm : ``(n,)`` ndarray
        Permutation expressed as an array.
    """
    # def __init__(self, weight, perm):
    #     self.weight = weight
    #     self.perm = perm
class Isomorphism(namedtuple('Isomorphism', ['g', 'perm'])):
    """The group element and permutation specifying a subgroup isomorphism.

    Attributes
    ----------
    g : ``(n, n)`` ndarray
        Group element used to conjugate the subgroup as
        :math:`g \cdot H \cdot g^{-1}`
    perm : ``(|H|,)`` ndarray
        Permutation p mapping from conjugated elements to target subgroup as
        :math:`H' = g \cdot H \cdot g^{-1}[p]`
    """
    # def __init__(self, g, perm):
    #     self.g = g
    #     self.perm = perm
class SpinShellTuple(namedtuple('SpinShellTuple', ['momenta', 'spins'])):
    """Pairing of momenta and spins making up an orbit with non-zero spins.

    Attributes
    ----------
    momenta : ``(norbit, nmomenta, 3)`` ndarray
        List of momentum lists in the orbit.
    spins : ``(norbit, nspin)`` ndarray
        List of spin configurations in the orbit.
    """
    # def __init__(self, momenta, spins):
    #     self.momenta = momenta
    #     self.spins = spins

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
    r"""
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
    :math:`T_{ij...n} \to T_{(ij...n)}`,
    where parantheses denote symmetrization.
    For instance, the symmetrization of a three-index tensor
    :math:`T_{ijk}`
    is
    :math:`T_{(ijk)} = \tfrac{1}{6}(T_{ijk} + T_{ikj} + T_{jik} + T_{jki} + T_{kij} + T_{kji})`
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
    r"""
    Computes the tensor product between tensors a and b.

    Parameters
    ----------
    a, b : array_like

    Returns
    -------
    tensor : array_like
        The tensor product of arrays `a` and `b`.

    Notes
    -----
    In index notation, this fuction computes the output tensor T defined by
    :math:`T_{ij \dots k rs \dots t} = a_{ij \dots k} b_{rs \dots t}`.
    """
    return np.tensordot(a, b, axes=0)


def tensor_nfold(*tensors):
    r"""
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
    Suppose the input tensors are
    :math:`\{a_i, b_{jk}, c_{lmn}\}`.
    This function computes the output tensor T defined by
    :math:`T_{ijklmn} = a_i b_{jk} c_{lmn}`.
    """
    return functools.reduce(tensor_product, tensors)


def decompose(monomial):
    """
    Decomposes a sympy monomial of the form
    ``c * x**nx * y**ny * z**nz``
    into a coefficient `c` and a triplet of exponents ``(nx, ny, nz)``.

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
    arr : ``(M, M, ..., M)`` array_like
        The tensor to transform.
    group_element : ``(M, M)`` array_like
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
    a : ``(M, M, ... , M)`` array_like
    b : ``(M, M, ... , M)`` array_like

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
    (6, 3)
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

    def rotation_vec(self, omega):
        """
        Computes the spinor rotation matrix associated with the 3-vector omega.

        Parameters
        ----------
        omega: ``(3,)`` ndarray or list
            The vector specifying the rotation

        Returns
        -------
        arr : ``(4, 4)`` ndarray
            The complex roation matrix
        """
        assert len(omega) == 3, "omega must be a 3-vector"
        g = [self.g1, self.g2, self.g3]
        eps = self.eps3
        arg = np.zeros(g[0].shape, dtype=complex)
        for i,j,k in itertools.product(range(3), repeat=3):
            arg += eps[i,j,k] * g[i] @ g[j] * omega[k]
        arg *= -0.25
        return expm(arg)

    def rotation(self, theta, direction):
        """
        Computes the spinor rotation matrix associated with an rotation
        of angle "theta" around the kth axis.

        Parameters
        ----------
        theta: float
            The rotation angle
        direction: int
            The axis number, with {1,2,3} <--> {x,y,z}.

        Returns
        -------
        arr : ``(4, 4)`` ndarray
            The complex rotation matrix
        """
        assert direction in (1,2,3), "Please specify direction in (1,2,3)."
        omega = np.zeros(3)
        omega[direction -1] = theta
        return self.rotation_vec(omega)


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
    group : ``(48, 3, 3)`` ndarray
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


def rotation(theta, direction):
    """
    Computes the rotation matrix by the angle theta around a particular axis.

    Parameters
    ----------
    theta : float
        The angle
    direction: int
        The axis number, with {1,2,3} <--> {x,y,z}.


    Returns
    -------
    arr : ``(3, 3)`` ndarray
        The rotation matrix
    """
    assert direction in (1,2,3), "Please specify direction in (1,2,3)."
    omega = np.zeros(3)
    omega[direction-1] = theta
    return rotation_vec(omega)


def rotation_vec(omega):
    """
    Computes the rotation matrix associated with the three-vector omega.

    Parameters
    ----------
    omega : ``(3, )`` ndarray or list
        The vector defining the rotation

    Returns
    -------
    arr : ``(3, 3)`` ndarray
        The rotation matrix
    """
    assert len(omega) == 3, "omega must be a 3-vector"
    eps = levi_civita(3)
    return expm(-eps @ omega)


def make_ohd():
    """
    Constructs a presentation of the "spinorial" double cover OhD of
    the cubic group ordering of group elements.

    Parameters
    ----------
    None

    Returns
    -------
    group : ``(96, 4, 4)`` ndarray
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
    little : ``(|G|, 3, 3)`` array_like
        The little group G.

    Returns
    -------
    group : ``(2*|G|, 4, 4)`` ndarray
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
    momenta : ``(nmomenta, 3)`` or ``(3, )`` array_like
        The ordered set of momenta that must be left invariant.
    group : ``(|G|, 3, 3)`` array_like
        The total group.

    Returns
    -------
    stabilizer : ``(|H|, 3, 3)`` ndarray
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


def make_canonical_stabilizer(name, group):
    """
    Computes the stabilizer subgroup with a canonical orientation inside the
    larger group Oh.

    Parameters
    ----------
    name : str
        The name of the desired stabilizer subgroup
    group : ``(|G|, 3, 3)`` array_like
        The group

    Returns
    -------
    stabilizer : ``(|H|, 3, 3)`` array_like
        The stabilizer subgroup
    """
    canonical_momenta = {
        "Oh": np.array([0,0,0]),
        "C4v": np.array([0,0,1]),
        "C3v": np.array([1,1,1]),
        "C2v": np.array([0,1,1]),
        "C2R": np.array([1,2,0]),
        "C2P": np.array([1,1,2]),
        "C1": np.array([1,2,3]),
    }
    ktot = canonical_momenta[name]
    return make_stabilizer(ktot, group)


def identify_stabilizer(stabilizer):
    """
    Identifies the name of the stabilizer group "H" by checking its order.

    Parameters
    ----------
    stabilizer : ``(|H|, 3, 3)`` array_like
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
    momenta : ``(nmomenta, 3)`` array_like
        The momenta
    group : ``(|G|, 3, 3)`` array_like
        The group

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


def conjugate(g, h):
    """
    Computes the conjugate of group element h by group element g:
    :math:`g.h.g^{-1}`.  Assumes that g is an orthogonal matrix so that
    :math:`g^{-1} = g^T`.

    Parameters
    ----------
    g : ``(n, n)`` ndarray
    h : ``(n, n)`` ndarray

    Returns
    -------
    h_conj : ``(n, n)`` ndarray
        The conjugated element, :math:`g.h.g^{-1}`
    """
    return g @ h @ g.T


def conjugate_group(g, group):
    """
    Computes the conjugate of the group G by the group element g: :math:`g.G.g^{-1}`.
    Assumes that g is an orthogonal matrix so that :math:`g^{-1} = g^T`.

    Parameters
    ----------
    group : ``(|G|, n, n)`` ndarray
        The group G.
    h : ``(n, n)`` ndarray
        The conjugating element.

    Returns
    -------
    group_conj : ``(|G|, n, n)`` ndarray
        The conjugated group :math:`g.G.g^{-1}`
    """
    return np.array([conjugate(g, elt) for elt in group])


def find_subgroup_isomorphism(group, subgroup_h1 , subgroup_h2):
    """
    Finds the isomorphism between conjugate subgroups H1 and H2.

    Parameters
    ----------
    group : ``(|G|, n, n)`` ndarray
        The group G.
    subgroup_h1 : ``(|H|, n, n)`` ndarray
        The subgroup H1.
    subgroup_h2 : ``(|H|, n, n)`` ndarray
        The subgroup H2.

    Returns
    -------
    (g, perm) : :class:`Isomorphism`
        The group element and permutation specifying the isomorphism.
    """
    assert subgroup_h1.shape == subgroup_h2.shape,\
        f"Incomensurate shapes for subgroups H1 and H2: {subgroup_h1.shape} {subgroup_h2.shape}"
    assert group.shape[1:] == subgroup_h1.shape[1:],\
        f"Incomensurate shapes for group G and the subgroup H1: {group.shape} {subgroup_h1.shape}"

    def argsort(group):
        """
        Computes the indices that would sort the group, using a hash
        """
        return np.argsort([force_hash(g) for g in group])

    ridx2 = argsort(subgroup_h2).argsort()
    for g in group:
        h1_conj = conjugate_group(g, subgroup_h1)
        idx1 = argsort(h1_conj)
        perm = idx1[ridx2]
        if np.allclose(h1_conj[perm], subgroup_h2):
            break
    else:
        raise ValueError("Failed to locate an isomorphism.")
    return Isomorphism(g, perm)


def apply_isomorphism(group, isomorphism):
    """
    Applies the isomorphism (g, perm) to the group G via :math:`(g.G.g^{-1})[perm]`.

    Parameters
    ----------
    group : ``(|G|, n, n)`` ndarray
    isomorphism : :class:`Isomorphism`

    Returns
    -------
    group_iso : ``(|G|, n, n)`` ndarray
        The group after applying the isomorphism, :math:`(g.G.g^{-1})[perm]`
    """
    return conjugate_group(isomorphism.g, group)[isomorphism.perm]


######################
# Orbit construction #
######################

def force_hash(arr):
    """
    Computes a hash for an array.
    """
    return int(sha256(arr.view(np.uint8)).hexdigest(), 16)


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
        self._hash = force_hash(arr)

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


def make_momentum_orbit(momenta, group, exchange_group=None):
    """
    Computes the orbit of an ordered set of vectors under a group action.

    Parameters
    ----------
    momenta : ``(nmomenta, 3)`` array_like
        The ordered momenta.
    group : ``(|G|, 3, 3)`` array_like
        The group matrices.
    exchange_group : list of :class:`WeightedPermutation`
        The exchange group projector.

    Returns
    -------
    orbit : list
        The matrices corresponding to ordered momenta in the orbit.

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
        exchange_group = [WeightedPermutation(None, np.arange(len(momenta)))]

    # Create the shell
    shell = []
    for group_element in group:
        # Note: g acts on each momentum vector within "momenta".
        momenta_new = np.einsum("ab,ib->ia", group_element, momenta)
        arr = HashableArray(momenta_new)
        if arr not in shell:
            shell.append(arr)

    # Apply the exchange group
    # Note: These separate loops for creating the shell and apply the exchange
    # group could be combined at the price of changing the ordering of the shell.
    for exchange_element in exchange_group:
        for arr in shell[:]:
            arr_new = HashableArray(np.array(arr[exchange_element.perm]))
            if arr_new not in shell:
                shell.append(arr_new)

    return np.array(shell)


def make_momentum_spin_orbit(momenta, spin_dims, group, exchange_group=None):
    """
    Computes the momentum-spin orbit, i.e., the tensor product of the
    momentum orbit and the associated spinor orbit.

    Parameters
    ----------
    momenta : ``(nmomenta, 3)`` array_like
        The ordered momenta
    spin_dims : list of ints
        The dimensions of the spinor spaces.
    group : ``(|G|, 3, 3)`` array_like
        The group matrices.
    exchange : array_like or None
        The exchange_group with namedtuple/WeightedPermutation elements

    Returns
    -------
    spin_shell : ndarray
        The flattened tensor product.

    Notes
    -----
    Consider a spinor transforming in an N-dimensional irrep.
    By definition, the different basis vectors for the irrep transform
    into each other under the action of the group.
    Thus, the spinorial part of the shell is just the tensor product
    of all the individual spinor spaces.
    """
    orbit = make_momentum_orbit(momenta, group, exchange_group)
    spin_space = make_tensor_product_space(spin_dims)
    # Compute flattened tensor product
    spin_shell = np.zeros(len(orbit)*len(spin_space), dtype=object)
    for idx, (mom, spins) in enumerate(itertools.product(orbit, spin_space)):
        spin_shell[idx] = SpinShellTuple(mom, spins)
    return spin_shell


def parity(permutation):
    """
    Computes the parity of permutation, assumed to be specified as a list of
    contiguous integers.

    Parameters
    ----------
    permutation: array_like
        The permutation

    Returns
    -------
    sign: +1 or -1
        The parity.

    Examples
    --------
    >>> parity([2,3,4,5])
    1
    >>> parity([5,2,3,4])
    -1
    """
    permutation = np.array(permutation)
    permutation = permutation - min(permutation)  # Work with respect to zero
    if not np.all(np.sort(permutation) == range(len(permutation))):
        raise ValueError(f"Non-contiguous integers: {permutation}")
    sign = 1
    for i in range(0, len(permutation)-1):
        if permutation[i] != i:
            sign *= -1
            mn = min(range(i, len(permutation)), key=permutation.__getitem__)
            permutation[i], permutation[mn] = permutation[mn], permutation[i]
    return sign


def partition(arr):
    """
    Computes the partitions of the array according to the unique elements.

    Parameters
    ----------
    arr : array_like
        The array to partition

    Returns
    -------
    partitions : dict
        The partitions, where the keys are the unique entries of the input array
        and the values are the associated indices in the input array.

    Notes
    -----
    The keys are sorted according to the first appearance in arr

    Examples
    --------
    >>> partition(['a','b','b','c']))
    {'a': array([0]), 'b': array([1, 2]), 'c': array([3])}
    """
    arr = np.asarray(arr)
    tmp = {elt: np.nonzero(arr == elt)[0] for elt in np.unique(arr)}
    # Sort keys according to their first appearance
    keys, _ = zip(*sorted([(k, min(idxs)) for k, idxs in tmp.items()], key=itemgetter(1)))
    return {key: tmp[key] for key in keys}


def recombine(labels, partition_keys, partition_idxs):
    """
    Recombines indices from partitioned sets.
    This function is equivalent to concatenating the indices for contiguous labels.
    For instance, ['b1', 'b1', 'b2] are contiguous, but ['b1', 'b2', 'b1'] are not.

    Parameters
    ----------
    labels : ``(n, )`` array_like
        The labels, e.g., ['b1', 'b2', 'b1']
    partition_keys : ``(m, )`` array_like
        The unique elements of the labels, ordered by first appearance.
    partition_idxs : ``(m, )`` tuple of array_like
        The indices to recombine, in the same order as the partition_keys

    Returns
    -------
    idxs : ``(n, )`` ndarray
        The recombined indices

    Examples
    --------
    This example builds the tensor product of permutations on the set of labels
    ['b1','b2','b1']. Since the label 'b1' is not contiguous, the permutations
    associated with this label should include the identity (0)(2) and the swap
    (0,2).

    >>> labels = ['b1','b2','b1']
    >>> partitions = partition(labels)
    >>> perms = {key: list(itertools.permutations(idxs)) for key, idxs in partitions.items()}
    >>> keys = np.array(list(partitions.keys()))
    >>> for idxs in itertools.product(*perms.values()):
    >>>     print(mhi.recombine(labels, keys, idxs), "(correct)")
    >>>     print(np.concatenate(idxs), "(wrong)")
    [0 1 2] (correct)
    [0 2 1] (wrong)
    [2 1 0] (correct)
    [2 0 1] (wrong)
    """
    assert len(partition_keys) == len(partition_idxs),\
        "Incommensurate partition_keys and indices"
    assert len(np.concatenate(partition_idxs)) == len(labels),\
        "Incommensurate labels and indices."
    idxs = np.zeros(len(labels), dtype=int)
    iterators_idxs = [iter(elt) for elt in partition_idxs]
    for i, label in enumerate(labels):
        j = np.argwhere(partition_keys == label).item()
        idxs[i] = next(iterators_idxs[j])
    return idxs


def make_exchange_projector(labels, tableau_map):
    """
    Computes a projector in the group algebra of the particle-exchange group.

    Parameters
    ----------
    labels : list
        Particle labels, e.g., ['a','b','c','b','c','b']
    tableau_map : dict
        Young tableaux associated with each label, e.g.,
        {'a': [[1]], 'b': [[1,2],[3]], 'c': [[1],[2]],}

    Returns
    -------
    projector : list of WeightedPermutation objects
        The projection operator in the group algebra of the exchange group
    """
    partitions = partition(labels)
    for key in partitions:
        if key not in tableau_map:
            raise KeyError(f"Key '{key}' missing from tableau_map.")
        if not is_valid_tableau(tableau_map[key]):
            raise ValueError(f"Invalid Young tableau for key '{key}'.")
        if len(partitions[key]) != len(np.concatenate(tableau_map[key])):
            raise ValueError(f"Incorrect tableau size for key '{key}'.")

    def remap_indices(idxs, wp):
        return WeightedPermutation(wp.weight, idxs[wp.perm])

    # Compute individual projectors associated with each label
    projectors = {}
    for key in partitions:
        idxs = partitions[key]
        # Compute Young projector on contiguous indices 1,2,...,len(idxs)
        proj_tmp =  make_young_projector(tableau_map[key], n=len(idxs))
        # Remap contiguous indices to the indices of the label
        projectors[key] = [remap_indices(idxs, wp) for wp in proj_tmp]

    # Compute tensor product of the projectors from each label
    projector = []
    keys = np.array(list(partitions.keys()))
    for wps in itertools.product(*projectors.values()):
        idxs = [wp.perm for wp in wps]
        weight = np.product([wp.weight for wp in wps])
        projector.append(
            WeightedPermutation(
                weight=weight,
                perm=recombine(labels, keys, idxs)))

    assert algebra_elements_are_close(
        projector,
        compose_permutation_algebra_elements(projector, projector)),\
        "Error: Projector not idempotent!"
    return projector


def make_exchange_projector_identical(labels):
    """
    Computes the exchange group associated with exchange of identical particles.
    The elements of this group are signed permuations.

    Parameters
    ----------
    labels : array_like or None
        The labels associated with the particles.
    fermions : dict
        Whether a given key (label) corresponds to a fermion.
        Values must be booleans: True for fermions and False for bosons.

    Returns
    -------
    exchange_group : list or None
        The exchange group, with namedtuple/WeightedPermutation elements.
        The result is None when no labels are specified.

    Notes
    -----
    Generic fermions are labeled 'f{n}' for integers n, e.g., 'f1', 'f2, ...
    Generic bosons are labeld 'b{n}' for integers n, e.g., 'b1', 'b2', ...
    Support also exists for certain named particles like 'pi', 'K', 'neutron',
    'proton', 'nucleon'.
    """
    if (labels is None) or (len(labels) == 0):
        return None  # No labels specified

    table = load_particle_info()
    fermions = {key : value.is_fermion for key, value in table.items()}
    fermion_template = re.compile(r'(f|F)(|-?|\d+)')  # generic fermion, e.g., f1
    boson_template = re.compile((r'(b|B)(|-?|\d+)'))  # generic boson, e.g., b1

    # Verify inputs
    for label in np.unique(labels):
        if fermion_template.match(label):
            fermions[label] = True
        elif boson_template.match(label):
            fermions[label] = False
        assert label in fermions, f"Missing fermion/boson specification for {label}"
    for key, val in fermions.items():
        assert isinstance(val, bool), f"Bad specification for {key}: {val}. Expected a bool."

    # Decide which particles have the same labels
    partitions = partition(labels)
    keys = np.array(list(partitions.keys()))
    # Permutations associated with individual labels, ignoring signs
    perms = {key: list(itertools.permutations(idxs)) for key, idxs in partitions.items()}

    # parity**0 = 1 for bosons
    # parity**1 = parity for fermions
    powers = [int(fermions[key]) for key in perms.keys()]

    # The full list of permutations is the tensor product of the individual permutations
    exchange_group = []
    for idxs in itertools.product(*perms.values()):
        # Overall sign = product of the parities of the individual permuations
        sign = np.prod([
            parity(np.argsort(perm))**power for perm, power in zip(idxs, powers)
        ])
        perm = recombine(labels, keys, idxs)
        exchange_group.append(WeightedPermutation(weight=sign, perm=perm))
    return exchange_group


def make_internal_symmetry_projector(orbit, internal_symmetry):
    r"""
    Computes the projection matrix associated with an internal symmetry group.

    Parameters
    ----------
    momentum_spin_orbit : ``(n, )`` list of :class:`SpinShellTuple`
        Each element in the list defines the (momenta, spin) configuration for a
        given element of the orbit.
    internal_group : list of :class:`WeightedPermutation`
         The exchange group projector in the group algebra.

    Returns
    -------
    proj : ``(n, n)`` ndarray
        projection matrix P, intended to be contracted against the extended
        momentum-spin representation matrices, giving projected matrices
        :math:`\hat{D}_{mm'}(R) = P_{mn} D_{nn'}(R) P_{nm'}`

    Notes
    -----
    As a projection matrix, "proj" is idempotent: ``proj @ proj == proj``.

    The full internal symmetry operator can be thought of as a linear
    combination of permutations. The representation matrices can be computed
    either before or after taking the linear combination. This implementation
    takes the latter route, computing a representation matrix for each and
    taking a suitable linear combination.
    """
    has_spins = False
    if hasattr(orbit[0], 'spins'):
        has_spins = True
    proj = np.zeros((len(orbit), len(orbit)), dtype=float)
    for weight, perm in internal_symmetry:
        # Permute the orbit
        if has_spins:
            orbit_permuted = [SpinShellTuple(momenta[perm], spins[perm]) for momenta, spins in orbit]
        else:
            orbit_permuted = [momenta[perm] for momenta in orbit]
        # Compute the matrix elements of the operator acting on the shell representation
        tmp = np.zeros((len(orbit), len(orbit)), dtype=float)
        for i, new in enumerate(orbit_permuted):
            for j, old in enumerate(orbit):
                if has_spins:
                    if np.allclose(new.momenta, old.momenta) and np.allclose(new.spins, old.spins):
                        tmp[i,j] = 1
                else:
                    if np.allclose(new, old):
                        tmp[i,j] = 1.0
        proj += float(weight) * tmp

    # Projection matrices have eigenvalues [0, 1].
    evals = np.linalg.eigvals(proj)
    norm = evals[np.argmax(np.abs(evals))]
    if np.isclose(norm, 0):
        return None  # Orbit is removed by the projector
    proj = proj / norm
    assert np.allclose(proj @ proj, proj), "Error: projector not idempotent"
    return proj


def multiply_perms(a, b):
    """
    Multiply the permutations a and b.

    Group multiplication acts as composition when considering group elements as
    left-acting operators, i.e.,
    :math:`(a \cdot b)(v) = (a \circ b)(v) = a(b(v))`.

    Parameters
    ----------
    a, b : ``(n,)`` ndarray

    Returns
    -------
    ndarray
        The product of permutations.
    """
    return b[a]


def invert_perm(a):
    """
    Compute the inverse permutation such that
    :math:`a \circ a^{-1} = a^{-1} \circ a = (1)`

    Parameters
    ----------
    a : ``(n,)`` array_like

    Returns
    -------
    ainv : ``(n,)`` ndarray
        The inverse permutation.
    """
    return np.argsort(a)


def compose_permutation_algebra_elements(a, b):
    """
    Compute the product of two elements in the algebra of the permutation group.

    Parameters
    ----------
    a, b : list of :class:`WeightedPermutation`
        The algebra elements to be multiplied.

    Returns
    -------
    list
        The product, as a list of :class:`WeightedPermutation` objects.
    """
    out = {}
    for w_perm1, w_perm2 in itertools.product(a, b):
        w = w_perm1.weight * w_perm2.weight
        perm = multiply_perms(w_perm1.perm, w_perm2.perm)
        key = tuple(perm)
        if key not in out:
            out[key] = WeightedPermutation(weight=0.0, perm=perm)
        out[key] = WeightedPermutation(weight=out[key].weight+w, perm=out[key].perm)
    return list(out.values())


def algebra_elements_are_close(x1, x2):
    """
    Checks whether elements of the group algebra of Sn are numerically close.

    Parameters
    ----------
    x1, x2 : list of :class:`WeightedPermutation` objects
        The group algebra elements.

    Returns
    -------
    are_close : bool
    """
    if len(x1) != len(x2):return False
    inds1 = np.lexsort(np.stack([wp.perm for wp in x1], axis=-1))
    inds2 = np.lexsort(np.stack([wp.perm for wp in x2], axis=-1))
    for i in range(len(x1)):
        wp1 = x1[inds1[i]]
        wp2 = x2[inds2[i]]
        if not np.all(wp1.perm == wp2.perm):
            return False
        if not np.isclose(wp1.weight, wp2.weight):
            return False
    return True


def is_valid_tableau(tableau):
    """
    Checks whether ragged list is a valid Young tableau.

    Parameters
    ----------
    tableau : list of lists
        The candidate Young tableau.

    Returns
    -------
    is_valid : bool
    """
    # Row lengths never increase
    if np.any(np.diff([len(row) for row in tableau]) > 0):
        return False
    # Values in rows strictly increasing
    if np.any(np.concatenate([np.diff(row) for row in tableau]) <= 0):
        return False
    # Values in columns strictly increasing
    tableau_t = transpose_tableau(tableau)
    if np.any(np.concatenate([np.diff(row) for row in tableau_t]) <= 0):
        return False
    return True


def transpose_tableau(tableau):
    """
    Transposes a Young tableau.

    Parameters
    ----------
    tableau : list of lists
        The tableau, given as a ragged list of integers.

    Returns
    -------
    tableau_t : list of lists
        The transposed tableau, as a ragged list of integers.
    """
    tableau_t = [[] for _ in range(len(tableau[0]))]
    for row in tableau:
        for i,elt in enumerate(row):
            tableau_t[i].append(elt)
    return tableau_t


def symmetrizer(row, *, signed, n):
    """Computes a symmetrizer over the specified indices.

    The symmetrizer is an element of the algebra of the group Sn that yields the
    symmetrized or anti-symmetrized (depending on `signed`) space over the
    indices in `row`.

    Parameters
    ----------
    row : array_like
        The (1-indexed) indices over which to symmetrize.
    signed : 0 or 1
        Whether to symmetrize (0) or antisymmetrize (1)
    n : The total number of indices

    Returns
    -------
    list
        The symmetrizer in the group algebra, represented as a list of
        WeightedPermutation objects.

    Notes
    -----
    Indices less than or equal to 1 are taken from the back of the list.

    Examples
    --------
    >>> mhi.symmetrizer([1,2], signed=1, n=3)
    [WeightedPermutation(weight=1, perm=array([0, 1, 2])),
     WeightedPermutation(weight=-1, perm=array([1, 0, 2]))]
    >>> mhi.symmetrizer([2,3], signed=0, n=3)
    [WeightedPermutation(weight=1, perm=array([0, 1, 2])),
     WeightedPermutation(weight=1, perm=array([1, 0, 2]))]
    >>> mhi.symmetrizer([0,-1], signed=0, n=3)
    [WeightedPermutation(weight=1, perm=array([0, 1, 2])),
     WeightedPermutation(weight=1, perm=array([0, 2, 1]))]
    """
    assert signed in [0,1]
    row = np.array(row)
    out = []
    for perm in itertools.permutations(tuple(range(len(row)))):
        perm = np.array(perm)
        sign = parity(perm)
        w = sign**signed
        tot_perm = np.arange(n)
        tot_perm[row-1] = tot_perm[row[perm]-1]
        out.append(WeightedPermutation(weight=w, perm=tot_perm))
    return out


def make_young_projector(tableau, *, n):
    """
    Makes the projection operator, acting on n indices, associated with the
    given Young tableau.

    Parameters
    ----------
    tableau : list of lists
        The Young tableau.
    n : int
        The number of indices for which to construct the projector.

    Returns
    -------
    proj : list of WeightedPermutation objects
        The projection operator in the group algebra.

    Notes
    -----
    Young tableaux are specified by indices 1,2,3,.... starting with 1.
    Indices in python arrays are zero indexed.
    This implementation uses definition of the Hermitian projection operators
    given by Ref. [1]_. In particular, this function uses Eq (86) in Theorem 3
    (KS Hermitian Young projectors). For large tableaux, it would likely
    advantageous to switch to the Measure Of Lexical Disorder (MOLD) definition
    of the projection operators given in Theorem 5.


    References
    ----------
    .. [1] J. Alcock-Zeilinger and H. Weigert
       "Compact Hermitian Young Projection Operators"
       J.Math.Phys. 58 (2017) 5, 051702
       [arXiv:1610.10088].
    """
    if not is_valid_tableau(tableau):
        raise ValueError("Invalid tableau")
    if n < len(np.concatenate(tableau)):
        raise ValueError("Too few indices for specified tableau.")

    # Base case for the recursive definition
    if tableau == [[1]]:
        return [WeightedPermutation(weight=1, perm=np.arange(n))]

    # Apply the "parent map" of Definition 1 in Eqs (37) and (38).
    # This map removes the box with the highest number from the tableau
    l = sum(len(t) for t in tableau)
    def remove_l(t):
        t2 = t[:]
        if l in t:
            t2.remove(l)
        return t2
    tableau_bar = [remove_l(t) for t in tableau]
    if tableau_bar[-1] == []:
        tableau_bar = tableau_bar[:-1]

    # Compute the projector associated with the parent tableau
    e_bar = make_young_projector(tableau_bar, n=n)

    def mul(*args):
        if len(args) == 1:
            return args[0]
        if len(args) == 2:
            return compose_permutation_algebra_elements(*args)
        return mul(*args[:-2], compose_permutation_algebra_elements(*args[-2:]))

    # Get symmetrizer each row
    ps = []
    for row in tableau:
        if len(row) > 0:
            ps.append(symmetrizer(row, signed=0, n=n))
    p = mul(*ps)

    # Get antisymmetrizer for each column
    ns = []
    tableau_t = transpose_tableau(tableau)
    for col in tableau_t:
        if len(col) > 0:
            ns.append(symmetrizer(col, signed=1, n=n))
    n = mul(*ns)

    # Compute the normalizaiton
    hook_norm = 1
    for i, row in enumerate(tableau):
        hook_r = len(row) - i
        for j in range(len(row)):
            col = tableau_t[j]
            hook_c = len(col) - j
            hook_norm *= (hook_r + hook_c - 1)

    # Apply the recursive definition in Eq (86)
    # The product "(symmetrizer) x (antisymmetrizer)" is Eq (26)
    proj_unnorm = mul(e_bar, p, n, e_bar)
    proj = [
        WeightedPermutation(weight / hook_norm, perm)
        for (weight, perm) in proj_unnorm ]
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
    permutation : ``(|O|, |O|)`` ndarray
        The representation matrix, which happens to be a permutation matrix.

    Notes
    -----
    A group acts on a vector to generate an orbit O. When the group then acts
    on the orbit, the result is a permutation of the vectors in the orbit.
    The permutation can be represented as a square matrix of size ``|O|x|O|``,
    where ``|O|`` is the size of the orbit. The full set of these matrices (with
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
    Computes the representation of a group G acting on an orbit O.

    Parameters
    ----------
    orbit : list
        The orbit of an ordered set of vectors under the group
    group : ``(|G|, 3, 3)`` array_like
        The group matrices.

    Returns
    -------
    representation : ``(|G|, |O|, |O|)`` ndarray
        The momentum-representation matrices :math:`D_{m,m'}(R)` for all
        :math:`R \in G`.
    """
    return np.array([make_momentum_orbit_rep_matrix(orbit, g) for g in group])


def make_momentum_spin_rep(Dmm, *Dspin):
    """
    Computes the combined momentum-spin representation matrices.

    Parameters
    ----------
    Dmm : ``(|G|,|O|,|O|)`` ndarray
        The momentum-representation matrices.
    *Dspin : ``(|G^D|, |\Gamma|, |\Gamma|)`` ndarray(s)
        The spin irrep matrices.

    Returns
    -------
    Dmm_spin : ``(|G^D|, dim_total, dim_total)`` ndarray
        The combined momomentum-spin representation matrices, where the total
        dimension is given by dim_total = ``|O|x|irrep1|x|irrep2|x...x|irrepN|``.
        As should be expected, the proudct includes all the representations
        appearing in the list of "Dspin" matrices.
    """
    if len(Dspin) == 0:
        # No spin irrep matrices are present
        return Dmm

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
    group_element : ``(3, 3)`` array_like
        The group element.

    Returns
    -------
    irrep_matrix : ``(|\Gamma|, |\Gamma|)`` ndarray
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
    group: ``(|G|, 3, 3)`` array_like
        The group matrices.

    Returns
    -------
    irrep : ``(|G|, |\Gamma|, |\Gamma|)`` ndarray
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
    group_element : ``(4, 4)`` ndarray
        The "spinorial" group element.

    Returns
    -------
    irrep_matrix : ``(|\Gamma|, |\Gamma|)`` ndarray
        The irrep matrix.

    Notes
    -----
    Each basis state should be a list of tuples "(coefficient, SpinorTuple)"
    """
    assert group_element.shape == (4,4), "group element should act on spinors"

    def _make_spinor_array_lincombo(basis):
        """
        Computes a linear combination of "spinor states" inside a suitable
        tensor product space of spin-1/2 states.
        """
        result = []
        for state in basis:
            result.append(
                np.sum([c*make_spinor_array(ket) for c, ket in state], axis=0))
        return np.array(result)

    # Grab total j, making sure that it is specified consistently for all states
    j = np.unique([psi.j for _, psi in irrep_basis[0]]).item()
    nprod = get_nprod(j)
    cgs = _make_spinor_array_lincombo(irrep_basis)
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
    group : ``(|G|, 4, 4)`` list or array_like
        The group matrices.

    Returns
    -------
    irrep : ``(|G|, |\Gamma|, |\Gamma|)`` ndarray
        The irrep matrices.
    """
    return np.array([make_irrep_matrix_spinor(basis, g) for g in group])


def make_irrep_from_group(little_group):
    """
    Computes the irrep matrices associated with a particular little group.

    Parameters
    ----------
    little_group : ``(|G|, 3, 3)`` ndarray
        The matrices for the little group G.

    Returns
    -------
    Dmumu : dict
        The irrep matrices as a dict. The keys give the name of the irrep.
        The values contain the irrep matrices themselves, each with shape
        ``(|G|, |\Gamma|, |\Gamma|)``.
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
    Computes double-cover irrep matrices associated with a given little group,
    including both spinorial and bosonic irreps.

    Parameters
    ----------
    little_group : ``(|G|, 3, 3)`` ndarray
        The matrices for the little group G.

    Returns
    -------
    Dmumu_double : dict
        The irrep matrices as a dict. The keys give the name of the irrep.
        The values contain the irrep matrices themselves, each with shape
        ``(|G^D|, |\Gamma|, |\Gamma|)``.
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


##################################
# Block diagonalization matrices #
##################################


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
    basis = np.array(basis)
    assert np.allclose(basis.conj() @ basis.T, np.eye(len(basis))),\
        "Failure to construct orthonormal basis."
    return basis


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
        raise RuntimeError(f"Incomensurate shapes, {basis.shape}, {vector.shape}")

    # Compute projections
    v_parallel = np.zeros(vector.shape)
    for w_vec in basis:
        v_parallel = v_parallel + w_vec*np.dot(np.conjugate(w_vec), vector)
    if direction == 'perpendicular':
        return vector - v_parallel
    return v_parallel  # parallel


def apply_schur(Dmm, Dmumu, verbose):
    """
    Computes the block diagonalization matrices using Schur's algorithm,
    including transition operators to move between rows in a given irrep.

    Parameters
    ----------
    Dmm : ``(|G|, |O|, |O|)`` array_like
        Momentum-(spin)-representation matrices
    Dmumu : dict
        The irrep matrices. The keys give the name of the irrep. The values
        contain the group irrep matrices, with shape ``(|G|, |\Gamma|, |\Gamma|)``.

    Returns
    -------
    u_matrix : dict
        The block diagonalization matrix.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the matrices, each of shape
        ``(|\Gamma|, |O|)``.
    transition_operators: dict
        Column of transition operators ``T_{\mu,0}``.
        The keys are irrep names.
        The values are ndarrays of shape ``(|\Gamma|, |O|, |O|)``.
    """
    u_matrix = {}
    transition_operators = {}
    for irrep_name in Dmumu:
        _, dim, _  = Dmumu[irrep_name].shape
        # Construct transition operators using Wonderful orthogonality theorem
        # Comute the column T_{mu,0}, since that's all that's needed.
        T = np.zeros((dim, Dmm.shape[1], Dmm.shape[2]), dtype=complex)
        for mu in range(dim):
            T[mu] = np.einsum("i,ijk->jk",
                                Dmumu[irrep_name][:,mu,0].conj(),
                                Dmm)
            T[mu] *= dim/len(Dmumu)

        if np.allclose(T[0], 0):
            # The present irrep doesn't appear in the decompostion. Carry on.
            continue
        transition_operators[irrep_name] = T

        # Count the number of degenerate copies
        rank = np.linalg.matrix_rank(T[0])
        basis = orth(T[0].T)
        assert len(basis) == rank, (
            "Failure: expected len(basis)=rank. "
            f"Found len(basis)={len(basis)} and rank={rank}.")
        if verbose:
            print(f"Located irrep {irrep_name} with degeneracy {rank}.")
        for kappa in range(rank):
            vecs = [basis[kappa]]
            # Construct subsequent rows with transition operators T_{mu,0}
            for mu in range(1, dim):
                new_vec = T[mu] @ basis[kappa]
                new_vec /= np.linalg.norm(new_vec)
                if np.isclose(np.linalg.norm(new_vec), 0):
                    raise ValueError(f"Zero vector encountered by T[{mu},0]")
                if not np.isclose(np.linalg.norm(new_vec), 1):
                    raise ValueError(f"Normalization broken during T[{mu},0].")
                vecs.append(new_vec)
            vecs = rephase(np.array(vecs), irrep_name)
            u_matrix[(irrep_name, kappa)] = vecs

        for kappa, vec in enumerate(vecs):
            if not np.allclose(np.linalg.norm(vec), 1):
                print(irrep_name, kappa, "norm", np.linalg.norm(vec))

    # Check results for consistency
    test_row_orthogonality(u_matrix, verbose=verbose)
    test_degenerate_orthogonality(u_matrix, verbose=verbose)
    test_block_diagonalization(Dmm, Dmumu, u_matrix, verbose=verbose)
    return u_matrix, transition_operators


def rephase(arr, irrep):
    """
    Applies a phase convention to block diagonalization matrices

    Parameters
    ----------
    arr : ``(|\Gamma|, |O|)`` array_like
        Table specifiying the matrix

    Returns
    -------
    arr : ``(|\Gamma|, |O|)`` ndarray
        The table with appropriate phases applied.

    Notes
    -----
    The phase convention is as follows:
      - For a generic irrep, the phase is chosen such that the first nonzero
        entry of the first row (:math:`\mu=1`) is real and positive.
      - For the irreps :math:`T_2^+` and :math:`T_2^-` only, the phase is chosen
        such that the second row (:math:`\mu=2`) is purely imaginary with a negative
        imaginary part. This choice matches the basis-vector conventions of
        Basak et al., where a particular combination of spheric harmonics
        :math:`(Y_2^2 - Y_2^{-2})` is used as the :math:`\mu=2` basis vector for T2.

    References
    ----------
    .. [1] S. Basak et al., "Clebsch-Gordan construction of lattice interpolating
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
        assert np.isclose(phi, 270), f"Bad angle, phi={phi} deg."
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

def load_particle_info(fname=None):
    """
    Loads tabulated information for particle names, spins (boson vs fermion),
    and spin irreps.

    Parameters
    ----------
    fname : str or None
        The path to the input yaml file with the tabulated data

    Returns
    -------
    table : dict
        The particle information in the form {<name> : (<fermion?>, <irrep>}
    """
    ParticleInfo = namedtuple("SpinIrrep", ["is_fermion", "irrep", "spin_dim"])
    if fname is None:
        fname = os.path.join(pathlib.Path(__file__).parent.resolve(), 'particles.yaml')
    with open(fname, 'r', encoding='utf-8') as ifile:
        table = yaml.safe_load(ifile)
    for key, value in table.items():
        table[key] = ParticleInfo(*value)
    return table


def make_pseudoscalar_irrep(little):
    """
    Instantiates the double-cover irrep matrices associated with a pseudoscalar
    particle, assumed to transform under the :math:`A_1^-` irrep. Usually this
    function is used when constructing the `Dspin` matrices.

    Parameters
    ----------
    little : ``(|G|, 3, 3)`` array_like
        The little-group matrices associated with some momenta

    Returns
    -------
    pseudoscalar : ``(|G^D|, 1, 1)`` ndarray
        The A1m irrep matrices restricted to the double cover of the little group.
    """
    oh = make_oh()
    idxs = np.hstack([np.where([np.allclose(gg, g) for gg in oh]) for g in little]).squeeze()
    pseudoscalar = np.vstack(2*[make_irrep_from_group(oh)['A1m'][idxs]])
    return pseudoscalar


def make_spin_half_irrep(little_double):
    """
    Instantiates the double-cover irrep matrices associated with a spin-half
    particle, assumed to transform under the :math:`G_1^+` irrep. Usually this
    function is used when constructing the `Dspin` matrices.

    Parameters
    ----------
    little_double : ``(|G^D|, 4, 4)`` array_like
        The spinorial little-group matrices associated with some momenta

    Returns
    -------
    pseudoscalar : ``(|G^D|, 1, 1)`` ndarray
        The G_1^+ irrep matrices restricted to the double cover of the little group.
    """
    return make_irrep_spinor(
            basis_functions.basis_spinors["nucleon"]["nucleon"],
            little_double)


def make_Dspin(spin_irreps, little, little_double):
    """
    Builds a list of spin irrep matrices for the specified particles.

    Parameters
    ----------
    spin_irreps : list
        The names of the particles' spin irreps as strings, e.g., ['A1m', 'G1p']
    little : ``(|G|, 3, 3)`` array_like
        The little-group matrices associated with some momenta
    little_double : ``(|G^D|, 4, 4)`` array_like
        The spinorial little-group matrices associated with some momenta

    Returns
    -------
    Dspin : list
        The "spin" irrep matrices for each particle
    """
    if spin_irreps is None:
        return []

    if 'A1m' in spin_irreps:
        pseudoscalar = make_pseudoscalar_irrep(little)
    if 'G1p' in spin_irreps:
        spin_half = make_spin_half_irrep(little_double)
    Dspin = []
    for irrep in spin_irreps:
        if irrep == 'A1m':
            Dspin.append(pseudoscalar)
        elif irrep == 'G1p':
            Dspin.append(spin_half)
        else:
            raise ValueError(f"Unexpected irrep '{irrep}'")
    return Dspin


def identify_spin_dim(irrep):
    """
    Identifies the dimension of the specified spin irrep.

    Parameters
    ----------
    str : irrep
        The name of the spin irrep, e.g., 'A1m' or 'G1p'.

    Returns
    -------
    dim : int
        The dimension of the irrep

    """
    bases = [basis_functions.basis_fcns, basis_functions.basis_spinors]
    for basis in bases:
        for irrep_dict in basis.values():
            for irrep_name, basis_fcns in irrep_dict.items():
                if irrep_name == irrep:
                    return len(basis_fcns)
    raise ValueError(f"Unable to locate irrep '{irrep}'")


class IrrepDecomposition:
    def __init__(self, decomp, orbit, Dmm, Dmumu, little_name, stab_name, transition_operators):
        """
        Container for results of computing the block-diagonalization matrices
        which project linear combinations of plane-wave states onto irreps of
        the cubic group.

        Parameters
        ----------
        decomp : dict
            The irrep decomposition and change-of-basis matrices.
            The keys are tuples (irrep_name, degeneracy_idx).
            The values are the block-diagonalization matrices, given as arrays
            of shape ``(|\Gamma|, |O|)``.
        orbit : ``(|O|,)`` list of :class:`SpinShellTuple`
            The "extended" spin-momentum orbit, where each element is a
            :class:`SpinShellTuple` specifying momentum and spin indices.
        Dmm : ``(|G|, |O|, |O|)``, ndarray
            The (reducible) representation matrices associated with the orbit.
        Dmumu : ``(|G|, |\Gamma|, |\Gamma|)``, ndarray
            The irrep matrices
        little_name : str
            The name of the little group leaving the total momentum invariant
        stab_name : str
            The name of the stabilizer group leaving the ordered set of
            momenta invariant.
        transition_operators: dict
            Column of transition operators ``T_{\mu,0}``.
            The keys are irrep names.
            The values are ndarrays of shape ``(|\Gamma|, |O|, |O|)``.


        Notes
        -----
        Projection onto cubic-group irreps requires two pieces.
        1.) A basis of momentum plane-wave correlation functions, presumably
            computed using lattice QCD.
        2.) The block-diagonalization matrices computed using this module.

        To carry out this projection, the basis of correlation functions must
        have the same order as the orbit used to compute the block-diagonalization
        matrices. The required ordering can be seen by examining the
        "orbit."
        """
        # Check expected shape of block diagonalization matrices
        for _, arr in decomp.items():
            assert arr.shape[1] == len(orbit)
        # Check expected shape of (reducible) representaiton matrices
        if Dmm is not None:
            assert (Dmm.shape[1] == Dmm.shape[2]) & (Dmm.shape[1] == len(orbit))

        self.decomp = decomp
        self.orbit = orbit
        self.Dmm = Dmm
        self.Dmumu = Dmumu
        self._little_name = little_name
        self._stab_name = stab_name
        self.transition_operators = transition_operators

    def format(self, latex=False):
        r"""
        Formats the irrep decomposition as text string.

        Parameters
        ----------
        latex : bool
            Whether or not return a latex-formatted sting

        Returns
        -------
        output : str
            The irrep decomposition, e.g., ``"A1p + A2p + 2*Ep"`` or
            ``"$A_1^+ \oplus A_2^+ \oplus 2E^+$"``.
        """
        # Count the degenerate copies
        irrep_counts = {}
        for irrep, _ in self.decomp:
            if irrep not in irrep_counts:
                irrep_counts[irrep] = 1
            else:
                irrep_counts[irrep] += 1

        # Sort the names according to a conventional ordering
        names = list(irrep_counts.keys())
        degeneracies = list(irrep_counts.values())
        names, degeneracies = zip(
                *sorted(zip(names, degeneracies),
                key=lambda pair: self._irrep_priority(pair[0])))
        if latex:
            output = r" \oplus ".join([f"{degen}*{irrep}" for degen, irrep in zip(degeneracies, names)])
            output = self._irrep_to_latex(output)
            output = output.replace("1*", "").replace("*", "")
            output = "$" + output + "$"
        else:
            output = " + ".join([f"{degen}*{irrep}" for degen, irrep in zip(degeneracies, names)])
        return output

    def _irrep_priority(self, irrep_name):
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

    def _irrep_to_latex(self, output):
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

    def _format_group_name(self, name):
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

    def little_name(self, latex=False):
        """
        Returns the name of the little group.

        Parameters
        ----------
        latex : bool
            Whether or not return a latex-formatted sting

        Returns
        -------
        little_name : str
            The name of the little group
        """
        if latex:
            return self._format_group_name(self._little_name)
        return self._little_name

    def stab_name(self, latex=False):
        """
        Returns the name of the stabilizer group.

        Parameters
        ----------
        latex : bool
            Whether or not return a latex-formatted sting

        Returns
        -------
        stab_name : str
            The name of the stabilizer group
        """
        if latex:
            return self._format_group_name(self._stab_name)
        return self._stab_name


def mhi(momenta, spin_irreps=None, internal_symmetry=None, verbose=False):
    r"""
    General-purpose driver function for construction of change-of-basis /
    block-diagonalization matrices which project linear combinations of plane-
    wave states onto irreps of the cubic group.

    Parameters
    ----------
    momenta : ``(nparticles, 3)`` or ``(3, )`` array_like
        The ordered momenta.
    particle_names : array_like
        The particle names, e.g., ['n', 'p'].
    verbose : bool
        Whether or not to print extra diagnostic information.
    return_Dmm : bool
        Whether or not to return the momentum-(spin) representation matrices.
    internal_symmetry : list of :class:`WeightedPermutation`
        The exchange group projector defined in the group algebra.

    Returns
    -------
    result : IrrepDecomposition
        object containing the results of the irrep decomposition.

    Notes
    -----
    The algorithm is as follows:
      - Compute the little group of the total momentum
      - Compute irrep matrices of the little group, :math:`D_{\mu\nu}(R)`
      - Compute the momentum(-spin) representation matrices, :math:`D_{mm'}(R)`
      - Apply exchange-group projection, giving :math:`\hat{D}(R) = P D(R) P`
      - Apply Schur's algorithm and transition operators to construct the
        block-diagonalization matrices
    """
    if (spin_irreps is not None) and len(spin_irreps) > 0:
        if len(spin_irreps) != len(momenta):
            raise ValueError("Incomensurate momenta and spin irreps specified.")

    # 1. Compute the little group of the total momentum
    little, stab = make_little_and_stabilizer(momenta, group=make_oh())
    little_name = identify_stabilizer(little)
    stab_name = identify_stabilizer(stab)
    little_canonical = make_canonical_stabilizer(little_name, group=make_oh())
    isomorphism = find_subgroup_isomorphism(make_oh(), little_canonical , little)
    little = little[isomorphism.perm]  # Rotate to conventional orientation
    little_double = make_spinorial_little_group(little)

    # 2. Compute the irrep matrics of the little group
    if spin_irreps is None:
        # Distinguishable spin-zero particles. Single-cover irrep matrics suffice.
        Dmumu = make_irrep_from_group(little_canonical)
    else:
        Dmumu = make_irrep_from_groupD(little_canonical)

    # 3. Compute momentum(-spin) representation matrices
    orbit = make_momentum_orbit(momenta, little, internal_symmetry)
    if spin_irreps is None:
        spin_dims = [1 for _ in range(len(momenta))]
    else:
        spin_dims = [identify_spin_dim(irrep) for irrep in spin_irreps]
    orbit_momspin = make_momentum_spin_orbit(momenta, spin_dims, little, internal_symmetry)

    if verbose:
        print(f"Size of extended orbit: {len(orbit_momspin)}")
    if len(orbit_momspin) > 100:
        print((
            "Warning: detailed optimization efforts have not been made "
            "in the reference implementation.\nAdditional optimization may be "
            "required for calculations with large momentum-spin orbits."
        ))
    Dmm = make_momentum_orbit_rep(orbit, little)
    Dspin = make_Dspin(spin_irreps, little, little_double)
    Dmm_momspin = make_momentum_spin_rep(Dmm, *Dspin)

    # 4. Compute and apply projection from the internal symmetry group
    if internal_symmetry:
        # Naively, the projector applied to the Dmm should be proj @ Dmm(R) @ proj.
        # However, the projector is idempotent, and the permutations commute
        # with rotations. Therefore, it suffices to compute Dmm(R) @ proj.
        proj = make_internal_symmetry_projector(orbit_momspin, internal_symmetry)
        if proj is None:
            Dmm_momspin = None
        else:
            Dmm_momspin = np.einsum("ajk,kl", Dmm_momspin, proj, optimize='greedy')

    # 5. Apply Schur's lemma
    if Dmm_momspin is None:
        result, transition_operators = {}, {}
    else:
        result, transition_operators = apply_schur(Dmm_momspin, Dmumu, verbose)

    if verbose and (len(result) == 0):
        print("Decomposition vanishes for specified inputs.")

    return IrrepDecomposition(result, orbit_momspin, Dmm_momspin, Dmumu, little_name, stab_name, transition_operators)


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
    Computes the anticommutator of two matrices.

    Parameters
    ----------
    arr1 : array_like
    arr2 : array_like

    Returns
    -------
    arr : ndarray
        The anticommutator {arr1, arr2}
    """
    return arr1 @ arr2 + arr2 @ arr1


def commutator(arr1, arr2):
    """
    Computes the commutator of two matrices.

    Parameters
    ----------
    arr1 : array_like
    arr2 : array_like

    Returns
    -------
    arr : ndarray
        The commutator {arr1, arr2}
    """
    return arr1 @ arr2 - arr2 @ arr1


def test_clifford(gamma, eta, verbose=False):
    r"""
    Tests the Clifford-algebra condition,
    :math:`\{\gamma_\mu, \gamma_\nu\} = 2*\eta_{\mu\nu}`

    Parameters
    ----------
    gamma : ``(4, )`` list or ``(4, 4, 4)`` array_like
        The gamma matrices gamma[i], i=0,1,2,3.
    eta : ``(4,4)`` ndarray
        The metric.
    verbose : bool
        Whether to print additional information about successful tests.

    Returns
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
    Tests anticommutation of gamma5,
    :math:`\{\gamma_\mu, \gamma_5\} = 0`

    Parameters
    ----------
    gamma : ``(4, )`` list or ``(4, 4, 4)`` array_like
        The gamma matrices gamma[i], i=0,1,2,3.
    gamma5 : ``(4,4)``
        The matrix gamma5.
    verbose : bool
        Whether to print additional information about successful tests.

    Returns
    -------
    None
    """
    zero4 = np.zeros((4, 4))
    for mu in range(4):
        assert np.all(anticommutator(gamma[mu], gamma5) == zero4),\
            f"gamma_{mu} does not anticommute with gamma5"
    if verbose:
        print("Success: gamma5 anticommutes.")


def test_row_orthogonality(u_matrix, verbose=False):
    """
    Tests the row orthogonality of tables of block-diagonalization matrices,
    for a given irrep.

    Parameters
    ----------
    u_matrix : dict
        The block diagonalization matrix.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the matrices, each of shape
        ``(|\Gamma|, |O|)``.
    verbose : bool
        Whether to print additional information about successful tests.

    Returns
    -------
    None
    """
    for irrep_name, kappa in u_matrix:
        arr = u_matrix[(irrep_name, kappa)]
        dim = arr.shape[0]
        assert np.allclose(arr.conj() @ arr.T, np.eye(dim)),\
            f"Failure of row orthogonality {irrep_name} {kappa}"
        if verbose:
            print(f"Success: orthogonal rows for {irrep_name}, kappa={kappa}")


def count_degeneracy(u_matrix):
    """
    Counts the degeneracy of irreps within the tables of block-diagnoalization
    matrices.

    Parameters
    ----------
    u_matrix : dict
        The block diagonalization matrix.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the matrices, each of shape
        ``(|\Gamma|, |O|)``.

    Returns
    -------
    degeneracy : dict
        The degeneracies, with keys corresponding to the irrep names and values
        to the counts.
    """
    degeneracy = {}
    for irrep_name, _ in u_matrix:
        if irrep_name in degeneracy:
            degeneracy[irrep_name] += 1
        else:
            degeneracy[irrep_name] = 1
    return degeneracy


def test_degenerate_orthogonality(u_matrix, verbose=False):
    """
    Tests that tables corresponding to degenerate irreps are orthogonal.

    Parameters
    ----------
    u_matrix : dict
        The block diagonalization matrix.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the matrices, each of shape
        ``(|\Gamma|, |O|)``.
    verbose : bool
        Whether to print additional information about successful tests.

    Returns
    -------
    None
    """
    degeneracy = count_degeneracy(u_matrix)
    for irrep_name, rank in degeneracy.items():
        if rank == 1:
            continue
        for k1, k2 in itertools.product(range(rank), repeat=2):
            if k1 == k2:
                continue
            arr1 = u_matrix[(irrep_name, k1)]
            arr2 = u_matrix[(irrep_name, k2)]
            dim = arr1.shape[0]
            assert np.allclose(arr1.conj() @ arr2.T, np.zeros(dim)),\
                f"Failure of orthogonality {irrep_name} {k1} {k2}"
    if verbose:
        print("Success: all irreps orthogonal")


def test_block_diagonalization(Dmm, Dmumu, u_matrix, verbose=False):
    r"""
    Tests the block diagonalization property of the change-of-basis matrices,
    :math:`D_{\mu\nu} = U^\dagger_{\mu m} D_{mm'} U_{m' \nu}`
    in terms of the given momentum-shell representation `Dmm`, the
    block-diagonalization matrix `U`, and the block-diagonal irrep matrix
    `Dmumu`.

    Parameters
    ----------
    Dmm : ``(|G|,|O|,|O|)`` ndarray
        The momentum-representation matrices.
    Dmumu : dict
        The irrep matrices as a dict. The keys give the name of the irrep.
        The values contain the irrep matrices themselves, each with shape
        ``(|G|, |\Gamma|, |\Gamma|)``.
    u_matrix : dict
        The block diagonalization matrix.
        The keys are tuples of the form (irrep_name, degeneracy_number).
        The values are arrays containing the matrix, each of shape
        ``(|\Gamma|, |O|)``.
    verbose : bool
        Whether to print additional information about successful tests.

    Returns
    -------
    None
    """
    for (irrep, _), umat in u_matrix.items():
        test = np.allclose(
            Dmumu[irrep],
            np.einsum("ia,rab,jb->rij", umat.conj(), Dmm, umat))
        assert test, f"Failure: block diagonalization {irrep}"
    if verbose:
        print("Success: block diagonalization using projection matrices")


############
# File I/O #
############

def write_hdf5(h5fname, result_dict):
    """
    Writes block-diagonalization matrices to HDF5.

    Parameters
    ----------
    h5fname : str
        The name of the output file
    result_dict : dict
        The block-diagonalization / change of basis matrices, where each key
        is a tuple the form (irrep_name, degeneracy) and each key is an ndarray
        with the associated matrix.

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
    Reads saved block-diagonalization matrices from HDF5.

    Parameters
    ----------
    h5fname : str
        The name of HDF5 file to read

    Returns
    -------
    result_dict : dict
        The block-diagonalization / change of basis matrices, where each key
        is a tuple the form (irrep_name, degeneracy) and each key is an ndarray
        with the associated matrix.

    """
    result_dict = {}
    with h5py.File(h5fname, 'r') as ifile:
        for irrep_name in ifile['data']:
            for degeneracy in ifile['data'][irrep_name]:
                print(irrep_name, degeneracy)
                result_dict[(irrep_name, int(degeneracy))] =\
                    np.array(ifile['data'][irrep_name][degeneracy][:])
    return result_dict
