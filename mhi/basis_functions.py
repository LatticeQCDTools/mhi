from collections import namedtuple
import numpy as np
import sympy


class SpinorTuple(namedtuple("SpinorTuple", ['j', 'jz', 'parity'])):
    """A spinor basis state ``|j, jz, parity>``.

    Attributes
    ----------
    j : float
        Half-integer total spin.
    jz : float
        Half-integer z component of spin.
    parity : int
        Parity defined as one of [+1,-1].
    """

X, Y, Z = sympy.symbols("x y z")
R = np.array([X, Y, Z])

basis_fcns = {}

#: Basis functions for irrep A1+ of group Oh.
basis_Oh_A1p = [(X**2 + Y**2 + Z**2)/np.sqrt(3)]
#: Basis functions for irrep A1- of group Oh.
basis_Oh_A1m = [(X*Y*Z * (X**4*(Y**2 - Z**2)+ Y**4*(Z**2 - X**2) + Z**4*(X**2 - Y**2)))/np.sqrt(6)]
#: Basis functions for irrep A2+ of group Oh.
basis_Oh_A2p = [(X**4 * (Y**2 - Z**2) + Y**4 * (Z**2 - X**2) + Z**4 * (X**2 - Y**2))/np.sqrt(6)]
#: Basis functions for irrep A2- of group Oh.
basis_Oh_A2m = [X*Y*Z]
#: Basis functions for irrep E+ of group Oh.
basis_Oh_Ep = [
    (2*Z**2 - X**2 - Y**2)/np.sqrt(6),
    (X**2 - Y**2)/np.sqrt(2),
]
#: Basis functions for irrep E- of group Oh.
basis_Oh_Em = [
    X*Y*Z * (X**2 - Y**2),
    -X*Y*Z*(2*Z**2 - X**2 - Y**2),
]
#: Basis functions for irrep T1- of group Oh.
basis_Oh_T1m = [
    np.sqrt(2)*Z,
    -X+1j*Y,
    X+1j*Y,
]
#: Basis functions for irrep T1+ of group Oh.
basis_Oh_T1p = [
    np.sqrt(2)*X*Y*(X**2 - Y**2),
    1j*Z*X*(Z**2 - X**2) - Y*Z*(Y**2 - Z**2),
    1j*Z*X*(Z**2 - X**2) + Y*Z*(Y**2 - Z**2),
]
#: Basis functions for irrep T2- of group Oh.
basis_Oh_T2m = [
    -Y*(Z**2 - X**2) + 1j*X*(Y**2 - Z**2),
    -1j*np.sqrt(2)*Z*(X**2 - Y**2),
    Y*(Z**2 - X**2) + 1j*X*(Y**2 - Z**2),
]
#: Basis functions for irrep T2+ of group Oh.
basis_Oh_T2p = [
    -Z*X + 1j*Y*Z,
    -1j*np.sqrt(2)*X*Y,
    Z*X + 1j*Y*Z,
]
basis_fcns["Oh"] = {
    "A1p": basis_Oh_A1p,
    "A1m": basis_Oh_A1m,
    "A2p": basis_Oh_A2p,
    "A2m": basis_Oh_A2m,
    "Ep": basis_Oh_Ep,
    "Em": basis_Oh_Em,
    "T1m": basis_Oh_T1m,
    "T1p": basis_Oh_T1p,
    "T2m": basis_Oh_T2m,
    "T2p": basis_Oh_T2p,
}

#: Basis functions for irrep A1 of C4v = A1+ of Oh.
basis_C4v_A1 = basis_fcns["Oh"]["A1p"]
#: Basis functions for irrep A2 of C4v = A1- of Oh.
basis_C4v_A2 = basis_fcns["Oh"]["A1m"]
#: Basis functions for irrep B1 of C4v = E1+[1] of Oh.
basis_C4v_B1 = [basis_fcns["Oh"]["Ep"][1],]
#: Basis functions for irrep B2 of C4v = T2+[1] of Oh.
basis_C4v_B2 = [basis_fcns["Oh"]["T2p"][1],]
#: Basis functions for irrep E of C4v.
basis_C4v_E = [-X*Z, -Y*Z]
basis_fcns["C4v"] = {
    "A1": basis_C4v_A1,
    "A2": basis_C4v_A2,
    "B1": basis_C4v_B1,
    "B2": basis_C4v_B2,
    "E" : basis_C4v_E,
}

#: Basis functions for irrep A1 of C3v = A1+ of Oh.
basis_C3v_A1 = basis_fcns["Oh"]["A1p"]
#: Basis functions for irrep A2 of C3v = A2+ of Oh.
basis_C3v_A2 = basis_fcns["Oh"]["A2p"]
#: Basis functions for irrep E of C3v = E+ of Oh.
basis_C3v_E = basis_fcns["Oh"]["Ep"]
basis_fcns["C3v"] = {
    "A1": basis_C3v_A1,
    "A2": basis_C3v_A2,
    "E": basis_C3v_E,
}

#: Basis functions for irrep A1 of C2v = A1+ of Oh.
basis_C2v_A1 = basis_fcns["Oh"]["A1p"]
#: Basis functions for irrep A2 of C2v = A1- of Oh.
basis_C2v_A2 = basis_fcns["Oh"]["A1m"]
#: Basis functions for irrep B1 of C2v = A2- of Oh.
basis_C2v_B1 = basis_fcns["Oh"]["A2m"]
#: Basis functions for irrep B2 of C2v = A2+ of Oh.
basis_C2v_B2 = basis_fcns["Oh"]["A2p"]
basis_fcns["C2v"] = {
    "A1": basis_C2v_A1,
    "A2": basis_C2v_A2,
    "B1": basis_C2v_B1,
    "B2": basis_C2v_B2,
}

#: Basis functions for irrep A of C2R = A1+ of Oh.
basis_C2R_A = basis_fcns["Oh"]["A1p"]
#: Basis functions for irrep B of C2R = A1- of Oh.
basis_C2R_B = basis_fcns["Oh"]["A1m"]
basis_fcns["C2R"] = {
    "A": basis_C2R_A,
    "B": basis_C2R_B,
}

#: Basis functions for irrep A of C2P = A1+ of Oh.
basis_C2P_A = basis_fcns["Oh"]["A1p"]
#: Basis functions for irrep B of C2P = A2+ of Oh.
basis_C2P_B = basis_fcns["Oh"]["A2p"]
basis_fcns["C2P"] = {
    "A": basis_C2P_A,
    "B": basis_C2P_B,
}

#: Basis functions for irrep A of C1 = A1+ of Oh.
basis_C1_A = basis_fcns["Oh"]["A1p"]
basis_fcns["C1"] = {
    "A": basis_C1_A,
}


basis_spinors = {}
#: Basis spinors for irrep G1+ of OhD.
basis_OhD_G1p = [
    [(1, SpinorTuple(1/2, 1/2, 1)),],
    [(1, SpinorTuple(1/2, -1/2, 1)),],
]
#: Basis spinors for irrep G1- of OhD.
basis_OhD_G1m = [
    [(1, SpinorTuple(1/2, 1/2, -1)),],
    [(1, SpinorTuple(1/2, -1/2, -1)),],
]
#: Basis spinors for irrep G2+ of OhD.
basis_OhD_G2p = [
    [(np.sqrt(1/6), SpinorTuple(5/2, 5/2, 1)), (-np.sqrt(5/6), SpinorTuple(5/2, -3/2, 1)),],
    [(np.sqrt(1/6), SpinorTuple(5/2, -5/2, 1)), (-np.sqrt(5/6), SpinorTuple(5/2, 3/2, 1)),],
]
#: Basis spinors for irrep G2- of OhD.
basis_OhD_G2m = [
    [(np.sqrt(1/6), SpinorTuple(5/2, 5/2, -1)), (-np.sqrt(5/6), SpinorTuple(5/2, -3/2, -1)),],
    [(np.sqrt(1/6), SpinorTuple(5/2, -5/2, -1)), (-np.sqrt(5/6), SpinorTuple(5/2, 3/2, -1)),],
]
#: Basis spinors for irrep H+ of OhD.
basis_OhD_Hp = [
    [(1, SpinorTuple(3/2, 3/2, 1)),],
    [(1, SpinorTuple(3/2, 1/2, 1)),],
    [(1, SpinorTuple(3/2, -1/2, 1)),],
    [(1, SpinorTuple(3/2, -3/2, 1)),],
]
#: Basis spinors for irrep H- of OhD.
basis_OhD_Hm = [
    [(1, SpinorTuple(3/2, 3/2, -1)),],
    [(1, SpinorTuple(3/2, 1/2, -1)),],
    [(1, SpinorTuple(3/2, -1/2, -1)),],
    [(1, SpinorTuple(3/2, -3/2, -1)),],
]
basis_spinors["OhD"] = {
    # Two-dimensional irreps
    "G1p" : basis_OhD_G1p,
    "G1m" : basis_OhD_G1m,
    "G2p" : basis_OhD_G2p,
    "G2m" : basis_OhD_G2m,
    # Four-dimensional irreps
    "Hp": basis_OhD_Hp,
    "Hm": basis_OhD_Hm,
}

#: Basis spinors for irrep G1 of Dic4.
basis_Dic4_G1 = [
    [(1, SpinorTuple(1/2, +1/2, +1)),],
    [(1, SpinorTuple(1/2, -1/2, +1)),],
]
#: Basis spinors for irrep G2 of Dic4.
basis_Dic4_G2 = [
    [(np.sqrt(1/6), SpinorTuple(5/2, +5/2, +1)), (-np.sqrt(5/6), SpinorTuple(5/2, -3/2, +1)),],
    [(np.sqrt(1/6), SpinorTuple(5/2, -5/2, +1)), (-np.sqrt(5/6), SpinorTuple(5/2, +3/2, +1)),],
]
basis_spinors["Dic4"] = {
    "G1": basis_Dic4_G1,
    "G2": basis_Dic4_G2,
}

#: Basis spinors for irrep G of Dic3.
basis_Dic3_G = [
    [(1, SpinorTuple(1/2, +1/2, +1)),],
    [(1, SpinorTuple(1/2, -1/2, +1)),],
]
#: Basis spinors for irrep F1 of Dic3.
basis_Dic3_F1 = [[
    (1/2, SpinorTuple(3/2, +3/2, +1)),
    (-(1-1j)*(np.sqrt(2)-2*1j)/(4*np.sqrt(3)), SpinorTuple(3/2, +1/2, +1)),
    ((np.sqrt(2)+1j)/(2*np.sqrt(3)), SpinorTuple(3/2, -1/2, +1)),
    (-(1+1j)/(2*np.sqrt(2)), SpinorTuple(3/2, -3/2, +1)),],
]
#: Basis spinors for irrep F2 of Dic3.
basis_Dic3_F2 = [[
    (1/2, SpinorTuple(3/2, +3/2, +1)),
    ((1+1j)*(2-1j*np.sqrt(2))/(4*np.sqrt(3)), SpinorTuple(3/2, 1/2, +1)),
    (-(np.sqrt(2)-1j)/(2*np.sqrt(3)), SpinorTuple(3/2, -1/2, +1)),
    ((1+1j)/(2*np.sqrt(2)), SpinorTuple(3/2, -3/2, +1)),],
]
basis_spinors["Dic3"] = {
    "G": basis_Dic3_G,
    "F1": basis_Dic3_F1,
    "F2": basis_Dic3_F2,
}

#: Basis spinors for irrep G of Dic2.
basis_Dic2_G = [
    [(1, SpinorTuple(1/2, +1/2, +1)),],
    [(1, SpinorTuple(1/2, -1/2, +1)),],
]
basis_spinors["Dic2"] = {
    "G": basis_Dic2_G,
}

#: Basis spinors for irrep F1 of C4R.
basis_C4R_F1 = [[(1, SpinorTuple(1/2, +1/2, +1)),],]
#: Basis spinors for irrep F2 of C4R.
basis_C4R_F2 = [[(1, SpinorTuple(1/2, -1/2, +1)),],]
basis_spinors["C4R"] = {
    "F1": basis_C4R_F1,
    "F2": basis_C4R_F2,
}

#: Basis spinors for irrep F1 of C4P.
basis_C4P_F1 = [[
    (1/np.sqrt(2), SpinorTuple(1/2, +1/2, +1)),
    ((1-1j)/2, SpinorTuple(1/2, -1/2, +1)),],
]
#: Basis spinors for irrep F2 of C4P.
basis_C4P_F2 = [[
    (1/np.sqrt(2), SpinorTuple(1/2, +1/2, +1)),
    (-(1-1j)/2, SpinorTuple(1/2, -1/2, +1)),],
]
basis_spinors["C4P"] = {
    "F1": basis_C4P_F1,
    "F2": basis_C4P_F2,
}

#: Basis spinors for irrep F of C1D.
basis_C1D_F = [[(1, SpinorTuple(1/2, +1/2, +1)),],]
basis_spinors["C1D"] = {
    "F": basis_C1D_F,
}

#: Basis spinors for the nucleon.
basis_nucleon = [
    [(1, SpinorTuple(1/2, +1/2, +1)),],
    [(1, SpinorTuple(1/2, -1/2, +1)),],
]
basis_spinors["nucleon"] = {
    "nucleon": basis_nucleon,
}

# Translation between subgroups of Oh and the corresponding subgroups of OhD
# to allow for accessing the relevant basis vectors using either name
_keymap = {
    'Oh': 'OhD', 'C4v': 'Dic4', 'C3v': 'Dic3',
    'C2v': 'Dic2', 'C2R': 'C4R', 'C2P': 'C4P', 'C1': 'C1D'}

for new, existing in _keymap.items():
    basis_spinors[new] = basis_spinors[existing]
