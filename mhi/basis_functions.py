import sys
from collections import namedtuple
import numpy as np
import sympy


SpinorTuple = namedtuple("SpinorTuple", ['j', 'jz', 'parity'])

X, Y, Z = sympy.symbols("x y z")
R = np.array([X, Y, Z])
basis_fcns = {}
basis_fcns["Oh"] = {
    "A1p": [(X**2 + Y**2 + Z**2)/np.sqrt(3)],
    "A1m": [(X*Y*Z * (X**4*(Y**2 - Z**2)+ Y**4*(Z**2 - X**2) + Z**4*(X**2 - Y**2)))/np.sqrt(6)],
    "A2p": [(X**4 * (Y**2 - Z**2) + Y**4 * (Z**2 - X**2) + Z**4 * (X**2 - Y**2))/np.sqrt(6)],
    "A2m": [X*Y*Z],
    "Ep": [(2*Z**2 - X**2 - Y**2)/np.sqrt(6),
           (X**2 - Y**2)/np.sqrt(2),
          ],
    "Em": [X*Y*Z * (X**2 - Y**2),
           -X*Y*Z*(2*Z**2 - X**2 - Y**2),
          ],
    "T1m": [np.sqrt(2)*Z,
           -X+1j*Y,
           X+1j*Y,
           ],
    "T1p": [np.sqrt(2)*X*Y*(X**2 - Y**2),
            1j*Z*X*(Z**2 - X**2) - Y*Z*(Y**2 - Z**2),
            1j*Z*X*(Z**2 - X**2) + Y*Z*(Y**2 - Z**2),
           ],
    "T2m": [-Y*(Z**2 - X**2) + 1j*X*(Y**2 - Z**2),
            -1j*np.sqrt(2)*Z*(X**2 - Y**2),
            Y*(Z**2 - X**2) + 1j*X*(Y**2 - Z**2),
           ],
    "T2p": [-Z*X + 1j*Y*Z,
            -1j*np.sqrt(2)*X*Y,
            Z*X + 1j*Y*Z,
           ],
}

basis_fcns["C4v"] = {
    "A1": basis_fcns["Oh"]["A1p"],
    "A2": basis_fcns["Oh"]["A1m"],
    "B1": [basis_fcns["Oh"]["Ep"][1],],
    "B2": [basis_fcns["Oh"]["T2p"][1],],
    "E" : [1j*Y*Z, -1j*X*Z],  # I had[-X*Z, -Y*Z]  # TODO -- get squared away about this convention
}

basis_fcns["C3v"] = {
    "A1": basis_fcns["Oh"]["A1p"],
    "A2": basis_fcns["Oh"]["A2p"],
    "E": basis_fcns["Oh"]["Ep"],
}

basis_fcns["C2v"] = {
    "A1": basis_fcns["Oh"]["A1p"],
    "A2": basis_fcns["Oh"]["A1m"],
    "B1": basis_fcns["Oh"]["A2m"],
    "B2": basis_fcns["Oh"]["A2p"],
}

basis_fcns["C2R"] = {
    "A": basis_fcns["Oh"]["A1p"],
    "B": basis_fcns["Oh"]["A1m"],
}

basis_fcns["C2P"] = {
    "A": basis_fcns["Oh"]["A1p"],
    "B": basis_fcns["Oh"]["A2p"],
}

basis_fcns["C1"] = {
    "A": basis_fcns["Oh"]["A1p"],
}

basis_spinors = {}
basis_spinors["OhD"] = {
    # Two-dimensional irreps
    "G1p" : [
        [(1, SpinorTuple(1/2, 1/2, 1)),],
        [(1, SpinorTuple(1/2, -1/2, 1)),],
    ],
    "G1m" : [
        [(1, SpinorTuple(1/2, 1/2, -1)),],
        [(1, SpinorTuple(1/2, -1/2, -1)),],
    ],
    "G2p" : [
        [(np.sqrt(1/6), SpinorTuple(5/2, 5/2, 1)), (-np.sqrt(5/6), SpinorTuple(5/2, -3/2, 1)),],
        [(np.sqrt(1/6), SpinorTuple(5/2, -5/2, 1)), (-np.sqrt(5/6), SpinorTuple(5/2, 3/2, 1)),],
    ],
    "G2m" : [
        [(np.sqrt(1/6), SpinorTuple(5/2, 5/2, -1)), (-np.sqrt(5/6), SpinorTuple(5/2, -3/2, -1)),],
        [(np.sqrt(1/6), SpinorTuple(5/2, -5/2, -1)), (-np.sqrt(5/6), SpinorTuple(5/2, 3/2, -1)),],
    ],
    # Four-dimensional irreps
    "Hp": [
        [(1, SpinorTuple(3/2, 3/2, 1)),],
        [(1, SpinorTuple(3/2, 1/2, 1)),],
        [(1, SpinorTuple(3/2, -1/2, 1)),],
        [(1, SpinorTuple(3/2, -3/2, 1)),],
    ],
    "Hm": [
        [(1, SpinorTuple(3/2, 3/2, -1)),],
        [(1, SpinorTuple(3/2, 1/2, -1)),],
        [(1, SpinorTuple(3/2, -1/2, -1)),],
        [(1, SpinorTuple(3/2, -3/2, -1)),],
    ],
}

basis_spinors["Dic4"] = {
    "G1": [
        [(1, SpinorTuple(1/2, +1/2, +1)),],
        [(1, SpinorTuple(1/2, -1/2, +1)),],
    ],
    "G2": [
        [(np.sqrt(1/6), SpinorTuple(5/2, -5/2, +1)), (-np.sqrt(5/6), SpinorTuple(5/2, +3/2, +1)),],
        [(np.sqrt(1/6), SpinorTuple(5/2, +5/2, +1)), (-np.sqrt(5/6), SpinorTuple(5/2, -3/2, +1)),],
    ],
}

basis_spinors["Dic3"] = {
    "G": [
        [(1, SpinorTuple(1/2, +1/2, +1)),],
        [(1, SpinorTuple(1/2, -1/2, +1)),],
    ],
    "F1": [[
        (1/2, SpinorTuple(3/2, +3/2, +1)),
        (-(1-1j)*(np.sqrt(2)-2*1j)/(4*np.sqrt(3)), SpinorTuple(3/2, +1/2, +1)),
        ((np.sqrt(2)+1j)/(2*np.sqrt(3)), SpinorTuple(3/2, -1/2, +1)),
        (-(1+1j)/(2*np.sqrt(2)), SpinorTuple(3/2, -3/2, +1)),],
    ],
    "F2": [[
        (1/2, SpinorTuple(3/2, +3/2, +1)),
        ((1+1j)*(2-1j*np.sqrt(2))/(4*np.sqrt(3)), SpinorTuple(3/2, 1/2, +1)),
        (-(np.sqrt(2)-1j)/(2*np.sqrt(3)), SpinorTuple(3/2, -1/2, +1)),
        ((1+1j)/(2*np.sqrt(2)), SpinorTuple(3/2, -3/2, +1)),],
    ],
}

basis_spinors["Dic2"] = {
    "G": [
        [(1, SpinorTuple(1/2, +1/2, +1)),],
        [(1, SpinorTuple(1/2, -1/2, +1)),],
    ],
}

basis_spinors["C4R"] = {
    "F1":[[(1, SpinorTuple(1/2, +1/2, +1)),],],
    "F2":[[(1, SpinorTuple(1/2, -1/2, +1)),],],
}

basis_spinors["C4P"] = {
    "F1": [[
        (1/np.sqrt(2), SpinorTuple(1/2, +1/2, +1)),
        ((1-1j)/2, SpinorTuple(1/2, -1/2, +1)),],
    ],
    "F2": [[
        (1/np.sqrt(2), SpinorTuple(1/2, +1/2, +1)),
        (-(1-1j)/2, SpinorTuple(1/2, -1/2, +1)),],
    ],
}

basis_spinors["C1D"] = {
    "F": [[(1, SpinorTuple(1/2, +1/2, +1)),],],
}

basis_spinors["nucleon"] = {
    "nucleon": [
        [(1, SpinorTuple(1/2, +1/2, +1)),],
        [(1, SpinorTuple(1/2, -1/2, +1)),],
    ],
}

_keymap = {
    'Oh': 'OhD', 'C4v': 'Dic4', 'C3v': 'Dic3',
    'C2v': 'Dic2', 'C2R': 'C4R', 'C2P': 'C4P', 'C1': 'C1D'}

for new, existing in _keymap.items():
    basis_spinors[new] = basis_spinors[existing]