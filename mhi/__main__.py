"""
Command-line interace for the MHI.
"""
import os
import sys
import yaml
import argparse
import numpy as np
import h5py
from . import mhi

def print_verbose(verbose, *objects):
    """
    Prints objects if verbose==True.
    """
    if verbose:
        print(*objects)

def parse_internal_symmetry(alist, nparticles):
    """
    Parses the yaml list representation of an internal symmetry projector
    into the form exected by the code.
    """
    result = []
    identity = np.arange(nparticles)
    for weight, perm in alist:
        assert np.allclose(np.sort(perm), identity),\
            f"Invalid permutation {perm}"
        result.append(mhi.WeightedPermutation(weight, perm))
    return result

# Handle command-line arguments
parser = argparse.ArgumentParser(
    prog='mhi',
    description=(
        'Constructs block-diagonalization coefficients to project '
        'plane-wave operators into irreps of the cubic group.'),
    epilog="Invoke using '$ python -m mhi <ifile> <ofile>'")
parser.add_argument('ifile',
                    type=str,
                    help=(
                        'Path to the input file specifying the momenta '
                        'and other properties of the interpolating operators.')
                    )
parser.add_argument('ofile',
                    type=str,
                    help='Path for the output file where the results will be saved.')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-c', '--clobber', action='store_true',
                    help='Whether or not to overwrite existing output files')
args = parser.parse_args()

# Verify input/output files
if not os.path.exists(args.ifile):
    raise FileNotFoundError(f"Input file {args.ifile} not found.")

if not args.ofile.endswith(('h5', 'hdf5')):
    raise ValueError("Output file must end with suffix 'h5' or 'hdf5'.")

if os.path.exists(args.ofile) and (not args.clobber):
    raise FileExistsError(f"Output file {args.ofile} already exists")

# Read and unpackage physics inputs
print_verbose(args.verbose, "Reading physics inputs from", args.ifile)
with open(args.ifile, 'r', encoding='utf-8') as ifile:
    inputs = yaml.safe_load(ifile)
momenta = np.array(inputs['momenta'])
spin_irreps = inputs['spin_irreps']
internal_symmetry = inputs['internal_symmetry']
if len(internal_symmetry):
    internal_symmetry = parse_internal_symmetry(internal_symmetry, len(momenta))
else:
    internal_symmetry = None

# Verify consistency of physics inputs
assert len(momenta.shape) == 2, f"Expected 2d array of momenta, found {len(momenta.shape)}"
assert momenta.shape[1] == 3, f"Momenta must be 3 vectors, found {momenta.shape[1]}-vectors."
assert len(momenta) == len(spin_irreps), "Number of momenta must equal number of particles."

# Reiterate what was found
if args.verbose:
    print("Constructing block diagonalization operators:")
    print("Irrep, Momentum")
    print("-"*20)
    for irrep, k in zip(spin_irreps, momenta):
        print(irrep, k)
    if internal_symmetry is None:
        print("No internal-symmetry projector specified")
    else:
        print("Internal-symmetry projector")
        print("-"*20)
        print("Term, Weight, Permutation")
        for idx, (weight, perm) in enumerate(internal_symmetry):
            print(idx+1, weight, perm)

# Do the calculation
print_verbose(args.verbose, "Starting calculation.")
result = mhi.mhi(momenta, spin_irreps, internal_symmetry=internal_symmetry, verbose=args.verbose)
print_verbose(args.verbose, "Finished calculation.")

# Save the results
print_verbose(args.verbose, "Writing results to",  args.ofile)
if args.clobber and os.path.exists(args.ofile):
    os.remove(args.ofile)
mhi.write_hdf5(args.ofile, result.decomp)
print_verbose(args.verbose, "Writing complete.")
sys.exit(0)
