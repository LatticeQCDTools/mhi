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
particles = inputs['particles']

# Verify consistency of physics inputs
assert len(momenta.shape) == 2, f"Expected 2d array of momenta, found {len(momenta.shape)}"
assert momenta.shape[1] == 3, f"Momenta must be 3 vectors, found {momenta.shape[1]}-vectors."
assert len(momenta) == len(particles), "Number of momenta must equal number of particles."

# Reiterate what was found
if args.verbose:
    print("Constructing block diagonalization operators:")
    print("Name, Irrep, Momentum")
    print("-"*20)
    for (name, irrep), k in zip(particles, momenta):
        print(name, irrep, k)

# Do the calculation
print_verbose(args.verbose, "Starting calculation.")
proj = mhi.mhi(momenta, particles, verbose=args.verbose)
print_verbose(args.verbose, "Finihsed calculation.")

# Save the results

print_verbose(args.verbose, "Writing results to",  args.ofile)
mhi.write_hdf5(args.ofile, proj)
print_verbose(args.verbose, "Writing complete.")

sys.exit(0)
