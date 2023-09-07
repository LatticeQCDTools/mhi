"""
Example script illustrating how to read results.
"""
import sys
import os
import h5py
import numpy as np
from mhi import mhi

def main():

    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <path/to/output.h5>")
        sys.exit(0)
    h5fname = sys.argv[1]

    if not h5fname.endswith(('h5', 'hdf5')):
        print("File must end with suffix 'h5' or 'hdf5")
        sys.exit(0)

    if not os.path.exists(h5fname):
        print("File not found", h5fname)
        sys.exit(0)

    result = mhi.read_hdf5(h5fname)
    for key, val in result.items():
        print(key, val.shape)

if __name__ == '__main__':
    main()
