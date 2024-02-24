# mhi
Irrep Projection for Multi-Hadron Interpolators

This package is associated with the paper

    W. Detmold, W. I. Jay, G. Kanwar, P. E. Shanahan, and M. L. Wagman,
    "Multi-particle interpolating operators in quantum field theories with cubic symmetries."

## Installation

After cloning the repo, move to the top-level directory containing setup.py.
Then use pip to install the module:

`$ pip install -e ./`

The flag "-e" is an "editable install", which means that local changes to the
code will appear code is run.

## Tests

To test the command-line interface, run the command

`$ python -m mhi example_input.yaml <output_name.h5> --verbose`

The flag "-m" is necessary and directs python to run the module's main function.
After running the main function, it is possible to test the ability to read the hdf5 file using

`$ python read.py <output_name.h5>`

To test the module against previously tabulated data, run the command

`$ python test.py <path/to/reference/data>`
