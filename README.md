# mhi
Irrep Projection for Multi-Hadron Interpolators

## Installation

After cloning the repo, move the top-level directory containing setup.py.
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

## TODO

There are some pending TODO items:
* Address lingering TODO flags in the code
* Finalize the "driver function" mhi.mhi(...), checking for correctness,
  generality, and consistency with the final algorithm as described in the paper.
* Add "group conjugations" to transform little groups into canonical form
