# XGC_reader
Python script for XGC data analysis

Requires adios 2.10 or newer


### xgc_reader.py

The `xgc_reader.py` script is part of the XGC1 loader module for regenerating general plots using ADIOS2. It reads data from XGC simulations, including 1D results and other small data outputs. The script provides functions to read all steps of a variable, define constants, and handle file reading operations.

### xgc_utils.py

The purpose of `xgc_utils.py` is to provide utility functions and helper methods for XGC input data. 

### eqd_file_reader.py

The `eqd_file_reader.py` script is responsible for reading and processing EQDSK files. It contains functions to extract equilibrium data from the dictionary of EQDSK data and create an `eqd_class` object. The script refines the magnetic axis and X-point locations, sets appropriate psi values, and extracts relevant parameters such as magnetic field values and plasma current.

### geqdsk_reader.py

In `geqdsk_reader.py`, GEQDSK files are read based on the `freeqdsk` module. The script extracts additional data such as R and Z grids for psi data, magnetic axis coordinates, poloidal flux values, and other equilibrium parameters. It also provides functions to refine the magnetic axis and X-point locations using 2D cubic spline interpolation and optimization techniques.


### xgc_distribution.py

In `xgc_distribution.py`, a `VelocityGrid` class is defined to represent a velocity grid for a distribution function. The class initializes with parameters related to velocity points and ranges. It likely contains methods to handle velocity grid calculations and manipulations for distribution function analysis.


### profile_input_reader.py

The `profile_input_reader.py` script is not ready.
Use read_prf in `xgc_utils.py`




