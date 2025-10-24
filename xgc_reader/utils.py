"""Utility functions for XGC reader."""

import numpy as np
import adios2
import time
import functools


def read_all_steps(f, var):
    """Read all time steps for a variable from ADIOS2 file (optimized)."""
    vars = f.available_variables()
    stc = vars[var].get("AvailableStepsCount")
    ct = vars[var].get("Shape")
    stc = int(stc)
    
    if ct != '':
        c = [int(i) for i in ct.split(',')]
        # Use more efficient reading for common cases
        if len(c) == 1:
            data = f.read(var, start=[0], count=c, step_selection=[0, stc])
            return data.reshape([stc, c[0]])
        elif len(c) == 2:
            data = f.read(var, start=[0, 0], count=c, step_selection=[0, stc])
            return data.reshape([stc, c[0], c[1]])
        elif len(c) == 3:
            data = f.read(var, start=[0, 0, 0], count=c, step_selection=[0, stc])
            return data.reshape([stc, c[0], c[1], c[2]])
    else:
        return f.read(var, step_selection=[0, stc])


def check_adios2_version():
    """Check that ADIOS2 version is compatible."""
    adios2_version_minor = int(adios2.__version__[2:adios2.__version__.find('.', 2)])
    if adios2_version_minor < 10:
        raise RuntimeError(f"Must use adios 2.10 or newer with the xgc_reader module, loaded 2.{adios2_version_minor}\n For 2.9.x version try adios_2_9_x branch")


def timing_decorator(func):
    """Decorator to time function execution for performance monitoring."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        if execution_time > 0.1:  # Only print for functions taking > 0.1s
            print(f"⏱️  {func.__name__} executed in {execution_time:.3f}s")
        return result
    return wrapper


def validate_array_input(var, var_name="variable"):
    """
    Validate array input for calculations with performance considerations.
    
    Parameters
    ----------
    var : array_like
        Input array to validate
    var_name : str, optional
        Name of variable for error messages
        
    Returns
    -------
    var : numpy.ndarray
        Validated numpy array
        
    Raises
    ------
    ValueError
        If input is invalid
    """
    if var is None:
        raise ValueError(f"{var_name} cannot be None")
    
    # Convert to numpy array if not already
    if not isinstance(var, np.ndarray):
        var = np.asarray(var)
    
    if var.size == 0:
        raise ValueError(f"{var_name} cannot be empty")
    
    # Quick check for common issues
    if not np.isfinite(var).all():
        n_invalid = np.sum(~np.isfinite(var))
        print(f"Warning: {var_name} contains {n_invalid} non-finite values")
    
    return var


def optimize_memory_usage():
    """Provide memory usage optimization tips."""
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        
        if mem_mb > 1000:  # > 1GB
            print(f"💾 Current memory usage: {mem_mb:.1f}MB")
            print("💡 Consider processing data in chunks for large datasets")
        
        return mem_mb
    except ImportError:
        return None


def adios2_get_shape(f, varname):
    """
    Get shape and step information for ADIOS2 variable.
    
    Parameters
    ----------
    f : adios2.FileReader
        ADIOS2 file reader
    varname : str
        Variable name
        
    Returns
    -------
    nstep : int
        Number of time steps
    lshape : tuple
        Shape of the variable
    """
    nstep = int(f.available_variables()[varname]['AvailableStepsCount'])
    shape = f.available_variables()[varname]['Shape']
    
    if shape == '':
        # Accessing Adios1 file - read data and figure out shape
        v = f.read(varname)
        lshape = v.shape
    else:
        lshape = tuple([int(xx.strip(',')) for xx in shape.strip().split()])
    
    return nstep, lshape


def adios2_read_all_time(f, varname):
    """
    Read all time steps for a variable from ADIOS2 file.
    
    Parameters
    ----------
    f : adios2.FileReader
        ADIOS2 file reader
    varname : str
        Variable name
        
    Returns
    -------
    data : array_like
        Data for all time steps
    """
    nstep, nsize = adios2_get_shape(f, varname)
    
    # Handle different dimensionalities
    if len(nsize) == 1:
        return np.squeeze(f.read(varname, start=(0,), count=nsize, step_start=0, step_count=nstep))
    elif len(nsize) == 2:
        return np.squeeze(f.read(varname, start=(0, 0), count=nsize, step_start=0, step_count=nstep))
    elif len(nsize) == 3:
        return np.squeeze(f.read(varname, start=(0, 0, 0), count=nsize, step_start=0, step_count=nstep))
    else:
        # Fallback for higher dimensions
        start = tuple([0] * len(nsize))
        return np.squeeze(f.read(varname, start=start, count=nsize, step_start=0, step_count=nstep))


def adios2_read_one_time(f, varname, step=-1):
    """
    Read one time step for a variable from ADIOS2 file.
    
    Parameters
    ----------
    f : adios2.FileReader
        ADIOS2 file reader
    varname : str
        Variable name
    step : int, optional
        Time step to read (-1 for last step)
        
    Returns
    -------
    data : array_like
        Data for specified time step
    """
    nstep, nsize = adios2_get_shape(f, varname)
    
    if step == -1:
        step = nstep - 1  # Use last step
    
    # Find the correct step
    idx = 0
    for f1 in f:
        if idx == step:
            break
        idx += 1
    
    # Read data for specified step
    if len(nsize) == 1:
        return np.squeeze(f1.read(varname, start=(0,), count=nsize, step_start=step, step_count=1))
    elif len(nsize) == 2:
        return np.squeeze(f1.read(varname, start=(0, 0), count=nsize, step_start=step, step_count=1))
    elif len(nsize) == 3:
        return np.squeeze(f1.read(varname, start=(0, 0, 0), count=nsize, step_start=step, step_count=1))
    elif len(nsize) == 4:
        return np.squeeze(f1.read(varname, start=(0, 0, 0, 0), count=nsize, step_start=step, step_count=1))
    else:
        # Fallback for higher dimensions
        start = tuple([0] * len(nsize))
        return np.squeeze(f1.read(varname, start=start, count=nsize, step_start=step, step_count=1))


def read_one_ad2_var(filestr, varstr, with_time=False):
    """
    Read one variable from ADIOS2 file with optional time.
    
    Parameters
    ----------
    filestr : str
        File path
    varstr : str
        Variable name
    with_time : bool, optional
        Whether to also read time variable
        
    Returns
    -------
    var : array_like
        Variable data
    time : array_like, optional
        Time data (if with_time=True)
    """
    with adios2.FileReader(filestr) as f:
        var = f.read(varstr)
        
    if with_time:
        try:
            time = f.read('time')
            return var, time
        except:
            return var, 0
    else:
        return var