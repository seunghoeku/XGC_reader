import numpy as np


# Merge two functions
# x_in and x_out - linear transition boundaries
# psi_e, var_e - target edge profiles
# psi_p, var_p - target core profiles

def merge(x_in, x_out, psi_e, var_e, psi_p, var_p):
    msk = np.nonzero(np.logical_and(x_in < psi_e, psi_e < x_out))
    msk_out = np.nonzero(psi_e >= x_out)
    psi = psi_e
    var_p2 = np.interp(psi_e, psi_p, var_p)
    var = np.copy(var_p2)
    var[msk_out] = var_e[msk_out]
    a = (psi[msk] - x_in) / (x_out - x_in)
    var[msk] = a * var_e[msk] + (1 - a) * var_p2[msk]
    return var


# three functions for plotting

def autoscale(ax=None, axis='y', margin=0.1):
    """Autoscales the x or y axis of a matplotlib Axes object."""
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    newlow, newhigh = np.inf, -np.inf

    for artist in ax.collections + ax.lines:
        x, y = get_xy(artist)
        if axis == 'y':
            setlim = ax.set_ylim
            lim = ax.get_xlim()
            fixed, dependent = x, y
        else:
            setlim = ax.set_xlim
            lim = ax.get_ylim()
            fixed, dependent = y, x

        low, high = calculate_new_limit(fixed, dependent, lim)
        newlow = low if low < newlow else newlow
        newhigh = high if high > newhigh else newhigh

    margin = margin * (newhigh - newlow)
    setlim(newlow - margin, newhigh + margin)


def calculate_new_limit(fixed, dependent, limit):
    """Calculates dependent-axis min/max within the visible window."""
    if len(fixed) > 2:
        mask = (fixed > limit[0]) & (fixed < limit[1])
        window = dependent[mask]
        low, high = window.min(), window.max()
    else:
        low = dependent[0]
        high = dependent[-1]
        if low == 0.0 and high == 1.0:
            low = np.inf
            high = -np.inf
    return low, high


def get_xy(artist):
    """Gets x/y coordinates of a matplotlib artist."""
    if "Collection" in str(artist):
        x, y = artist.get_offsets().T
    elif "Line" in str(artist):
        x, y = artist.get_xdata(), artist.get_ydata()
    else:
        raise ValueError("This type of object isn't implemented yet")
    return x, y
