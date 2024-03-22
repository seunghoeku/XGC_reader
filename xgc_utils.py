import numpy as np

def load_prf(filename):
    import pandas as pd
    #space separated file 
    df = pd.read_csv(filename, sep = r'\s{2,}',engine='python')
    psi = df.index.values
    psi = psi[0:-1]
    var = df.values
    var = var[0:-1]
    return(psi,var)

def save_prf(x, y, fname):
    with open(fname, "w") as f:
        sz=np.size(x)        
        f.write("%d\n"%sz)
        for i in range(sz):
            f.write("%19.13e  %19.13e\n"%(x[i],y[i]))
        f.write("-1\n")

# Merge two functions
# x_in and x_out - linear transition boundaries
# psi_e, var_e - target edge profiles
# psi_p, var_p - target core porfiles

def merge(x_in, x_out, psi_e, var_e, psi_p, var_p):
    msk=np.nonzero(np.logical_and(x_in < psi_e, psi_e < x_out))
    msk_out = np.nonzero(psi_e >= x_out)
    psi=psi_e # psi mesh is the same as edge profiles
    var_p2= np.interp(psi_e,psi_p, var_p) #core profiles on psi_e grid
    var=np.copy(var_p2)
    var[msk_out]=var_e[msk_out] #outside of x_out has edge profile values
    a = (psi[msk]-x_in)/(x_out-x_in)
    var[msk]=a*var_e[msk] + (1-a) * var_p2[msk] #linear transition 
    return var


def autoscale(ax=None, axis='y', margin=0.1):
    '''Autoscales the x or y axis of a given matplotlib ax object
    to fit the margins set by manually limits of the other axis,
    with margins in fraction of the width of the plot

    Defaults to current axes object if not specified.
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        ax = plt.gca()
    newlow, newhigh = np.inf, -np.inf

    for artist in ax.collections + ax.lines:
        x,y = get_xy(artist)
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

    margin = margin*(newhigh - newlow)

    setlim(newlow-margin, newhigh+margin)

def calculate_new_limit(fixed, dependent, limit):
    '''Calculates the min/max of the dependent axis given 
    a fixed axis with limits
    '''
    if len(fixed) > 2:
        mask = (fixed>limit[0]) & (fixed < limit[1])
        window = dependent[mask]
        low, high = window.min(), window.max()
    else:
        low = dependent[0]
        high = dependent[-1]
        if low == 0.0 and high == 1.0:
            # This is a axhline in the autoscale direction
            low = np.inf
            high = -np.inf
    return low, high

def get_xy(artist):
    '''Gets the xy coordinates of a given artist
    '''
    if "Collection" in str(artist):
        x, y = artist.get_offsets().T
    elif "Line" in str(artist):
        x, y = artist.get_xdata(), artist.get_ydata()
    else:
        raise ValueError("This type of object isn't implemented yet")
    return x, y



