import numpy as np

# Load a profile file - wapper for load_prf2
def load_prf(filename):
    psi, var = load_prf2(filename)
    return psi, var

# Load a profile file.
# Returns psi, var
def load_prf2(filename):
    with open(filename, 'r') as file:
        #read dimensions
        [n] = map(int, file.readline().strip().split())
        #allocate array
        psi=np.zeros(n)
        var=np.zeros(n)

        for l in range(n):
            [psi[l], var[l]]=map(float, file.readline().strip().split() )

        #read end flag
        [end_flag]=map(int, file.readline().strip().split())
        if(end_flag!=-1):
            print('Error: end flag is not -1. end_flag= %d'%end_flag)
        return(psi,var)

# Save a profile file 
def save_prf(x, y, fname):
    with open(fname, "w") as f:
        sz=np.size(x)        
        f.write("%d\n"%sz)
        for i in range(sz):
            f.write("%19.13e  %19.13e\n"%(x[i],y[i]))
        f.write("-1\n")

# three functions related with Kinetic EFIT p-file
def read_kefit_profile(pfilename):
    """
    Read kinetic EFIT p-file containing multiple profiles and ion species information.
    
    Parameters:
    -----------
    pfilename : str
        Path to the p-file to be read
        
    Returns:
    --------
    profiles : list of dictionaries
        Each dictionary contains information about a profile:
        - 'num_points': Number of points in the profile
        - 'name': Profile name (e.g., 'ne', 'te')
        - 'units': Units of the profile (e.g., '10^20/m^3', 'KeV')
        - 'gradient_label': Label for the gradient (e.g., 'dne/dpsiN')
        - 'data': numpy array with shape (3, num_points) containing:
            - data[0,:]: psinorm values
            - data[1,:]: profile values
            - data[2,:]: gradient values
    
    ion_species : dict or None
        If ion species information is present in the file:
        - 'count': Number of ion species
        - 'description': Description string (e.g., 'N Z A of ION SPECIES')
        - 'data': numpy array with shape (count, 3) containing N, Z, A values for each species
    """
    profiles = []
    ion_species = None
    
    with open(pfilename, 'r') as file:
        lines = file.readlines()
    
    line_index = 0
    
    while line_index < len(lines):
        line = lines[line_index].strip()
        if not line:  # Skip empty lines
            line_index += 1
            continue
            
        # Try to parse the header line
        try:
            parts = line.split()
            if len(parts) >= 1 and parts[0].isdigit():
                num_points = int(parts[0])
                
                # Check if this is ion species data
                if len(parts) >= 5 and parts[1] == "N" and parts[2] == "Z" and parts[3] == "A":
                    # This is ion species data
                    ion_species = {
                        'count': num_points,
                        'description': ' '.join(parts[1:])
                    }
                    
                    # Read the ion species data
                    species_data = np.zeros((num_points, 3))
                    for i in range(num_points):
                        if line_index + 1 + i < len(lines):
                            data_line = lines[line_index + 1 + i].strip().split()
                            if len(data_line) >= 3:
                                species_data[i, 0] = float(data_line[0])  # N
                                species_data[i, 1] = float(data_line[1])  # Z
                                species_data[i, 2] = float(data_line[2])  # A
                    
                    ion_species['data'] = species_data
                    line_index += num_points + 1
                    
                else:
                    # Normal profile data
                    profile_info = {}
                    profile_info['num_points'] = num_points
                    
                    # Parse the rest of the header line for profile name and units
                    if len(parts) >= 3:
                        profile_info['name'] = parts[1]  # e.g., "psinorm"
                        
                        # Extract name and units from the second part
                        # e.g., "ne(10^20/m^3)" -> name="ne", units="10^20/m^3"
                        second_part = parts[2]
                        if '(' in second_part and ')' in second_part:
                            name, units = second_part.split('(')
                            units = units.split(')')[0]
                            profile_info['name'] = name
                            profile_info['units'] = units
                        else:
                            profile_info['name'] = second_part
                            profile_info['units'] = ''
                            
                        # Get gradient label if available
                        if len(parts) >= 4:
                            profile_info['gradient_label'] = parts[3]
                        else:
                            profile_info['gradient_label'] = ''
                    
                    # Read the data points
                    data = np.zeros((3, num_points))
                    for i in range(num_points):
                        if line_index + 1 + i < len(lines):
                            data_line = lines[line_index + 1 + i].strip().split()
                            if len(data_line) >= 3:
                                data[0, i] = float(data_line[0])  # psinorm
                                data[1, i] = float(data_line[1])  # profile value
                                data[2, i] = float(data_line[2])  # gradient
                    
                    profile_info['data'] = data
                    profiles.append(profile_info)
                    
                    # Move to the next profile
                    line_index += num_points + 1
            else:
                line_index += 1
        except Exception as e:
            print(f"Error parsing line {line_index}: {e}")
            line_index += 1
    
    return profiles, ion_species

def plot_profiles(profiles):
    """
    Plot the profiles read from the p-file.
    
    Parameters:
    -----------
    profiles : list of dictionaries
        The profiles returned by read_kefit_profile
    """
    try:
        import matplotlib.pyplot as plt
        
        for i, profile in enumerate(profiles):
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(profile['data'][0], profile['data'][1])
            plt.xlabel('psinorm')
            plt.ylabel(f"{profile['name']} ({profile['units']})")
            plt.title(f"Profile {i+1}: {profile['name']}")
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(profile['data'][0], profile['data'][2])
            plt.xlabel('psinorm')
            plt.ylabel(profile['gradient_label'])
            plt.title(f"Gradient of {profile['name']}")
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
    except ImportError:
        print("Matplotlib is not installed. Cannot plot profiles.")

def display_ion_species(ion_species):
    """
    Display information about ion species.
    
    Parameters:
    -----------
    ion_species : dict
        Ion species information returned by read_kefit_profile
    """
    if ion_species is None:
        print("No ion species information in file.")
        return
        
    print(f"Ion Species Information ({ion_species['description']}):")
    print(f"Number of species: {ion_species['count']}")
    
    print("    N      Z      A")
    print("----------------------")
    for i in range(ion_species['count']):
        print(f"{ion_species['data'][i, 0]:6.2f} {ion_species['data'][i, 1]:6.2f} {ion_species['data'][i, 2]:10.6f}")



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

# three functions for plotting
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

