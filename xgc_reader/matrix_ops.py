"""Matrix operations and analysis classes."""

import numpy as np
import adios2

try:
    from scipy.sparse import csr_matrix
except ImportError:
    csr_matrix = None
    print("Warning: scipy not available. Matrix operations will not work.")


class xgc_mat(object):
    """Base class for XGC matrix operations."""
    
    def create_sparse_xgc(self, nelement, eindex, value, m=None, n=None):
        """
        Create Python sparse matrix from XGC data.
        
        Parameters
        ----------
        nelement : array_like
            Number of elements per row
        eindex : array_like
            Element indices
        value : array_like
            Element values
        m : int, optional
            Number of rows
        n : int, optional
            Number of columns
            
        Returns
        -------
        spmat : scipy.sparse.csr_matrix
            Sparse matrix
        """
        if csr_matrix is None:
            raise ImportError("scipy is required for sparse matrix operations")
        
        if m is None: 
            m = nelement.size
        if n is None: 
            n = nelement.size
        
        # Format for Python sparse matrix
        indptr = np.insert(np.cumsum(nelement), 0, 0)
        indices = np.empty((indptr[-1],))
        data = np.empty((indptr[-1],))
        
        for i in range(nelement.size):
            indices[indptr[i]:indptr[i+1]] = eindex[i, 0:nelement[i]]
            data[indptr[i]:indptr[i+1]] = value[i, 0:nelement[i]]
        
        # Create sparse matrix
        spmat = csr_matrix((data, indices, indptr), shape=(m, n))
        return spmat


class grad_rz(xgc_mat):
    """
    Gradient operation class for (R,Z) or (psi,theta) coordinate systems.
    """
    
    def __init__(self, path):
        """
        Initialize gradient matrices from xgc.grad_rz.bp file.
        
        Parameters
        ----------
        path : str
            Path to XGC data files
        """
        try:
            with adios2.FileReader(path + "xgc.grad_rz.bp") as f:
                # Flag indicating whether gradient is (R,Z) or (psi,theta)
                self.mat_basis = f.read('basis')

                # Set up matrix for psi/R derivative
                nelement = f.read('nelement_r')
                eindex = f.read('eindex_r') - 1  # Convert to 0-based indexing
                value = f.read('value_r')
                nrows = f.read('m_r')
                ncols = f.read('n_r')
                self.mat_psi_r = self.create_sparse_xgc(nelement, eindex, value, m=nrows, n=ncols)

                # Set up matrix for theta/Z derivative
                nelement = f.read('nelement_z')
                eindex = f.read('eindex_z') - 1  # Convert to 0-based indexing
                value = f.read('value_z')
                nrows = f.read('m_z')
                ncols = f.read('n_z')
                self.mat_theta_z = self.create_sparse_xgc(nelement, eindex, value, m=nrows, n=ncols)
                
        except Exception as e:
            print(f"Warning: Could not load gradient matrices: {e}")
            self.mat_psi_r = None
            self.mat_theta_z = None
            self.mat_basis = 0
    
    def apply_gradient(self, field, component='both'):
        """
        Apply gradient operation to field.
        
        Parameters
        ----------
        field : array_like
            Input field
        component : str, optional
            Which gradient component ('r', 'z', or 'both')
            
        Returns
        -------
        grad_field : array_like or tuple
            Gradient of field
        """
        if self.mat_psi_r is None or self.mat_theta_z is None:
            raise ValueError("Gradient matrices not loaded")
        
        if component == 'r' or component == 'psi':
            return self.mat_psi_r @ field
        elif component == 'z' or component == 'theta':
            return self.mat_theta_z @ field
        elif component == 'both':
            grad_r = self.mat_psi_r @ field
            grad_z = self.mat_theta_z @ field
            return grad_r, grad_z
        else:
            raise ValueError("component must be 'r', 'z', or 'both'")


class ff_mapping(xgc_mat):
    """
    Field-following mapping class for coordinate transformations.
    """
    
    def __init__(self, ff_name, path):
        """
        Initialize field-following mapping from xgc.ff_*.bp file.
        
        Parameters
        ----------
        ff_name : str
            Name of field-following mapping
        path : str
            Path to XGC data files
        """
        try:
            fn = 'xgc.ff_' + ff_name + '.bp'
            with adios2.FileReader(path + fn) as f:
                nelement = f.read('nelement')
                eindex = f.read('eindex') - 1  # Convert to 0-based indexing
                value = f.read('value')
                nrows = f.read('nrows')
                ncols = f.read('ncols')
                dl_par = f.read('dl_par')
                
                self.mat = self.create_sparse_xgc(nelement, eindex, value, m=nrows, n=ncols)
                self.dl = dl_par
                self.name = ff_name
                
        except Exception as e:
            print(f"Warning: Could not load field-following mapping {ff_name}: {e}")
            self.mat = None
            self.dl = None
            self.name = ff_name
    
    def apply_mapping(self, field):
        """
        Apply field-following mapping to field.
        
        Parameters
        ----------
        field : array_like
            Input field
            
        Returns
        -------
        mapped_field : array_like
            Field after mapping
        """
        if self.mat is None:
            raise ValueError(f"Field-following mapping {self.name} not loaded")
        
        return self.mat @ field


def load_grad_rz(xgc_instance):
    """
    Load gradient R-Z matrices.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance
        
    Returns
    -------
    grad : grad_rz
        Gradient operation object
    """
    path = xgc_instance.path if hasattr(xgc_instance, 'path') else './'
    return grad_rz(path)


def load_ff_mapping(xgc_instance):
    """
    Load field-following mapping matrices.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance
        
    Returns
    -------
    mappings : dict
        Dictionary of field-following mapping objects
    """
    path = xgc_instance.path if hasattr(xgc_instance, 'path') else './'
    map_names = ["1dp_fwd", "1dp_rev", "hdp_fwd", "hdp_rev"]
    mappings = {}
    
    for ff_name in map_names:
        try:
            mappings[ff_name] = ff_mapping(ff_name, path)
        except Exception as e:
            print(f"Warning: Could not load mapping {ff_name}: {e}")
            mappings[ff_name] = None
    
    return mappings


def convert_3d_grad_all(xgc_instance, field):
    """
    Convert field into gradient representation (placeholder implementation).
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance
    field : array_like
        Input field
        
    Returns
    -------
    grad_field : array_like
        Gradient representation of field
    """
    # This is a simplified placeholder - full implementation would require
    # detailed field-following coordinate transformations
    print("Warning: convert_3d_grad_all is a placeholder implementation")
    
    if not hasattr(xgc_instance, 'grad') or xgc_instance.grad is None:
        raise ValueError("Gradient matrices not loaded. Call load_grad_rz() first.")
    
    # Simple gradient calculation
    if hasattr(xgc_instance.grad, 'apply_gradient'):
        return xgc_instance.grad.apply_gradient(field)
    else:
        return field  # Fallback


def conv_real2ff(xgc_instance, field):
    """
    Convert real space field to field-following representation.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with field-following mapping
    field : array_like
        Input field (2D or 3D)
        
    Returns
    -------
    field_ff : array_like
        Field in field-following representation
    """
    if not hasattr(xgc_instance, 'ff_hdp_rev') or not hasattr(xgc_instance, 'ff_hdp_fwd'):
        raise ValueError("Field-following mapping not loaded. Call load_ff_mapping() first.")
    
    if field.ndim == 3:
        field_work = field
    elif field.ndim == 2:
        field_work = np.zeros((field.shape[0], field.shape[1], 1), dtype=field.dtype)
        field_work[:, :, 0] = field[:, :]
    else:
        print("conv_real2ff: input field has wrong shape.")
        return -1
    
    fdim = field_work.shape[2]
    nphi = field_work.shape[1]
    field_ff = np.zeros((field_work.shape[0], nphi, fdim, 2), dtype=field_work.dtype)
    
    for iphi in range(nphi):
        iphi_l = iphi - 1 if iphi > 0 else nphi - 1
        iphi_r = iphi
        for j in range(fdim):
            field_ff[:, iphi, j, 0] = xgc_instance.ff_hdp_rev.mat.dot(field_work[:, iphi_l, j])
            field_ff[:, iphi, j, 1] = xgc_instance.ff_hdp_fwd.mat.dot(field_work[:, iphi_r, j])
    
    field_ff = np.transpose(field_ff, (1, 0, 2, 3))
    if fdim == 1:
        field_ff = (np.transpose(field_ff, (0, 1, 3, 2)))[:, :, :]
        
    return field_ff


def GradPlane(xgc_instance, field):
    """
    Calculate plane gradient of field.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with gradient matrices
    field : array_like
        Input field (1D or 2D)
        
    Returns
    -------
    grad_field : array_like
        Gradient field
    """
    if not hasattr(xgc_instance, 'grad') or xgc_instance.grad is None:
        raise ValueError("Gradient matrices not loaded. Call load_grad_rz() first.")
    
    if field.ndim > 2:
        print("GradPlane: Wrong array shape of field, must be (nnode,nphi) or (nnode)")
        return -1
    
    nnode = field.shape[0]
    if field.ndim == 2:
        field_loc = field
        nphi = field.shape[1]
    else:
        nphi = 1
        field_loc = np.zeros((nnode, nphi), dtype=field.dtype)
        field_loc[:, 0] = field
    
    grad_field = np.zeros((nnode, nphi, 2), dtype=field.dtype)
    for iphi in range(nphi):
        grad_field[:, iphi, 0] = xgc_instance.grad.mat_psi_r.dot(field_loc[:, iphi])
        grad_field[:, iphi, 1] = xgc_instance.grad.mat_theta_z.dot(field_loc[:, iphi])
    
    return grad_field


def GradParX(xgc_instance, field):
    """
    Calculate parallel gradient of field.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with field-following data
    field : array_like
        Input field (must be 2D: nnode, nphi)
        
    Returns
    -------
    bdotgrad_field : array_like
        Parallel gradient field
    """
    if field.ndim != 2:
        print("GradParX: Wrong array shape of field, must be (nnode,nphi)", field.shape)
        return -1
    
    nphi = field.shape[1]
    nnode = field.shape[0]
    
    if not hasattr(xgc_instance, 'ff_1dp_fwd') or xgc_instance.ff_1dp_fwd.mat is None:
        raise ValueError("1D field-following mapping not loaded")
    
    if nnode != xgc_instance.ff_1dp_fwd.mat.shape[0]:
        return -1
    
    sgn = np.sign(xgc_instance.bfield[2, 0])  # toroidal field at the magnetic axis
    l_l = xgc_instance.ff_1dp_rev.dl
    l_r = xgc_instance.ff_1dp_fwd.dl
    l_tot = l_r + l_l
    bdotgrad_field = np.zeros_like(field)
    
    for iphi in range(nphi):
        iphi_l = iphi - 1 if iphi > 0 else nphi - 1
        iphi_r = np.fmod((iphi + 1), nphi)
        
        field_l = xgc_instance.ff_1dp_rev.mat.dot(field[:, iphi_l])
        field_r = xgc_instance.ff_1dp_fwd.mat.dot(field[:, iphi_r])
        
        bdotgrad_field[:, iphi] = sgn * (field_r - field_l) / l_tot
    
    return bdotgrad_field


def write_dAs_ff_for_poincare(xgc_instance, fnum):
    """
    Write field-following vector potential for Poincare analysis.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance
    fnum : int
        File number
    """
    # Load As
    fn = 'xgc.3d.%5.5d.bp' % fnum
    with adios2.FileReader(fn) as f:
        As = f.read('apars').transpose()
        print('Read As[%d,%d] from ' % (As.shape[0], As.shape[1]) + fn)
        print('As', As.shape)
        
        # Calculate grad(As) and transform As and grad(As) to field-following representation
        dAs = convert_3d_grad_all(xgc_instance, As)
        print('dAs', dAs.shape)
        
        As_phi_ff = conv_real2ff(xgc_instance, As)
        print('As_phi_ff', As_phi_ff.shape)
        
        dAs_phi_ff = -conv_real2ff(xgc_instance, dAs)
        print('dAs_phi_ff', dAs_phi_ff.shape)
        
        # Write ADIOS file with perturbed vector potential in field-following representation
        print("Warning: ADIOS output writing not fully implemented")


def create_sparse_xgc(nelement, eindex, value, m=None, n=None):
    """Create Python sparse matrix from XGC data."""
    from scipy.sparse import csr_matrix
    
    if m is None: 
        m = nelement.size
    if n is None: 
        n = nelement.size
        
    # Create sparse matrix - eindex is 1-based in XGC, convert to 0-based
    return csr_matrix((value, (eindex[:, 0] - 1, eindex[:, 1] - 1)), shape=(m, n))


# Add this method to xgc_mat class
xgc_mat.create_sparse_xgc = staticmethod(create_sparse_xgc)