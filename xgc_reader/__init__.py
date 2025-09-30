"""
XGC Reader - Modular package for XGC simulation data analysis.

This package provides a comprehensive toolkit for reading and analyzing 
X-point Gyrokinetic Code (XGC) simulation data with improved modularity,
performance, and maintainability.

Key Features:
- Modular architecture with specialized analysis modules
- Optimized performance for large datasets  
- Comprehensive error handling and validation
- Full backward compatibility with original XGC reader
- Extensive documentation and examples

Basic Usage:
    import xgc_reader
    x = xgc_reader.xgc1('/path/to/xgc/data/')
    x.load_basic()  # Load units, 1D data, mesh, volumes
    x.print_plasma_info()  # Display plasma parameters
"""

from .base import xgc1
from .constants import cnst
from .utils import read_all_steps, check_adios2_version

__version__ = "1.0.0"
__author__ = "XGC Development Team"
__license__ = "MIT"

# Main interface - expose xgc1 class at package level
__all__ = ['xgc1', 'cnst', 'read_all_steps', 'check_adios2_version']


def optimize_performance():
    """Enable performance optimizations and show tips for large datasets."""
    from .utils import optimize_memory_usage
    print("🚀 XGC Reader Performance Optimization")
    print("=" * 40)
    mem_usage = optimize_memory_usage()
    if mem_usage is None:
        print("💾 Memory monitoring: Install 'psutil' for memory tracking")
    print("💡 Tips for better performance:")
    print("   • Use load_basic() for initial data loading")
    print("   • Process large 3D data in time chunks") 
    print("   • Vectorized operations are optimized")
    print("   • Monitor memory usage for large datasets")


def show_modules():
    """Display available XGC Reader modules and their functions."""
    modules = {
        'Core': ['base', 'constants', 'utils'],
        'Data Loading': ['oned_data', 'mesh_data', 'volume_data', 'field_data', 'flux_data'],
        'Analysis': ['geometry', 'analysis', 'heat_diagnostics'],
        'Visualization': ['plotting', 'report'],
        'Advanced': ['matrix_ops']
    }
    
    print("📚 XGC Reader Module Architecture")
    print("=" * 40)
    for category, module_list in modules.items():
        print(f"{category:15}: {', '.join(module_list)}")
    
    total_modules = sum(len(mods) for mods in modules.values())
    print(f"\n✨ Total: {total_modules} specialized modules")
    print("💡 All functions accessible through main xgc1 class")