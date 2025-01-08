import sys
sys.path.append('/global/homes/s/sku/git/XGC_reader')
import argparse
import matplotlib.pyplot as plt
import xgc_reader

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot AD2 variable contour')
    parser.add_argument('filename', help='Input AD2 file path')
    parser.add_argument('varname', help='Variable name to plot')
    parser.add_argument('--plane', type=int, default=0, help='Plane index (default: 0)')
    parser.add_argument('--output', help='Output PNG filename', default='output.png')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for saved image')
    parser.add_argument('--noshow', action='store_true', help='Do not display plot')
    parser.add_argument('--figwidth', type=float, default=10.0, help='Figure width in inches')
    parser.add_argument('--figheight', type=float, default=8.0, help='Figure height in inches')
 
    args = parser.parse_args()

    x=xgc_reader.xgc1()
    x.load_unitsm()
    x.setup_mesh()

    # Read variable
    var = x.read_one_ad2_var(args.filename, args.varname)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(args.figwidth, args.figheight))
    
    # Plot contour
    if(var.ndim == 2):
        var=var[args.plane,:]

    x.contourf_one_var(fig, ax, var)
    
    # Add title
    ax.set_title(f'{args.varname} at plane {args.plane} of {args.filename}')
    
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    
    if not args.noshow:
        plt.show()

if __name__ == '__main__':
    main()
