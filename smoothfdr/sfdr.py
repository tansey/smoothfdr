import matplotlib as mpl
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import argparse
import csv
import sys
from scipy.sparse import csc_matrix, dia_matrix, linalg as sla
from scipy.stats import norm
from smoothed_fdr import SmoothedFdr, GaussianKnown, calc_plateaus
from normix import GridDistribution, predictive_recursion, empirical_null
import signal_distributions
from utils import *
from plotutils import *


def main():
    parser = argparse.ArgumentParser(description='Runs the smoothed FDR algorithm.')

    parser.add_argument('data_file', help='The file containing the raw z-score data.')
    parser.add_argument('signal_distribution_file', help='The file location where the estimated signal distribution will be saved.')
    parser.add_argument('--verbose', type=int, default=0, help='Print detailed progress information to the console. 0=none, 1=outer-loop only, 2=all details.')
    parser.add_argument('--data_header', action='store_true', help='Specifies that there is a header line in the data file.')

    # Predictive recursion settings
    parser.add_argument('--pr_grid_x', nargs=3, type=int, default=[-7,7,57], help='The grid parameters (min, max, points) for the predictive recursion approximate distribution.')
    parser.add_argument('--pr_sweeps', type=int, default=50, help='The number of randomized sweeps to make over the data.')
    parser.add_argument('--pr_nullprob', type=float, default=1.0, help='The initial guess for the marginal probability of coming from the null distribution.')
    parser.add_argument('--pr_decay', type=float, default=-0.67, help='The exponential decay rate for the recursive update weights.')


    parser.set_defaults(data_header=False)

    # Get the arguments from the command line
    args = parser.parse_args()
    
    # Load the dataset from file
    data = np.loadtxt(args.data_file, delimiter=',', skiprows=1 if args.data_header else 0)

    if args.verbose:
        print('Estimating null distribution empirically via Efron\'s method.')

    null_mean, null_stdev = empirical_null(data.flatten())
    null_dist = GaussianKnown(null_mean, null_stdev)

    if args.verbose:
        print('Null: N({0}, {1}^2)'.format(null_mean, null_stdev))

    if args.verbose:
        print('Performing predictive recursion to estimate the signal distribution [{0}, {1}] ({2} bins)'.format(args.pr_grid_x[0], args.pr_grid_x[1], args.pr_grid_x[2]))
    
    grid_x = np.linspace(args.pr_grid_x[0], args.pr_grid_x[1], args.pr_grid_x[2])
    signal_data = data.flatten()

    pr_results = predictive_recursion(signal_data,
                             args.pr_sweeps, grid_x,
                             mu0=args.null_mean, sig0=args.null_stdev,
                             nullprob=args.pr_nullprob, decay=args.pr_decay)

    # Get the estimated distribution
    estimated_dist = GridDistribution(pr_results['grid_x'], pr_results['y_signal'])

    penalties = load_trails(args.trails)

    #TODO: Fill in the rest so there's a better lightweight script
    



















