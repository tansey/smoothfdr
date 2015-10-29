import numpy as np
import argparse
import csv
import sys
from scipy.stats import norm
from smoothed_fdr import GaussianKnown, calc_plateaus
from normix import GridDistribution, predictive_recursion, empirical_null
import signal_distributions
from utils import generate_data, ProxyDistribution
from plotutils import plot_2d

def calculate_signal_weights(width, height, default_weight, x_min, x_max, y_min, y_max, weights):
    '''Generate signal weights from the user-specified splits.'''
    signal_weights = np.zeros((width, height)) + default_weight
    for region in zip(x_min, x_max, y_min, y_max, weights):
        signal_weights[region[0]:region[1]+1,region[2]:region[3]+1] = region[4]
    return signal_weights

def main():
    parser = argparse.ArgumentParser(description='Generates a 2-dimensional grid dataset.')

    parser.add_argument('data_file', help='The location of the file where the data will be saved.')
    parser.add_argument('weights_file', help='The location of the file where the true prior weights will be saved.')
    parser.add_argument('signals_file', help='The location of the file where the underlying true signals will be saved.')
    parser.add_argument('oracle_file', help='The location of the file where the oracle posteriors will be saved.')
    parser.add_argument('edges_file', help='The location of the file where the grid graph edges will be saved.')
    parser.add_argument('trails_file', help='The location of the file where the trails will be saved.')
    
    parser.add_argument('--verbose', type=int, default=0, help='Print detailed progress information to the console. 0=none, 1=outer-loop only, 2=all details.')
    
    # Grid dimensions
    parser.add_argument('--width', type=int, default=128, help='The width of the 2d grid')
    parser.add_argument('--height', type=int, default=128, help='The height of the 2d grid')
    
    # Signal region settings
    parser.add_argument('--region_min_x', nargs='+', type=int, default=[10, 40], help='The min x locations at which the signal weight changes.')
    parser.add_argument('--region_max_x', nargs='+', type=int, default=[25, 50], help='The max x locations at which the signal weight changes.')
    parser.add_argument('--region_min_y', nargs='+', type=int, default=[10, 50], help='The min y locations at which the signal weight changes.')
    parser.add_argument('--region_max_y', nargs='+', type=int, default=[25, 60], help='The max y locations at which the signal weight changes.')
    parser.add_argument('--region_weights', nargs='+', type=float, default=[0.5, 0.8], help='The value of the signal weight for every region.')
    parser.add_argument('--default_weight', type=float, default=0.05, help='The default signal weight for any areas not in the specified regions.')
    
    # Distribution settings
    parser.add_argument('--null_mean', type=float, default=0., help='The mean of the null distribution.')
    parser.add_argument('--null_stdev', type=float, default=1., help='The variance of the null distribution.')
    parser.add_argument('--signal_mean', type=float, default=0., help='The mean of the signal distribution.')
    parser.add_argument('--signal_stdev', type=float, default=3., help='The variance of the signal distribution.')
    parser.add_argument('--signal_dist_name', help='The name of the signal distribution. This will dynamically call it by name. It must be in the signal_distributions.py file and have both the foo_pdf and foo_sample functions defined.')

    # Plot results
    parser.add_argument('--plot', help='Plot the resulting data and save to the specified file.')

    # Get the arguments from the command line
    args = parser.parse_args()

    if args.verbose:
            print 'Generating data and saving to {0}'.format(args.data_file)

    # Get the form of the signal distribution
    if args.signal_dist_name:
        signal_pdf = getattr(signal_distributions, '{0}_pdf'.format(args.signal_dist_name))
        noisy_signal_pdf = getattr(signal_distributions, '{0}_noisy_pdf'.format(args.signal_dist_name))
        signal_sample = getattr(signal_distributions, '{0}_sample'.format(args.signal_dist_name))
        signal_dist = ProxyDistribution(args.signal_dist_name, signal_pdf, signal_sample)
    else:
        signal_dist = GaussianKnown(args.signal_mean, args.signal_stdev)
        noisy_signal_pdf = signal_dist.noisy_pdf

    signal_weights = calculate_signal_weights(args.width, args.height,
                                                  args.default_weight,
                                                  args.region_min_x, args.region_max_x,
                                                  args.region_min_y, args.region_max_y,
                                                  args.region_weights)

    # Create the synthetic dataset
    data, signals = generate_data(args.null_mean, args.null_stdev, signal_dist, signal_weights)

    # Save the dataset to file
    np.savetxt(args.data_file, data, delimiter=',', fmt='%f')

    # Save the dataset to file
    np.savetxt(args.weights_file, signal_weights, delimiter=',', fmt='%f')

    # Save the truth to file
    np.savetxt(args.signals_file, signals, delimiter=',', fmt='%d')

    # Save the oracle posteriors to file
    oracle_signal_weight = signal_weights * noisy_signal_pdf(data)
    oracle_null_weight = (1-signal_weights) * norm.pdf(data, loc=args.null_mean, scale=args.null_stdev)
    oracle_posteriors = oracle_signal_weight / (oracle_signal_weight + oracle_null_weight)
    np.savetxt(args.oracle_file, oracle_posteriors, delimiter=',', fmt='%f')

    # Save the edges to file
    indices = np.arange(args.width * args.height).reshape((args.width, args.height))
    edges = np.array(list(zip(indices[:, :-1].flatten(), indices[:, 1:].flatten())) +\
                        list(zip(indices[:-1].flatten(), indices[1:].flatten())))
    np.savetxt(args.edges_file, edges, delimiter=',', fmt='%d')

    # Save the trails to file
    trails = np.array(list(indices) + list(indices.T))
    np.savetxt(args.trails_file, trails, delimiter=',', fmt='%d')

    # Plot the data
    if args.plot:
        plot_2d(args.plot, data, weights=None, true_weights=signal_weights)

