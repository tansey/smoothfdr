import matplotlib as mpl
mpl.use('Agg')
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


def calculate_1d_signal_weights(split_points, split_weights):
    '''Generate signal weights from the user-specified splits.'''
    signal_weights = np.zeros((split_points[-1] + 1, 1))
    cur_split = 0
    cur_point = 0
    while cur_split < len(split_weights):
        cur_weight = split_weights[cur_split]
        while cur_point < split_points[cur_split]:
            signal_weights[cur_point] = cur_weight
            cur_point += 1
        cur_split += 1

    return signal_weights

def calculate_2d_signal_weights(width, height, default_weight, x_min, x_max, y_min, y_max, weights):
    '''Generate signal weights from the user-specified splits.'''
    signal_weights = np.zeros((width, height)) + default_weight
    for region in zip(x_min, x_max, y_min, y_max, weights):
        signal_weights[region[0]:region[1]+1,region[2]:region[3]+1] = region[4]
    return signal_weights

def calculate_3d_signal_weights(width, height, depth, default_weight, x_min, x_max, y_min, y_max, z_min, z_max, weights):
    '''Generate signal weights from the user-specified splits.'''
    signal_weights = np.zeros((width, height, depth)) + default_weight
    for region in zip(x_min, x_max, y_min, y_max, z_min, z_max, weights):
        signal_weights[region[0]:region[1]+1,region[2]:region[3]+1,region[4]:region[5]+1] = region[6]
    return signal_weights

def generate_data_helper(flips, null_mean, null_stdev, signal_dist):
    '''Recursively builds multi-dimensional datasets.'''
    if len(flips.shape) > 1:
        return np.array([generate_data_helper(row, null_mean, null_stdev, signal_dist) for row in flips])

    # If we're on the last dimension, return the vector
    return np.array([signal_dist.sample() if flip else 0 for flip in flips]) + np.random.normal(loc=null_mean, scale=null_stdev, size=len(flips))

def generate_data(null_mean, null_stdev, signal_dist, signal_weights):
    '''Create a synthetic dataset.'''
    # Flip biased coins to decide which distribution to draw each sample from
    flips = np.random.random(size=signal_weights.shape) < signal_weights

    # Recursively generate the dataset
    samples = generate_data_helper(flips, null_mean, null_stdev, signal_dist)

    # Observed z-scores
    z = (samples - null_mean) / null_stdev

    return (z, flips)

def save_data(data, filename, header=True):
    '''Saves a CSV file containing the z-scores.'''
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        
        # write a header line
        if header:
            writer.writerow(['Z{0}'.format(x+1) for x in xrange(data.shape[1])])

        # write the data to file
        writer.writerows(data)

def load_data(filename, header=True):
    '''Loads a CSV file containing the z-scores.'''
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        data = []

        # skip the header line
        if header:
            reader.next()

        # read in all the rows
        for line in reader:
            data.append(np.array([float(x) if x != 'True' and x != 'False' else (1 if x == 'True' else 0) for x in line], dtype='double'))

    # Return the matrix of z-scores
    return np.array(data, dtype='double') if len(data) > 1 else data[0]

def load_neurodata(filename, header=True):
    '''Loads a CSV file containing the z-scores of neuro-image data that is not rectangular.'''
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        
        # skip the header line
        if header:
            reader.next()

        rows = []
        for line in reader:
            if len(line) == 0:
                continue
            rows.append(np.array([float(x) for x in line]))
        return np.array(rows).T

def save_sweeporder(data, sweeporder, filename):
    '''Saves the sweeporder to file.'''
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Z-Score'])
        for s in sweeporder:
            writer.writerow([s, data[s]])

def save_plateaus(plateaus, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['PlateauID','Level','NodeIDs'])
        for plateau,(level,nodes) in enumerate(plateaus):
            nodes = list(nodes)
            if type(nodes[0]) is tuple:
                nodes = [';'.join([str(x) for x in y]) for y in nodes]
            writer.writerow([plateau, level] + list(nodes))

def load_plateaus(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        reader.next()
        plateaus = []
        for line in reader:
            vals = line[2:]
            plateaus.append([tuple([int(y) for y in x.split(';')]) for x in vals])
    return plateaus


def calc_signal_weights(args):
    if args.dimensions == '3d':
        # Get the weights for each point in the 3d grid
        return calculate_3d_signal_weights(args.width, args.height, args.depth,
                                                  args.default_weight,
                                                  args.region_min_x, args.region_max_x,
                                                  args.region_min_y, args.region_max_y,
                                                  args.region_min_z, args.region_max_z,
                                                  args.region_weights)
    elif args.dimensions == '2d':
        # Get the weights for each point in the 2d grid
        return calculate_2d_signal_weights(args.width, args.height,
                                                  args.default_weight,
                                                  args.region_min_x, args.region_max_x,
                                                  args.region_min_y, args.region_max_y,
                                                  args.region_weights)
    elif args.dimensions == '1d':
        # Get the weights for each point along the line
        return calculate_1d_signal_weights(args.split_points, args.split_weights)
    else:
        raise Exception('Only 1- and 2-dimensional data are supported.')

def main():
    parser = argparse.ArgumentParser(description='Runs the smoothed FDR algorithm.')

    parser.add_argument('--verbose', type=int, default=0, help='Print detailed progress information to the console. 0=none, 1=outer-loop only, 2=all details.')
    parser.add_argument('--data_file', help='The file containing the raw z-score data.')
    parser.add_argument('--no_data_header', action='store_true', help='Specifies that there is no header line in the data file.')
    parser.add_argument('--signals_file', help='The file containing the true signal points.')
    parser.add_argument('--generate_data', dest='generate_data', action='store_true', help='Generate synthetic data and save it to the file specified.')
    parser.add_argument('--save_weights', help='The file where the resulting smoothed weights will be saved.')
    parser.add_argument('--save_final_weights', help='The file where the resulting smoothed weights after any postprocessing will be saved.')
    parser.add_argument('--save_posteriors', help='The file where the resulting smoothed posteriors will be saved.')
    parser.add_argument('--save_plateaus', help='The file where the resulting smoothed plateaus will be saved.')
    parser.add_argument('--save_signal', help='The file where the estimated signal will be saved.')
    parser.add_argument('--save_final_posteriors', help='The file where the final resulting posteriors after any postprocessing will be saved.')
    parser.add_argument('--save_oracle_posteriors', help='The file where the oracle posteriors will be saved.')

    # Generic data settings
    parser.add_argument('--empirical_null', dest='empirical_null', action='store_true', help='Estimate the null distribution empirically (recommended).')
    parser.add_argument('--null_mean', type=float, default=0., help='The mean of the null distribution.')
    parser.add_argument('--null_stdev', type=float, default=1., help='The variance of the null distribution.')
    parser.add_argument('--signal_mean', type=float, default=0., help='The mean of the signal distribution.')
    parser.add_argument('--signal_stdev', type=float, default=3., help='The variance of the signal distribution.')
    parser.add_argument('--signal_dist_name', help='The name of the signal distribution. This will dynamically call it by name. It must be in the signal_distributions.py file and have both the foo_pdf and foo_sample functions defined.')

    # Predictive recursion settings
    parser.add_argument('--estimate_signal', dest='estimate_signal', action='store_true', help='Use predictive recursion to estimate the signal distribution.')
    parser.add_argument('--pr_grid_x', nargs=3, type=int, default=[-7,7,57], help='The grid parameters (min, max, points) for the predictive recursion approximate distribution.')
    parser.add_argument('--pr_sweeps', type=int, default=50, help='The number of randomized sweeps to make over the data.')
    parser.add_argument('--pr_nullprob', type=float, default=1.0, help='The initial guess for the marginal probability of coming from the null distribution.')
    parser.add_argument('--pr_decay', type=float, default=-0.67, help='The exponential decay rate for the recursive update weights.')
    parser.add_argument('--pr_save_sweeporder', help='Save the sweep orders to the specified file.')
    
    # Plot settings
    parser.add_argument('--plot_data', help='The file to which the scatterplot of the data will be saved.')
    parser.add_argument('--plot_results', help='The file to which the results will be plotted.')
    parser.add_argument('--plot_signal', help='The file to which the estimated signal distribution will be plotted.')
    parser.add_argument('--plot_true_signal', dest='plot_true_signal', action='store_true', help='Plot the true signal distribution along with the estimated one in plot_signal.')
    parser.add_argument('--plot_signal_bounds', nargs=2, type=int, default=[-7, 7], help='The min and max values to plot the signal along')
    parser.add_argument('--plot_path', help='The file to which the solution path of the penalty (lambda) will be plotted.')
    parser.add_argument('--plot_adaptive', help='The file to which the results of the adaptive lasso solution will be plotted.')
    parser.add_argument('--plot_final', help='The file to which the results of the final solution will be plotted.')
    parser.add_argument('--plot_discoveries', help='The file to which the final discoveries (the ones after post-processing) will be plotted.')
    parser.add_argument('--plot_final_discoveries', help='The file to which the discoveries will be plotted.')
    parser.add_argument('--plot_path_results', help='The file format of the intermediate results plots along the path.')
    
    # Penalty (lambda) settings
    parser.add_argument('--solution_path', dest='solution_path', action='store_true', help='Use the solution path of the generalized lasso to find a good value for the penalty weight (lambda).')
    parser.add_argument('--min_penalty_weight', type=float, default=0.2, help='The minimum amount the lambda penalty can take in the solution path.')
    parser.add_argument('--max_penalty_weight', type=float, default=1.5, help='The maximum amount the lambda penalty can take in the solution path.')
    parser.add_argument('--penalty_bins', type=int, default=30, help='The number of lambda penalty values in the solution path.')
    parser.add_argument('--dof_tolerance', type=float, default=1e-4, help='The difference threshold for calculating the degrees of freedom.')
    parser.add_argument('--penalty_weight', '--lambda', type=float, default=0.3, help='The lambda penalty that controls the sparsity (only used if --solution_path is not specified).')
    parser.add_argument('--adaptive_lasso', dest='adaptive_lasso', action='store_true', help='Use an adaptive lasso value that re-weights the penalties to be inversely proportional to the size of the solution path choice with the uniform penalty.')
    parser.add_argument('--adaptive_lasso_gamma', type=float, default=1.0, help='The exponent to use for the adaptive lasso weights.')
    parser.add_argument('--postprocess_plateaus', dest='postprocess_plateaus', action='store_true', help='Perform unpenalized regression on each plateau as a final post-processing step.')

    # Smoothed FDR optimization settings
    parser.add_argument('--converge', type=float, default=1e-6, help='The convergence threshold for the main optimization loop.')
    parser.add_argument('--max_steps', type=int, default=100, help='The maximum number of steps for the main optimization loop.')
    parser.add_argument('--m_converge', type=float, default=1e-6, help='The convergence threshold for the q-step <-> m-step loop.')
    parser.add_argument('--m_max_steps', type=float, default=1, help='The maximum number of steps for the q-step <-> m-step loop.')
    parser.add_argument('--cd_converge', type=float, default=1e-6, help='The convergence threshold for the inner loop.')
    parser.add_argument('--cd_max_steps', type=float, default=100000, help='The maximum number of steps for the inner loop.')
    parser.add_argument('--admm_alpha', type=float, default=1.8, help='The step size value for the ADMM solver (if used).')
    parser.add_argument('--admm_adaptive', dest='admm_adaptive', action='store_true', help='Use an adaptive soft-thresholding value instead of the constant penalty value.')
    parser.add_argument('--admm_inflate', type=float, default=2., help='The inflation/deflation rate for the ADMM step size.')
    parser.add_argument('--dual_solver', choices=['cd', 'sls', 'lbfgs', 'admm', 'graph'], default='admm', help='The method used to solve the fused lasso problem in the M-step.')

    # FDR reporting settings
    parser.add_argument('--fdr_level', type=float, default=0.1, help='The false discovery rate level to use when reporting discoveries.')

    subparsers = parser.add_subparsers(dest='dimensions', help='The dimensions of the dataset (1d, 2d, 3d, or fmri).')

    # 1D data settings
    parser_1d = subparsers.add_parser('1d', help='Settings for 1-dimensional data.')
    parser_1d.add_argument('--split_points', nargs='+', type=int, default=[0, 250, 500, 750, 1000], help='The locations at which the signal weight changes. The first split point should always be 0.')
    parser_1d.add_argument('--split_weights', nargs='+', type=float, default=[0.2, 0.4, 0.8, 0.1, 0.35], help='The value of the signal weight for every split.')

    # 2D data settings
    parser_2d = subparsers.add_parser('2d', help='Settings for 2-dimensional data.')
    parser_2d.add_argument('--width', type=int, default=128, help='The width of the 2d grid')
    parser_2d.add_argument('--height', type=int, default=128, help='The height of the 2d grid')
    parser_2d.add_argument('--region_min_x', nargs='+', type=int, default=[10, 40], help='The min x locations at which the signal weight changes.')
    parser_2d.add_argument('--region_max_x', nargs='+', type=int, default=[25, 50], help='The max x locations at which the signal weight changes.')
    parser_2d.add_argument('--region_min_y', nargs='+', type=int, default=[10, 50], help='The min y locations at which the signal weight changes.')
    parser_2d.add_argument('--region_max_y', nargs='+', type=int, default=[25, 60], help='The max y locations at which the signal weight changes.')
    parser_2d.add_argument('--region_weights', nargs='+', type=float, default=[0.5, 0.8], help='The value of the signal weight for every region.')
    parser_2d.add_argument('--default_weight', type=float, default=0.05, help='The default signal weight for any areas not in the specified regions.')

    # 3D data settings
    parser_3d = subparsers.add_parser('3d', help='Settings for 3-dimensional data.')
    parser_3d.add_argument('--width', type=int, default=30, help='The width of the 3d grid')
    parser_3d.add_argument('--height', type=int, default=30, help='The height of the 3d grid')
    parser_3d.add_argument('--depth', type=int, default=30, help='The depth of the 3d grid')
    parser_3d.add_argument('--region_min_x', nargs='+', type=int, default=[5, 15], help='The min x locations at which the signal weight changes.')
    parser_3d.add_argument('--region_max_x', nargs='+', type=int, default=[15, 25], help='The max x locations at which the signal weight changes.')
    parser_3d.add_argument('--region_min_y', nargs='+', type=int, default=[10, 20], help='The min y locations at which the signal weight changes.')
    parser_3d.add_argument('--region_max_y', nargs='+', type=int, default=[20, 28], help='The max y locations at which the signal weight changes.')
    parser_3d.add_argument('--region_min_z', nargs='+', type=int, default=[10, 20], help='The min y locations at which the signal weight changes.')
    parser_3d.add_argument('--region_max_z', nargs='+', type=int, default=[20, 28], help='The max y locations at which the signal weight changes.')
    parser_3d.add_argument('--region_weights', nargs='+', type=float, default=[0.5, 0.8], help='The value of the signal weight for every region.')
    parser_3d.add_argument('--default_weight', type=float, default=0.05, help='The default signal weight for any areas not in the specified regions.')

    # fMRI settings -- i.e. 2D data that is non-rectangular
    parser_fmri = subparsers.add_parser('fmri', help='Settings for fMRI data. **DEPRECATED**')
    parser_fmri.add_argument('--filter_value', type=float, default=0., help='The value representing data to be ignored in the 2d grid (i.e. data not in the non-rectangular area.)')
    parser_fmri.add_argument('--positive_signal', dest='positive_signal', action='store_true', help='Set all z-scores less than zero to zero when estimating the signal.')
    parser_fmri.set_defaults(positive_signal=False)

    parser_graph = subparsers.add_parser('graph', help='Settings for generic graph connections. Note: --dual_solver must be "graph" as well.')
    parser_graph.add_argument('--trails', help='The file containing the graph trails (use the maketrails utility in pygfl to generate these).')

    parser.set_defaults(generate_data=False, no_data_header=False,
                        empirical_null=False, admm_adaptive=False,
                        estimate_signal=False, plot_true_signal=False,
                        adaptive_lasso=False, postprocess_plateaus=False)

    # Get the arguments from the command line
    args = parser.parse_args()

    mpl.rcParams['axes.linewidth'] = FIG_BORDER_WIDTH

    # Get the form of the signal distribution
    if args.signal_dist_name:
        signal_pdf = getattr(signal_distributions, '{0}_pdf'.format(args.signal_dist_name))
        noisy_signal_pdf = getattr(signal_distributions, '{0}_noisy_pdf'.format(args.signal_dist_name))
        signal_sample = getattr(signal_distributions, '{0}_sample'.format(args.signal_dist_name))
        signal_dist = ProxyDistribution(args.signal_dist_name, signal_pdf, signal_sample)
    else:
        signal_dist = GaussianKnown(args.signal_mean, args.signal_stdev)

    if args.generate_data:
        if args.verbose:
            print 'Generating data and saving to {0}'.format(args.data_file)

        signal_weights = calc_signal_weights(args)

        # Create the synthetic dataset
        data, signals = generate_data(args.null_mean, args.null_stdev, signal_dist, signal_weights)

        # Save the dataset to file
        save_data(data, args.data_file)

        # Save the truth to file
        save_data(signals, args.signals_file)
    elif args.data_file is not None:
        if args.verbose:
            print 'Loading data from {0}'.format(args.data_file)
        
        if args.dimensions == 'fmri':
            grid_data = load_neurodata(args.data_file)
            data, d2f, f2d = filter_nonrectangular_data(grid_data, filter_value=args.filter_value)
        else:
            # Load the dataset from file
            data = load_data(args.data_file, header=not args.no_data_header)

        # Load the true signals from file
        if args.signals_file is not None:
            signals = load_data(args.signals_file).astype(bool)
    else:
        raise Exception('Either --generate_data or --data_file must be specified.')

    if args.plot_data is not None:
        print 'Plotting data to {0}'.format(args.plot_data)
        if args.dimensions == '1d':
            points, weights = (args.split_points, args.split_weights) if args.generate_data else (None, None)
            plot_1d_data(data, args.plot_data, split_points=points, split_weights=weights)
        elif args.dimensions == '2d':
            plot_2d(args.plot_data, data, weights=None,
                        true_weights=signal_weights if args.generate_data else None)
        elif args.dimensions == '3d':
            plot_3d(args.plot_data, data, weights=None,
                        true_weights=signal_weights if args.generate_data else None)
        elif args.dimensions == 'fmri':
            empty_bg = np.zeros(grid_data.shape)
            empty_bg[d2f == -1] = np.nan
            empty_bg[d2f != -1] = np.abs(data)
            if args.positive_signal:
                empty_bg[grid_data < 0] = 0
            plot_2d(args.plot_data, empty_bg)

    if args.empirical_null:
        print 'Estimating null distribution empirically via Efron\'s method.'
        null_mean, null_stdev = empirical_null(data.flatten()) # use the default parameters
        null_dist = GaussianKnown(null_mean, null_stdev)
        print 'Null: N({0}, {1}^2)'.format(null_mean, null_stdev)
    else:
        print 'Using known null distribution: N({0}, {1}^2)'.format(
                                                    args.null_mean,
                                                    args.null_stdev)
        null_dist = GaussianKnown(args.null_mean, args.null_stdev)

    if args.save_oracle_posteriors:
        signal_weights = calc_signal_weights(args)
        oracle_signal_weight = signal_weights * noisy_signal_pdf(data)
        oracle_null_weight = (1-signal_weights) * null_dist.pdf(data)
        oracle_posteriors = oracle_signal_weight / (oracle_signal_weight + oracle_null_weight)
        if args.verbose:
            print 'Saving oracle posteriors to {0}'.format(args.save_oracle_posteriors)
        np.savetxt(args.save_oracle_posteriors, oracle_posteriors, delimiter=",")

    if args.estimate_signal:
        print 'Performing predictive recursion to estimate the signal distribution [{0}, {1}] ({2} bins)'.format(args.pr_grid_x[0], args.pr_grid_x[1], args.pr_grid_x[2])
        grid_x = np.linspace(args.pr_grid_x[0], args.pr_grid_x[1], args.pr_grid_x[2])
        signal_data = data.flatten()
        if args.dimensions == 'fmri' and args.positive_signal:
            signal_data[signal_data < 0] += 2

        pr_results = predictive_recursion(signal_data,
                             args.pr_sweeps, grid_x,
                             mu0=args.null_mean, sig0=args.null_stdev,
                             nullprob=args.pr_nullprob, decay=args.pr_decay)

        if args.pr_save_sweeporder:
            print 'Saving sweep order to file: {0}'.format(args.pr_save_sweeporder)
            save_sweeporder(data.flatten(), pr_results['sweeporder'], args.pr_save_sweeporder)

        # Get the estimated distribution
        estimated_dist = GridDistribution(pr_results['grid_x'], pr_results['y_signal'])

        if args.save_signal:
            if args.verbose:
                print 'Saving estimated signal to {0}'.format(args.save_signal)

            with open(args.save_signal, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(pr_results['grid_x'])
                writer.writerow(pr_results['y_signal'])

        if args.plot_signal:
            print 'Plotting estimated signal to {0}'.format(args.plot_signal)
            fig = plt.figure()
            plt.tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
            x = np.linspace(args.plot_signal_bounds[0], args.plot_signal_bounds[1], 100)
            if args.plot_true_signal:
                signal_pdf = signal_dist.pdf(x)
                plt.plot(x, signal_pdf, color='dimgray', label='Truth', linestyle='-', marker='.', markerfacecolor='black', fillstyle='full', markersize=10, alpha=0.3, lw=FIG_LINE_WIDTH)
                # Add noise to the signal
                b = norm.pdf(x, 0, 1)
                noisy_pdf = np.convolve(signal_pdf, b, mode='same')
                noisy_pdf = noisy_pdf / (noisy_pdf * (x[1] - x[0])).sum() # normalize it to sum to approx. 1
                plt.plot(x, noisy_pdf, color='blue', label='Noisy Truth', lw=FIG_LINE_WIDTH, ls='-')
            plt.plot(x, estimated_dist.pdf(x), color='orange', label='Estimated', lw=FIG_LINE_WIDTH, ls='--')
            plt.title('Estimated Signal Distribution', fontsize=FIG_TITLE_FONTSIZE)
            axes = plt.gca()
            a = ['{0}    '.format(i) if i < 0 else i for i in axes.get_xticks().tolist()]
            axes.set_xticklabels(a)
            plt.legend(loc='upper right')
            plt.savefig(args.plot_signal, bbox_inches='tight')
            plt.clf()

        # Use the estimated distribution from here on out
        signal_dist = estimated_dist
    else:
        print 'Using known signal distribution: {0}'.format(signal_dist)

    if args.verbose:
        print 'Creating penalty matrix'
    
    if args.dimensions == '1d':
        penalties = sparse_1d_penalty_matrix(len(data))
    elif args.dimensions == '2d':
        penalties = sparse_2d_penalty_matrix(data.shape)
    elif args.dimensions == '3d':
        penalties = load_trails_from_reader(cube_trails(data.shape[0], data.shape[1], data.shape[2]))
    elif args.dimensions == 'fmri':
        penalties = sparse_2d_penalty_matrix(grid_data.shape, nonrect_to_data=d2f)
    elif args.dimensions == 'graph':
        penalties = load_trails(args.trails)

    if args.verbose:
        print 'Starting Smoothed FDR Experiment'

    fdr = SmoothedFdr(signal_dist, null_dist)

    if args.solution_path:
        if args.verbose:
            print 'Finding optimal penalty (lambda) via solution path.'

        if args.dimensions == 'fmri':
            grid_map = d2f
            grid_data = grid_data
        else:
            grid_data=None
            grid_map=None

        passed_data = data.flatten() if args.dual_solver == 'graph' else data

        results = fdr.solution_path(passed_data, penalties,
            dof_tolerance=args.dof_tolerance, min_lambda=args.min_penalty_weight,
            max_lambda=args.max_penalty_weight, lambda_bins=args.penalty_bins,
            converge=args.converge, max_steps=args.max_steps,
            m_converge=args.m_converge, m_max_steps=args.m_max_steps,
            cd_converge=args.cd_converge, cd_max_steps=args.cd_max_steps,
            verbose=args.verbose, dual_solver=args.dual_solver,
            admm_alpha=args.admm_alpha, admm_adaptive=args.admm_adaptive,
            admm_inflate=args.admm_inflate,
            grid_data=grid_data, grid_map=grid_map)

        if args.plot_path:
            if args.verbose:
                print 'Plotting penalty (lambda) solution path to {0}'.format(args.plot_path)

            plot_path(results, args.plot_path)

        if args.plot_path_results:
            if args.verbose:
                print 'Plotting intermediary results of the solution path to {0}'.format(args.plot_path_results)

            for ith_lambda, ith_weights in zip(results['lambda'], results['c']):
                ith_filename = args.plot_path_results.format(ith_lambda)
                if args.dimensions == '1d':
                    points, split_weights = (args.split_points, args.split_weights) if args.generate_data else (None, None)
                    plot_1d_results(data, ith_weights, ith_filename,
                                split_points=points, split_weights=split_weights)
                elif args.dimensions == '2d':
                    plot_2d(ith_filename, data, weights=ith_weights,
                                true_weights=signal_weights if args.generate_data else None)
                elif args.dimensions == '3d':
                    plot_3d(ith_filename, data, weights=ith_weights,
                                true_weights=signal_weights if args.generate_data else None)
                elif args.dimensions == 'fmri':
                    plot_fmri_results(grid_data, ith_weights, d2f, ith_filename)

        weights = results['c'][results['best']].reshape(data.shape)
        posteriors = results['w'][results['best']].reshape(data.shape)
        plateaus = results['plateaus']
        _lambda = results['lambda'][results['best']]
    else:
        if args.verbose:
            print 'Fitting values using fixed penalty (lambda) of {0}'.format(args.penalty_weight)

        results = fdr.run(data.flatten(), penalties, args.penalty_weight,
                converge=args.converge, max_steps=args.max_steps,
                m_converge=args.m_converge, m_max_steps=args.m_max_steps,
                cd_converge=args.cd_converge, cd_max_steps=args.cd_max_steps,
                verbose=args.verbose, dual_solver=args.dual_solver,
                admm_alpha=args.admm_alpha, admm_adaptive=args.admm_adaptive,
                admm_inflate=args.admm_inflate)
        weights = results['c']
        posteriors = results['w']

        if args.dimensions == 'fmri':
            grid_points = np.zeros(grid_data.shape)
            grid_points[:,:] = np.nan
            grid_points[d2f != -1] = results['beta'][d2f[d2f != -1]]
        else:
            grid_points = results['beta'].reshape(data.shape)
        plateaus = calc_plateaus(grid_points, rel_tol=args.dof_tolerance, verbose=args.verbose)
        _lambda = args.penalty_weight
        
    if args.plot_results is not None:
        if args.verbose:
            print 'Plotting results to {0}'.format(args.plot_results)
        if args.dimensions == '1d':
            points, split_weights = (args.split_points, args.split_weights) if args.generate_data else (None, None)
            plot_1d_results(data, weights, args.plot_results,
                        split_points=points, split_weights=split_weights)
        elif args.dimensions == '2d':
            plot_2d(args.plot_results, data, weights=weights,
                        true_weights=signal_weights if args.generate_data else None)
        elif args.dimensions == '3d':
            plot_3d(args.plot_results, data, weights=weights,
                        true_weights=signal_weights if args.generate_data else None)
        elif args.dimensions == 'fmri':
            plot_fmri_results(grid_data, weights, d2f, args.plot_results)

    if args.save_weights:
        if args.verbose:
            print 'Saving weights to {0}'.format(args.save_weights)
        if args.dimensions == '3d':
            np.savetxt(args.save_weights, weights.flatten(), delimiter=",")
        else:
            np.savetxt(args.save_weights, weights, delimiter=",")

    if args.save_posteriors:
        if args.verbose:
            print 'Saving posteriors to {0}'.format(args.save_posteriors)
        if args.dimensions == '3d':
            np.savetxt(args.save_posteriors, posteriors.flatten(), delimiter=",")
        else:
           np.savetxt(args.save_posteriors, posteriors, delimiter=",")

    if args.dimensions == '3d' or args.dimensions == '2d':
        fdr_signals = calc_fdr(posteriors.flatten(), args.fdr_level).reshape(data.shape)
    else:
        fdr_signals = calc_fdr(posteriors, args.fdr_level)
    if args.plot_discoveries:
        if args.verbose:
            print 'Plotting discoveries with FDR level of {0:.2f}%'.format(args.fdr_level * 100)
        if args.dimensions == 'fmri':
            plot_fmri_results(grid_data, fdr_signals, d2f, args.plot_discoveries)

    if args.adaptive_lasso:
        if args.verbose:
            print 'Re-running with adaptive lasso penalty (lambda={0}, gamma={1})'.format(_lambda, args.adaptive_lasso_gamma)

        if args.solution_path:
            best_idx = results['best']
            results = {'beta': results['beta'][best_idx],
                       'c': results['c'][best_idx],
                       'u': results['u'][best_idx]}

        null_probs = null_dist.pdf(data.flatten())
        sig_probs = signal_dist.pdf(data.flatten())
        ols_weights = sig_probs / (null_probs + sig_probs)
        adaptive_weights = (np.abs(penalties.dot(ols_weights)) ** -args.adaptive_lasso_gamma).clip(0, 1e6)
        
        if args.verbose:
            print 'Adaptive weights range: [{0}, {1}]'.format(adaptive_weights.min(), adaptive_weights.max())

        adaptive_weights = dia_matrix(([adaptive_weights], 0), shape=(len(adaptive_weights), len(adaptive_weights)))
        adaptive_penalties = (penalties.T * adaptive_weights).T

        # Cache the LU decomposition with the new weights
        L = csc_matrix(adaptive_penalties.T.dot(adaptive_penalties) + csc_matrix(np.eye(adaptive_penalties.shape[1])))
        results['u']['lu_factor'] = sla.splu(L, permc_spec='MMD_AT_PLUS_A')

        fdr.reset()

        results = fdr.run(data.flatten(), adaptive_penalties, _lambda,
                converge=args.converge, max_steps=args.max_steps,
                m_converge=args.m_converge, m_max_steps=args.m_max_steps,
                cd_converge=args.cd_converge, cd_max_steps=args.cd_max_steps,
                verbose=args.verbose, dual_solver=args.dual_solver,
                admm_alpha=args.admm_alpha, admm_adaptive=args.admm_adaptive,
                admm_inflate=args.admm_inflate, initial_values=results
                )
        weights = results['c']
        posteriors = results['w']

        if args.plot_adaptive is not None:
            if args.verbose:
                print 'Plotting adaptive lasso results to {0}'.format(args.plot_adaptive)
            if args.dimensions == '1d':
                points, split_weights = (args.split_points, args.split_weights) if args.generate_data else (None, None)
                plot_1d_results(data, weights, args.plot_adaptive,
                            split_points=points, split_weights=split_weights)
            elif args.dimensions == '2d':
                plot_2d(args.plot_adaptive, data, weights=weights,
                            true_weights=signal_weights if args.generate_data else None)
            elif args.dimensions == '3d':
                plot_3d(args.plot_adaptive, data, weights=weights,
                            true_weights=signal_weights if args.generate_data else None)

    if args.postprocess_plateaus:
        if args.verbose:
            print 'Post-processing plateaus via unpenalized 1-d regression.'
        weights, posteriors = fdr.plateau_regression(plateaus, data, grid_map=d2f if args.dimensions=='fmri' else None, verbose=args.verbose)

    if args.save_plateaus:
        if args.verbose:
            print 'Saving plateaus to {0}'.format(args.save_plateaus)
        save_plateaus(plateaus, args.save_plateaus)
    
    if args.save_final_weights:
        if args.verbose:
            print 'Saving weights to {0}'.format(args.save_final_weights)
        if args.dimensions == '3d':
            np.savetxt(args.save_final_weights, weights.flatten(), delimiter=",")
        else:
            np.savetxt(args.save_final_weights, weights, delimiter=",")

    if args.save_final_posteriors:
        if args.verbose:
            print 'Saving posteriors to {0}'.format(args.save_final_posteriors)
        if args.dimensions == '3d':
            np.savetxt(args.save_final_posteriors, posteriors.flatten(), delimiter=",")
        else:
            np.savetxt(args.save_final_posteriors, posteriors, delimiter=",")


    if args.plot_final:
        if args.verbose:
            print 'Plotting final results to {0}'.format(args.plot_final)
        if args.dimensions == '1d':
            points, split_weights = (args.split_points, args.split_weights) if args.generate_data else (None, None)
            plot_1d_results(data, weights, args.plot_final,
                        split_points=points, split_weights=split_weights)
        elif args.dimensions == '2d':
            plot_2d(args.plot_final, data, weights=weights,
                        true_weights=signal_weights if args.generate_data else None)
        elif args.dimensions == '3d':
            plot_3d(args.plot_final, data, weights=weights,
                        true_weights=signal_weights if args.generate_data else None)
        elif args.dimensions == 'fmri':
            plot_fmri_results(grid_data, weights, d2f, args.plot_final)

    if args.dimensions == '3d':
        fdr_signals = calc_fdr(posteriors.flatten(), args.fdr_level).reshape(data.shape)
    else:
        fdr_signals = calc_fdr(posteriors, args.fdr_level)
    if args.plot_final_discoveries:
        if args.verbose:
            print 'Plotting discoveries with FDR level of {0:.2f}%'.format(args.fdr_level * 100)
        if args.dimensions == 'fmri':
            plot_fmri_results(grid_data, fdr_signals, d2f, args.plot_final_discoveries)

