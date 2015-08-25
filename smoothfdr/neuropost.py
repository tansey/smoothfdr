import numpy as np
import nibabel as nib
import argparse
import csv
import os
from collections import defaultdict
from smoothed_fdr import calc_plateaus
from utils import *
from plotutils import *

def load_nii(filename):
    img = nib.load(filename)
    return img.get_data(), img.get_affine()

def save_nii(filename, data, coords):
    img = nib.Nifti1Image(data, coords)
    nib.save(img, filename)

def load_edges(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        edges = defaultdict(list)
        for line in reader:
            x,y = int(line[0]), int(line[1])
            if y not in edges[x]:
                edges[x].append(y)
            if x not in edges[y]:
                edges[y].append(x)
    return edges


def load_shape_lookup(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        shape = [int(x) for x in reader.next()]
        lookup = {}
        for line in reader:
            lookup[int(line[0])] = (int(line[1]), int(line[2]), int(line[3]))
    return shape, lookup

def load_weights(filename, data, lookup, fdr=None):
    weights = np.loadtxt(filename, delimiter=',')
    if fdr is not None:
        weights = calc_fdr(weights, fdr)
    for i, w in enumerate(weights):
        v = lookup[i]
        data[v] = w

def blob_fdr(weights_filename, posteriors_filename, data, lookup, edges, fdr):
    weights = np.loadtxt(weights_filename, delimiter=',')
    betas = -np.log(1.0/weights - 1.)
    posteriors = np.loadtxt(posteriors_filename, delimiter=',')
    plateaus = calc_plateaus(betas, edges=edges)
    aggregate = np.array([posteriors[list(p)].mean() for v,p in plateaus])
    discoveries = calc_fdr(aggregate, fdr)
    for (v,p), d in zip(plateaus, discoveries):
        weights[list(p)] = d
    for i, w in enumerate(weights):
        v = lookup[i]
        data[v] = w
    return plateaus

def thresholded_posterior_clusters(posteriors_filename, data, lookup, edges, fdr, mincluster):
    posteriors = np.loadtxt(posteriors_filename, delimiter=',')
    fakebetas = np.array(posteriors)
    fakebetas[fakebetas < 1.0 - fdr] = 0
    fakebetas[fakebetas > 0] = 1
    plateaus = calc_plateaus(fakebetas, edges=edges)
    for v,p in plateaus:
        if len(p) < mincluster:
            continue
        m = posteriors[list(p)].mean()
        if 1-m > fdr:
            continue
        print 'Cluster size: {0}\tmean:{1}'.format(len(p), m)
        for i in p:
            data[lookup[i]] = m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads a smoothed FDR result for an fMRI image and post-processes it.')

    parser.add_argument('indir', help='The directory where everything is stored.')
    parser.add_argument('--verbose', type=int, default=1, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    parser.add_argument('--missingval', type=float, default=0, help='The value used to signify a missing data point in the array. Typically this is zero.')
    parser.add_argument('--fdr_level', type=float, default=0.1, help='The false discovery rate level to use when reporting discoveries.')
    parser.add_argument('--mincluster', type=int, default=30, help='The minimum size a cluster must be to not be filtered out.')
    parser.add_argument('--plot', action='store_true', help='If specified, will generate 2d slice plots.')

    parser.set_defaults(plot=False)

    args = parser.parse_args()

    expdir = args.indir + ('' if args.indir.endswith('/') else '/')

    if args.verbose:
        print 'Loading raw data from {0}'.format(expdir+'data.nii.gz')

    data, coords = load_nii(expdir+'data.nii.gz')

    if args.verbose:
        print 'Loading edges from {0}'.format(expdir+'edges.csv')

    edges = load_edges(expdir+'edges.csv')

    if args.verbose:
        print 'Loading shape and lookup data from {0}'.format(expdir+'lookup.csv')

    shape, lookup = load_shape_lookup(expdir+'lookup.csv')
    smoothdata = np.zeros(shape)
    smoothposts = np.zeros(shape)
    smoothdiscs = np.zeros(shape)

    if args.verbose:
        print 'Original data shape: {0} Smoothed data shape: {1} (should be the same)'.format(data.shape, smoothdata.shape)


    if args.verbose:
        print 'Loading smoothed weights from {0}'.format(expdir + 'weights.csv')

    load_weights(expdir + 'weights.csv', smoothdata, lookup)

    if args.verbose:
        print 'Loading posteriors from {0}'.format(expdir + 'posteriors.csv')

    load_weights(expdir + 'posteriors.csv', smoothposts, lookup)

    if args.verbose:
        print 'Filtering down to a local FDR threshold of {0}'.format(args.fdr_level)

    print 'smoothdiscs size: {0}'.format(smoothdiscs.shape)
    thresholded_posterior_clusters(expdir + 'posteriors.csv', smoothdiscs, lookup, edges, args.fdr_level, args.mincluster)


    # if args.verbose:
    #     print 'Loading discoveries at a FDR level of {0}'.format(args.fdr_level)

    # load_weights(expdir + 'posteriors.csv', smoothdiscs, lookup, fdr=args.fdr_level)
    
    #plateaus = blob_fdr(expdir + 'weights.csv', expdir + 'posteriors.csv', smoothdiscs, lookup, edges, args.fdr_level)
    # if args.verbose:
    #     print '# blobs: {0}.\nSaving blob size vs. average posterior to {1}'.format(len(plateaus), expdir + 'img/plateau_sizes_vs_posteriors.pdf')
    # plot_plateau_sizes_vs_posteriors(plateaus, np.loadtxt(expdir+'posteriors.csv', delimiter=','), expdir + 'img/plateau_sizes_vs_posteriors.pdf')

    if args.verbose:
        print 'Saving .nii.gz versions'

    save_nii(expdir + 'weights.nii.gz', smoothdata, coords)
    save_nii(expdir + 'posteriors.nii.gz', smoothposts, coords)
    save_nii(expdir + 'thresholded_posterior_clusters.nii.gz', smoothdiscs, coords)

    # if not os.path.exists(expdir + 'thresholded_posterior_clusters/'):
    #     os.makedirs(expdir + 'thresholded_posterior_clusters/')

    # for i in xrange(smoothdiscs.shape[0]):
    #     np.savetxt(expdir + 'thresholded_posterior_clusters/{0}.csv'.format(i), smoothdiscs[i], delimiter=',')

    if args.plot:
        if args.verbose:
            print 'Plotting to {0}'.format(expdir + 'img/')

        data[np.where(data == args.missingval)] = np.nan
        smoothdata[np.where(data == args.missingval)] = np.nan
        smoothposts[np.where(data == args.missingval)] = np.nan


        # Create the image directory if it doesn't already exist
        if not os.path.exists(expdir + 'img/'):
            os.makedirs(expdir + 'img/')


        for i in xrange(3):
            # Create the axis directory if it doesn't already exist
            if not os.path.exists(expdir + 'img/' + str(i)):
                os.makedirs(expdir + 'img/' + str(i))
            # Plot the 3D image by taking 2D slices along the i'th axis
            plot_3d(expdir + 'img/' + str(i) + '/{0:04d}.pdf', data, weights=smoothdata, posteriors=smoothposts, discoveries=smoothdiscs, axis=i)


    
    