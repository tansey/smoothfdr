import numpy as np
import nibabel as nib
import argparse
import csv
import os
from utils import *

def load_nii(filename):
    img = nib.load(filename)
    return img.get_data()

def cube_to_vector(data, edges):
    lookup = {}
    beta = []
    beta_edges = []
    for x1, x2 in edges:
        if x1 in lookup:
            y1 = lookup[x1]
        else:
            y1 = len(beta)
            lookup[x1] = y1
            beta.append(data[x1])
        if x2 in lookup:
            y2 = lookup[x2]
        else:
            y2 = len(beta)
            lookup[x2] = y2
            beta.append(data[x2])
        beta_edges.append((y1, y2))
    return np.array(beta), beta_edges, lookup

def cube_trails_to_vector_trails(cube_trails, lookup):
    vec_trails = []
    for t in cube_trails:
        vec_trails.append([lookup[x] for x in t])
    return vec_trails

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads a .nii or .nii.gz file and processes it.')

    parser.add_argument('infile', help='The neuroimaging file.')
    parser.add_argument('--outdir', default='../data/', help='Save the betas, edges, and mapping from betas to x,y,z coords to the specified directory in CSV format.')
    parser.add_argument('--verbose', type=int, default=1, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    parser.add_argument('--missingval', type=float, default=0, help='The value used to signify a missing data point in the array. Typically this is zero.')
    
    parser.set_defaults()

    args = parser.parse_args()

    if args.verbose:
        print 'Loading data from {0}'.format(args.infile)

    data = load_nii(args.infile)

    if args.verbose:
        print 'Data shape: {0}'.format(data.shape)

    raw_edges = cube_edges(data, missing_val=args.missingval)

    if args.verbose:
        print 'Edges: {0}'.format(len(raw_edges))

    betas, edges, lookup = cube_to_vector(data, raw_edges)

    if args.verbose:
        print 'Vertices: {0}'.format(len(betas))

    trails = cube_trails_missing(data, missing_val=args.missingval)
    trails = cube_trails_to_vector_trails(trails, lookup)
    
    if args.verbose:
        print 'Trails: {0}'.format(len(trails))

    outdir = args.outdir + ('' if args.outdir.endswith('/') else '/')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.verbose:
        print 'Saving betas to {0}'.format(outdir+'betas.csv')
    np.savetxt(outdir+'betas.csv', betas, delimiter=',')

    if args.verbose:
        print 'Saving edges to {0}'.format(outdir+'edges.csv')
    with open(outdir+'edges.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(edges)

    if args.verbose:
        print 'Saving trails to {0}'.format(outdir+'trails.csv')
    save_trails(outdir+'trails.csv', trails)

    if args.verbose:
        print 'Saving map from vector index -> (x,y,z) to {0}'.format(outdir+'lookup.csv')
        print 'NOTE: the first line will be the dimensions of the original data.'
    with open(outdir+'lookup.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(data.shape)
        for (x,y,z), bidx in lookup.iteritems():
            writer.writerow([bidx,x,y,z])

    
    