import numpy as np
import sys
import csv
from collections import defaultdict
from smoothfdr.utils import local_agg_fdr, p_value

def load_edges(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        edges = defaultdict(list)
        for line in reader:
            nodes = [int(x) for x in line]
            for n1,n2 in zip(nodes[:-1], nodes[1:]):
                edges[n1].append(n2)
                edges[n2].append(n1)
    return edges

edges = load_edges(sys.argv[1] + 'edges.csv')
data = np.loadtxt(sys.argv[1] + 'data.csv', delimiter=',').flatten()
pvals = p_value(data)
fdr_level = 0.1


discoveries = local_agg_fdr(pvals, edges, fdr_level, lmbda = 0.2)
results = np.zeros(data.shape)
results[discoveries] = 1
np.savetxt(sys.argv[1] + 'fdrl_discoveries.csv', results, delimiter=',', fmt='%d')
