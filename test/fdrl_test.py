import numpy as np
import scipy.stats as st
import csv
from pygfl.utils import load_edges
from smoothfdr.utils import local_agg_fdr

raw_z = np.loadtxt('/Users/wesley/Projects/smoothfdr/test/data.csv', delimiter=',', skiprows=1)
z_scores = raw_z.flatten()
p_values = 2*(1.0 - st.norm.cdf(np.abs(z_scores)))
edges = load_edges('/Users/wesley/Projects/smoothfdr/test/edges.csv')
fdr_level = 0.1
lmbda = 0.2
discoveries = local_agg_fdr(p_values, edges, fdr_level, lmbda = lmbda)
results = np.zeros(z_scores.shape)
results[discoveries] = 1
results = results.reshape(raw_z.shape)

with open('/Users/wesley/Projects/smoothfdr/test/signals.csv', 'rb') as f:
    reader = csv.reader(f)
    truth = []
    reader.next() # skip header
    for line in reader:
        truth.append(np.array([1 if x == 'True' else 0 for x in line]))
truth = np.array(truth)

tpr = np.logical_and(truth == 1, results == 1).sum() / float((truth == 1).sum())
fdr = np.logical_and(truth == 0, results == 1).sum() / float((results == 1).sum())