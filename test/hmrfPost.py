import numpy as np
import scipy.stats as st
import csv
import sys
from smoothfdr.utils import calc_fdr

expdir = sys.argv[1]

with open(expdir + 'buffer_lis.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=' ')
    lis = np.array([float(x) for x in reader.next() if x != ''])

# Try treating the LIS as a prior probability of coming from the null hypothesis
# data = np.loadtxt(expdir + 'flatdata.csv', delimiter=',').flatten()
# params = np.loadtxt(expdir + 'buffer_estimate_result.csv')[2:]
# means = params[0::3]
# variances = params[1::3]
# mix_weights = params[2::3]
# sigprob = lambda x: np.sum([w * st.norm.pdf(x, loc=m, scale=np.sqrt(v)) for w, m, v in zip(means, variances, mix_weights)], axis=0)
# nullprob = lambda x: st.norm.pdf(x)
# postprob = (sigprob(data) * (1-lis)) / (sigprob(data) * (1-lis) + nullprob(data) * lis)
# discovered = calc_fdr(postprob, 0.1)

# Step-up procedure for LIS
alpha = 0.1 # FDR level
lis_orders = np.argsort(lis)[::1]
lis_sum = 0
max_i = len(lis)
for i, s in enumerate(lis_orders):
    lis_sum += lis[s]
    threshold = 1. / (1. + i) * lis_sum
    if threshold > alpha:
        max_i = i
        break

# Save the discoveries to file
discovered = np.zeros((128,128))
selected = lis_orders[:max_i]
discovered = discovered.flatten()
discovered[selected] = 1
discovered = discovered.reshape((128,128))
np.savetxt(expdir + 'hmrf_discoveries.csv', discovered, delimiter=',', fmt='%d')
