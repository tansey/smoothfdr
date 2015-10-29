import sys
import numpy as np
from smoothfdr.utils import calc_fdr

posteriors = np.loadtxt(sys.argv[1] + 'oracle_posteriors.csv', delimiter=',')
discoveries = calc_fdr(posteriors, 0.1)
np.savetxt(sys.argv[1] + 'oracle_discoveries.csv', discoveries, delimiter=',', fmt='%d')