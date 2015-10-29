import sys
import numpy as np

exp_dir = sys.argv[1]

if not exp_dir.endswith('/'):
    exp_dir += '/'

x = np.zeros((3,130,130))
x[1, 1:129, 1:129] = np.loadtxt(exp_dir + 'data.csv', delimiter=',')
np.savetxt(exp_dir + 'bufferdata.csv', x.flatten(), delimiter=',', fmt='%f')

# Buffered regions
x = np.arange(3*130*130).reshape((3,130,130))+1
np.savetxt(exp_dir + 'bufferregions.csv', x[1,1:129,1:129].flatten(), delimiter=',', fmt='%d')

# Buffered p-values
x = np.zeros((3,130,130))
y = np.loadtxt(exp_dir + 'flatdata_pvalues.csv', delimiter=',')
x[1, 1:129, 1:129] = y.reshape((128,128))
np.savetxt(exp_dir + 'bufferpvalues.csv', x.flatten(), delimiter=',', fmt='%f')
