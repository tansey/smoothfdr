import matplotlib.pylab as plt
import numpy as np
from smoothfdr.easy import smooth_fdr

data = np.loadtxt('data.csv', delimiter=',')
fdr_level = 0.05

# Runs the FDR smoothing algorithm with the default settings
# and using a 2d grid edge set
# Note that verbose is a level not a boolean, so you can
# set verbose=0 for silent, 1 for high-level output, 2+ for more details
results = smooth_fdr(data, fdr_level, verbose=5)

# results is a dictionary containing all the information shown in
# the images for the example. let's just plot the empirical Bayes
# priors, posteriors, and the discoveries
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(data, cmap='gray_r')
ax[0,0].set_title('Raw data')

ax[0,1].imshow(results['priors'], cmap='gray_r', vmin=0, vmax=1)
ax[0,1].set_title('Smoothed prior')

ax[1,0].imshow(results['posteriors'], cmap='gray_r', vmin=0, vmax=1)
ax[1,0].set_title('Posteriors')

ax[1,1].imshow(results['discoveries'], cmap='gray_r', vmin=0, vmax=1)
ax[1,1].set_title('Discoveries at FDR={0}'.format(fdr_level))
plt.show()