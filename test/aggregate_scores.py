import numpy as np
import sys
import csv

scores = np.loadtxt(sys.argv[1], delimiter=',', skiprows=1)
with open(sys.argv[2], 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(scores.mean(axis=0))
print '{0}'.format(sys.argv[2])

means = scores.mean(axis=0)
stdevs = scores.std(axis=0)
print '---> TPR <--- BH: {0:.3f} ({1:.3f}) FDR-L: {2:.3f} ({3:.3f}) HMRF: {4:.3f} ({5:.3f}) SFDR: {6:.3f} ({7:.3f}) Oracle: {8:.3f} ({9:.3f})'.format(means[0], stdevs[0], means[2], stdevs[2], means[4], stdevs[4], means[6], stdevs[6], means[8], stdevs[8])
print '---> FDR <--- BH: {0:.3f} ({1:.3f}) FDR-L: {2:.3f} ({3:.3f}) HMRF: {4:.3f} ({5:.3f}) SFDR: {6:.3f} ({7:.3f}) Oracle: {8:.3f} ({9:.3f})'.format(means[1], stdevs[1], means[3], stdevs[3], means[5], stdevs[5], means[7], stdevs[7], means[9], stdevs[9])