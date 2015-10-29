import sys
import numpy as np
import os.path


def tpr_fdr(truth, estimated):
    # Get the true positive and false discovery rate
    trues = truth == 1
    tpr = estimated[trues].sum() / float(trues.sum())
    discs = estimated == 1
    fdr = (1 - truth[discs]).sum() / float(discs.sum())
    return (tpr, fdr)


# Get the two signals, true and estimated
truth = np.loadtxt(sys.argv[1] + 'true_signals.csv', delimiter=',').flatten()
bh = np.loadtxt(sys.argv[1] + 'bh_discoveries.csv', delimiter=',').flatten()
fdrl = np.loadtxt(sys.argv[1] + 'fdrl_discoveries.csv', delimiter=',').flatten()
hmrf = np.loadtxt(sys.argv[1] + 'hmrf_discoveries.csv', delimiter=',').flatten()
sfdr = np.loadtxt(sys.argv[1] + 'sfdr_discoveries.csv', delimiter=',').flatten()
oracle = np.loadtxt(sys.argv[1] + 'oracle_discoveries.csv', delimiter=',').flatten()

bh_tpr, bh_fdr = tpr_fdr(truth, bh)
fdrl_tpr, fdrl_fdr = tpr_fdr(truth, fdrl)
hmrf_tpr, hmrf_fdr = tpr_fdr(truth, hmrf)
sfdr_tpr, sfdr_fdr = tpr_fdr(truth, sfdr)
oracle_tpr, oracle_fdr = tpr_fdr(truth, oracle)

if not os.path.isfile(sys.argv[2]):
    with open(sys.argv[2], 'wb') as f:
        f.write('BH_TPR,BH_FDR,FDRL_TPR,FDRL_FDR,HMRF_TPR,HMRF_FDR,SFDR_TPR,SFDR_FDR,ORACLE_TPR,ORACLE_FDR\n')

with open(sys.argv[2], 'a') as f:
    f.write('{0:.4f},{1:.4f},{2:.4f},{3:.4f},{4:.4f},{5:.4f},{6:.4f},{7:.4f},{8:.4f},{9:.4f}\n'.format(bh_tpr, bh_fdr, fdrl_tpr, fdrl_fdr, hmrf_tpr, hmrf_fdr, sfdr_tpr, sfdr_fdr, oracle_tpr, oracle_fdr))
