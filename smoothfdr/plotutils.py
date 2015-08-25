import matplotlib as mpl
mpl.use('Agg')
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.sparse import csc_matrix, dia_matrix, linalg as sla
from utils import *

FIG_FONTSIZE = 18
FIG_TITLE_FONTSIZE = 28
FIG_SUBPLOT_TITLE_FONTSIZE = 18
FIG_LINE_WIDTH = 4
FIG_TICK_LABEL_SIZE = 14
FIG_BORDER_WIDTH = 2
FIG_TICK_WIDTH = 2

def plot_1d_data(data, filename, split_points=None, split_weights=None):
    fig = plt.figure()
    plt.tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
    plt.scatter(np.arange(len(data)), data, color='lightgray')
    if split_points is not None:
        xmin = [0] + split_points[0:-1]
        plt.hlines(y=split_weights, xmin=xmin, xmax=split_points, color='red')
    plt.xlim(0,split_points[-1])
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def plot_fmri_results(grid_data, weights, d2f, filename):
    points = np.zeros(grid_data.shape)
    points[:,:] = np.nan
    points.T[d2f.T != -1] = weights
    #print 'points[d2f != -1]: {0}'.format(points[d2f != -1])
    #print 'd2f[50]: {0}'.format((d2f != -1)[50])
    #print 'points[50]: {0}'.format(points[50])
    plot_2d(filename, grid_data, weights=points.flatten())

def plot_3d(filename, data, weights=None, true_weights=None, posteriors=None, discoveries=None, axis=0):
    for i in xrange(data.shape[axis]):
        if axis == 0:
            d = np.array(data[i])
            w = np.array(weights[i]) if weights is not None else None
            t = np.array(true_weights[i]) if true_weights is not None else None
            p = np.array(posteriors[i]) if posteriors is not None else None
            r = np.array(discoveries[i]) if discoveries is not None else None
        elif axis == 1:
            d = np.array(data[:,i,:])
            w = np.array(weights[:,i,:]) if weights is not None else None
            t = np.array(true_weights[:,i,:]) if true_weights is not None else None
            p = np.array(posteriors[:,i,:]) if posteriors is not None else None
            r = np.array(discoveries[:,i,:]) if discoveries is not None else None
        elif axis == 2:
            d = np.array(data[:,:,i])
            w = np.array(weights[:,:,i]) if weights is not None else None
            t = np.array(true_weights[:,:,i]) if true_weights is not None else None
            p = np.array(posteriors[:,:,i]) if posteriors is not None else None
            r = np.array(discoveries[:,:,i]) if discoveries is not None else None
        else:
            raise Exception('Invalid 3d axis value: axis={0}'.format(axis))
        plot_2d(filename.format(i), d, w, t, p, r)

def plot_2d_slice(ax, data, weights, true_weights, posteriors, discoveries):
    cmap = cm.binary
    cmap.set_bad('white', 1.)
    if true_weights is not None:
        ax[0].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
        ax[0].imshow(true_weights, cmap=cm.binary, interpolation='none', origin='lower', vmin=0, vmax=1)
        ax[0].set_title('Truth', fontsize=FIG_SUBPLOT_TITLE_FONTSIZE)
        ax = np.delete(ax, 0)
    if weights is not None:
        ax[1].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
        ax[1].imshow(weights, cmap=cm.binary, interpolation='none', origin='lower', vmin=0, vmax=1)
        ax[1].set_title('Smoothed Weights', fontsize=FIG_SUBPLOT_TITLE_FONTSIZE)
        ax = np.delete(ax, 1)
    if posteriors is not None:
        ax[1].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
        ax[1].imshow(posteriors, cmap=cm.binary, interpolation='none', origin='lower', vmin=0, vmax=1)
        ax[1].set_title('Posteriors', fontsize=FIG_SUBPLOT_TITLE_FONTSIZE)
        ax = np.delete(ax, 1)
    if discoveries is not None:
        ax[1].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
        ax[1].imshow(discoveries, cmap=cm.binary, interpolation='none', origin='lower', vmin=0, vmax=1)
        ax[1].set_title('Discoveries', fontsize=FIG_SUBPLOT_TITLE_FONTSIZE)
        ax = np.delete(ax, 1)
    if type(ax) is np.ndarray:
        ax = ax[0]
    masked_data = np.ma.array(data, mask=np.isnan(data))
    ax.tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
    heatmap = ax.imshow(masked_data, cmap=cmap, interpolation='none', origin='lower', vmin=0, vmax=1)
    ax.set_title('Observed', fontsize=FIG_SUBPLOT_TITLE_FONTSIZE)
    heatmap.set_norm(colors.Normalize(vmin=0., vmax=1.))
    return heatmap

def plot_2d(filename, data, weights=None, true_weights=None, posteriors=None, discoveries=None):
    cols = 1
    if discoveries is not None:
        cols += 1
    if posteriors is not None:
        cols += 1
    if true_weights is not None:
        cols += 1
    if weights is not None:
        cols += 1
    fig, ax = plt.subplots(1, cols, figsize=(5*cols+1, 5))
    heatmap = plot_2d_slice(ax, data, weights, true_weights, posteriors, discoveries)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.09, 0.02, 0.8])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def plot_1d_results(data, weights, filename, split_points=None, split_weights=None):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(data)), data, color='lightgray')
    if split_points is not None:
        xmin = [0] + split_points[0:-1]
        ax.hlines(y=split_weights, xmin=xmin, xmax=split_points, color='blue', label='Truth')
    ax.tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
    ax.plot(np.arange(len(weights)), weights, label='Smoothed FDR', color='orange')
    ax.set_xlim(0,len(data))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def plot_path(results, filename):
    fig, axarr = plt.subplots(1,4, sharex=True, figsize=(21, 5))
    axarr[0].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
    axarr[1].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
    axarr[2].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
    axarr[3].tick_params(axis='both', which='major', labelsize=FIG_TICK_LABEL_SIZE, width=FIG_TICK_WIDTH)
    axarr[0].plot(results['lambda'], results['loglikelihood'], lw=FIG_LINE_WIDTH)
    axarr[0].axvline(results['lambda'][np.argmax(results['loglikelihood'])], ymin=results['loglikelihood'].min(), ymax=results['loglikelihood'].max(), color='r', linestyle='--')
    axarr[1].plot(results['lambda'], results['dof'], lw=FIG_LINE_WIDTH)
    axarr[1].axvline(results['lambda'][np.argmin(results['dof'])], ymin=results['dof'].min(), ymax=results['dof'].max(), color='r', linestyle='--')
    axarr[2].plot(results['lambda'], results['aic'], lw=FIG_LINE_WIDTH)
    axarr[2].axvline(results['lambda'][np.argmin(results['aic'])], ymin=results['aic'].min(), ymax=results['aic'].max(), color='r', linestyle='--')
    axarr[3].plot(results['lambda'], results['bic'], lw=FIG_LINE_WIDTH)
    axarr[3].axvline(results['lambda'][np.argmin(results['bic'])], ymin=results['bic'].min(), ymax=results['bic'].max(), color='r', linestyle='--')
    axarr[0].set_title('Log-Likelihood', fontsize=FIG_TITLE_FONTSIZE)
    axarr[1].set_title('Degrees of Freedom', fontsize=FIG_TITLE_FONTSIZE)
    axarr[2].set_title('AIC', fontsize=FIG_TITLE_FONTSIZE)
    axarr[3].set_title('BIC', fontsize=FIG_TITLE_FONTSIZE)
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def plot_plateau_sizes_vs_posteriors(plateaus, posteriors, filename):
    probs = np.array([posteriors[list(p)].mean() for v,p in plateaus])
    sizes = np.array([len(p) for v,p in plateaus])
    #no_outliers = abs(sizes - np.mean(sizes)) < 1 * np.std(sizes)
    no_outliers = sizes < 100
    fig = plt.figure()
    plt.scatter(sizes[no_outliers], probs[no_outliers])
    plt.xlabel('Plateau size')
    plt.ylabel('Mean posterior probability')
    plt.ylim(0,1)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close(fig)