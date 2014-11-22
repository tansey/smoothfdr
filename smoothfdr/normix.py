import numpy as np
from scipy.stats import norm as norm
from copy import deepcopy

class GridDistribution:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def pdf(self, data):
		# Find the closest bins
		rhs = np.searchsorted(self.x, data)
		lhs = (rhs - 1).clip(0)
		rhs = rhs.clip(0, len(self.x) - 1)

		# Linear approximation (trapezoid rule)
		rhs_dist = np.abs(self.x[rhs] - data)
		lhs_dist = np.abs(self.x[lhs] - data)
		denom = rhs_dist + lhs_dist
		denom[denom == 0] = 1. # handle the zero-distance edge-case
		rhs_weight = 1.0 - rhs_dist / denom
		lhs_weight = 1.0 - rhs_weight

		return lhs_weight * self.y[lhs] + rhs_weight * self.y[rhs]

def trapezoid(x, y):
	return np.sum((x[1:] - x[0:-1]) * (y[1:] + y[0:-1]) / 2.)

def generate_sweeps(num_sweeps, num_samples):
	results = []
	for sweep in xrange(num_sweeps):
		a = np.arange(num_samples)
		np.random.shuffle(a)
		results.extend(a)
	return np.array(results)

def predictive_recursion(z, num_sweeps, grid_x, mu0=0., sig0=1.,
							nullprob=1.0, decay=-0.67):
	sweeporder = generate_sweeps(num_sweeps, len(z))
	theta_guess = np.ones(len(grid_x)) / float(len(grid_x))
	return predictive_recursion_fdr(z, sweeporder, grid_x, theta_guess,
									mu0, sig0, nullprob, decay)

def predictive_recursion_fdr(z, sweeporder, grid_x, theta_guess, mu0 = 0.,
							sig0 = 1.0, nullprob = 1.0, decay = -0.67):
	gridsize = grid_x.shape[0]
	theta_subdens = deepcopy(theta_guess)
	pi0 = nullprob
	joint1 = np.zeros(gridsize)
	ftheta1 = np.zeros(gridsize)

	# Begin sweep through the data
	for i, k in enumerate(sweeporder):
		cc = (3. + i)**decay
		joint1 = norm.pdf(grid_x, loc=z[k] - mu0, scale=sig0) * theta_subdens
		m0 = pi0 * norm.pdf(z[k] - mu0, 0., sig0)
		m1 = trapezoid(grid_x, joint1)
		mmix = m0 + m1
		pi0 = (1. - cc) * pi0 + cc * (m0 / mmix)
		ftheta1 = joint1 / mmix
		theta_subdens = (1. - cc) * theta_subdens + cc * ftheta1

	# Now calculate marginal distribution along the grid points
	y_mix = np.zeros(gridsize)
	y_signal = np.zeros(gridsize)
	for i, x in enumerate(grid_x):
		joint1 = norm.pdf(grid_x, x - mu0, sig0) * theta_subdens
		m0 = pi0 * norm.pdf(x, mu0, sig0)
		m1 = trapezoid(grid_x, joint1)
		y_mix[i] = m0 + m1;
		y_signal[i] = m1 / (1. - pi0)

	return {'grid_x': grid_x,
            'sweeporder': sweeporder,
			'theta_subdens': theta_subdens,
			'pi0': pi0,
			'y_mix': y_mix,
			'y_signal': y_signal}