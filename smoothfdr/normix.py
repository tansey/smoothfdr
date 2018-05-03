import numpy as np
from scipy.stats import norm as norm
from scipy.optimize import fmin_bfgs
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
	for sweep in range(num_sweeps):
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

def empirical_null(z, nmids=150, pct=-0.01, pct0=0.25, df=4, verbose=0):
    '''Estimate f(z) and f_0(z) using a polynomial approximation to Efron (2004)'s method.'''
    N = len(z)
    med = np.median(z)
    lb = med + (1 - pct) * (z.min() - med)
    ub = med + (1 - pct) * (z.max() - med)

    breaks = np.linspace(lb, ub, nmids+1)
    zcounts = np.histogram(z, bins=breaks)[0]
    mids = (breaks[:-1] + breaks[1:])/2

    ### Truncated Polynomial

    # Truncate to [-3, 3]
    selected = np.logical_and(mids >= -3, mids <= 3)
    zcounts = zcounts[selected]
    mids = mids[selected]

    # Form a polynomial basis and multiply by z-counts
    X = np.array([mids ** i for i in range(df+1)]).T
    beta0 = np.zeros(df+1)
    loglambda_loss = lambda beta, X, y: -((X * y[:,np.newaxis]).dot(beta) - np.exp(X.dot(beta).clip(-20,20))).sum() + 1e-6*np.sqrt((beta ** 2).sum())
    results = fmin_bfgs(loglambda_loss, beta0, args=(X, zcounts), disp=verbose)
    a = np.linspace(-3,3,1000)
    B = np.array([a ** i for i in range(df+1)]).T
    beta_hat = results

    # Back out the mean and variance from the Taylor terms
    x_max = mids[np.argmax(X.dot(beta_hat))]
    loglambda_deriv1_atmode = np.array([i * beta_hat[i] * x_max**(i-1) for i in range(1,df+1)]).sum()
    loglambda_deriv2_atmode = np.array([i * (i-1) * beta_hat[i] * x_max**(i-2) for i in range(2,df+1)]).sum()
    
    # Handle the edge cases that arise with numerical precision issues
    sigma_enull = np.sqrt(-1.0/loglambda_deriv2_atmode) if loglambda_deriv2_atmode < 0 else 1
    mu_enull = (x_max - loglambda_deriv1_atmode/loglambda_deriv2_atmode) if loglambda_deriv2_atmode != 0 else 0

    return (mu_enull, sigma_enull)
