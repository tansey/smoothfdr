import itertools
import numpy as np
from scipy import sparse
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csc_matrix, linalg as sla
from functools import partial
from collections import deque, namedtuple
from pygfl.solver import TrailSolver
from smoothed_fdr import GaussianKnown
from pygfl.trails import decompose_graph, save_chains
from pygfl.utils import chains_to_trails, calc_plateaus
from networkx import Graph, connected_components
from smoothfdr.normix import *
from smoothfdr.utils import calc_fdr

def smooth_fdr(data, edges, fdr_level, initial_values=None, verbose=0, null_dist=None):
    # Decompose the graph into trails
    g = Graph()
    g.add_edges_from(edges)
    chains = decompose_graph(g, heuristic='greedy')
    ntrails, trails, breakpoints, edges = chains_to_trails(chains)

    if null_dist is None:
        # empirical null estimation
        mu0, sigma0 = empirical_null(data, verbose=max(0,verbose-1))
    else:
        mu0, sigma0 = null_dist
    null_dist = GaussianKnown(mu0, sigma0)

    if verbose:
        print 'Empirical null: {0}'.format(null_dist)

    # signal distribution estimation
    num_sweeps = 10
    grid_x = np.linspace(min(-20, data.min() - 1), max(data.max() + 1, 20), 220)
    pr_results = predictive_recursion(data, num_sweeps, grid_x, mu0=mu0, sig0=sigma0)
    signal_dist = GridDistribution(pr_results['grid_x'], pr_results['y_signal'])

    solver = TrailSolver()
    solver.set_data(data, edges, ntrails, trails, breakpoints)

    results = solution_path_smooth_fdr(data, solver, null_dist, signal_dist, verbose=max(0, verbose-1))

    results['discoveries'] = calc_fdr(results['posteriors'], fdr_level)
    results['null_dist'] = null_dist
    results['signal_dist'] = signal_dist

    return results

def solution_path_smooth_fdr(data, solver, null_dist, signal_dist, min_lambda=0.20, max_lambda=1.5, lambda_bins=30, verbose=0, initial_values=None):
        '''Follows the solution path of the generalized lasso to find the best lambda value.'''
        lambda_grid = np.exp(np.linspace(np.log(max_lambda), np.log(min_lambda), lambda_bins))
        aic_trace = np.zeros(lambda_grid.shape) # The AIC score for each lambda value
        aicc_trace = np.zeros(lambda_grid.shape) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros(lambda_grid.shape) # The BIC score for each lambda value
        dof_trace = np.zeros(lambda_grid.shape) # The degrees of freedom of each final solution
        log_likelihood_trace = np.zeros(lambda_grid.shape)
        beta_trace = []
        u_trace = []
        w_trace = []
        c_trace = []
        results_trace = []
        best_idx = None
        best_plateaus = None
        for i, _lambda in enumerate(lambda_grid):
            if verbose:
                print '#{0} Lambda = {1}'.format(i, _lambda)

            # Fit to the final values
            results = fixed_penalty_smooth_fdr(data, solver, _lambda, null_dist, signal_dist,
                                               verbose=max(0,verbose - 1),
                                               initial_values=initial_values)

            if verbose:
                print 'Calculating degrees of freedom'

            plateaus = calc_plateaus(results['beta'], solver.edges)
            dof_trace[i] = len(plateaus)

            if verbose:
                print 'Calculating AIC'

            # Get the negative log-likelihood
            log_likelihood_trace[i] = -_data_negative_log_likelihood(data, results['c'], null_dist, signal_dist)

            # Calculate AIC = 2k - 2ln(L)
            aic_trace[i] = 2. * dof_trace[i] - 2. * log_likelihood_trace[i]
            
            # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
            aicc_trace[i] = aic_trace[i] + 2 * dof_trace[i] * (dof_trace[i]+1) / (data.shape[0] - dof_trace[i] - 1.)

            # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
            bic_trace[i] = -2 * log_likelihood_trace[i] + dof_trace[i] * (np.log(len(data)) - np.log(2 * np.pi))

            # Track the best model thus far
            if best_idx is None or bic_trace[i] < bic_trace[best_idx]:
                best_idx = i
                best_plateaus = plateaus

            # Save the final run parameters to use for warm-starting the next iteration
            initial_values = results

            # Save the trace of all the resulting parameters
            beta_trace.append(results['beta'])
            w_trace.append(results['w'])
            c_trace.append(results['c'])

            if verbose:
                print 'DoF: {0} AIC: {1} AICc: {2} BIC: {3}'.format(dof_trace[i], aic_trace[i], aicc_trace[i], bic_trace[i])

        if verbose:
            print 'Best setting (by BIC): lambda={0} [DoF: {1}, AIC: {2}, AICc: {3} BIC: {4}]'.format(lambda_grid[best_idx], dof_trace[best_idx], aic_trace[best_idx], aicc_trace[best_idx], bic_trace[best_idx])

        return {'aic': aic_trace,
                'aicc': aicc_trace,
                'bic': bic_trace,
                'dof': dof_trace,
                'loglikelihood': log_likelihood_trace,
                'beta_iters': np.array(beta_trace),
                'posterior_iters': np.array(w_trace),
                'prior_iters': np.array(c_trace),
                'lambda_iters': lambda_grid,
                'best': best_idx,
                'betas': beta_trace[best_idx],
                'priors': c_trace[best_idx],
                'posteriors': w_trace[best_idx],
                'lambda': lambda_grid[best_idx],
                'plateaus': best_plateaus}

def fixed_penalty_smooth_fdr(data, solver, _lambda, null_dist, signal_dist, initial_values=None, verbose=0):
    converge = 1e-6
    max_steps = 30
    m_steps = 1
    m_converge = 1e-6

    w_iters = []
    beta_iters = []
    c_iters = []
    delta_iters = []    

    delta = converge + 1
        
    if initial_values is None:
        beta = np.zeros(data.shape)
        prior_prob = np.exp(beta) / (1 + np.exp(beta))
    else:
        beta = initial_values['beta']
        prior_prob = initial_values['c']

    prev_nll = 0
    cur_step = 0
    
    while delta > converge and cur_step < max_steps:
        if verbose:
            print 'Step #{0}'.format(cur_step)

        if verbose:
            print '\tE-step...'

        # Get the likelihood weights vector (E-step)
        post_prob = _e_step(data, prior_prob, null_dist, signal_dist)

        if verbose:
            print '\tM-step...'

        # Find beta using an alternating Taylor approximation and convex optimization (M-step)
        beta, initial_values = _m_step(beta, prior_prob, post_prob, _lambda,
                                       solver, m_converge, m_steps,
                                       max(0,verbose-1), initial_values)

        # Get the signal probabilities
        prior_prob = ilogit(beta)
        cur_nll = _data_negative_log_likelihood(data, prior_prob, null_dist, signal_dist)
        
        # Track the change in log-likelihood to see if we've converged
        delta = np.abs(cur_nll - prev_nll) / (prev_nll + converge)

        if verbose:
            print '\tDelta: {0}'.format(delta)

        # Track the step
        w_iters.append(post_prob)
        beta_iters.append(beta)
        c_iters.append(prior_prob)
        delta_iters.append(delta)

        # Increment the step counter
        cur_step += 1

        # Update the negative log-likelihood tracker
        prev_nll = cur_nll

        # DEBUGGING
        if verbose:
            print '\tbeta: [{0:.4f}, {1:.4f}]'.format(beta.min(), beta.max())
            print '\tprior_prob:    [{0:.4f}, {1:.4f}]'.format(prior_prob.min(), prior_prob.max())
            print '\tpost_prob:    [{0:.4f}, {1:.4f}]'.format(post_prob.min(), post_prob.max())
            
    w_iters = np.array(w_iters)
    beta_iters = np.array(beta_iters)
    c_iters = np.array(c_iters)
    delta_iters = np.array(delta_iters)

    # Return the results of the run
    return {'beta': beta, 'w': post_prob, 'c': prior_prob,
            'z': initial_values['z'], 'u': initial_values['u'],
            'w_iters': w_iters, 'beta_iters': beta_iters,
            'c_iters': c_iters, 'delta_iters': delta_iters}

def _data_negative_log_likelihood(data, prior_prob, null_dist, signal_dist):
    '''Calculate the negative log-likelihood of the data given the weights.'''
    signal_weight = prior_prob * signal_dist.pdf(data)
    null_weight = (1-prior_prob) * null_dist.pdf(data)
    return -np.log(signal_weight + null_weight).sum()

def _e_step(data, prior_prob, null_dist, signal_dist):
    '''Calculate the complete-data sufficient statistics (weights vector).'''
    signal_weight = prior_prob * signal_dist.pdf(data)
    null_weight = (1-prior_prob) * null_dist.pdf(data)
    post_prob = signal_weight / (signal_weight + null_weight)
    return post_prob

def _m_step(beta, prior_prob, post_prob, _lambda,
                solver, converge, max_steps,
                verbose, initial_values):
    '''
    Alternating Second-order Taylor-series expansion about the current iterate
    and coordinate descent to optimize Beta.
    '''
    prev_nll = _m_log_likelihood(post_prob, beta)
    delta = converge + 1
    cur_step = 0
    while delta > converge and cur_step < max_steps:
        if verbose:
            print '\t\tM-Step iteration #{0}'.format(cur_step)
            print '\t\tTaylor approximation...'

        # Cache the exponentiated beta
        exp_beta = np.exp(beta)

        # Form the parameters for our weighted least squares
        weights = (prior_prob * (1 - prior_prob))
        y = beta - (prior_prob - post_prob) / weights

        solver.set_values_only(y, weights=weights)
        if initial_values is None:
            initial_values = {'beta': solver.beta, 'z': solver.z, 'u': solver.u}
        else:
            solver.beta = initial_values['beta']
            solver.z = initial_values['z']
            solver.u = initial_values['u']
        solver.solve(_lambda)
        # if np.abs(beta).max() > 20:
        #     beta = np.clip(beta, -20, 20)
        #     u = None

        beta = initial_values['beta']

        # Get the current log-likelihood
        cur_nll = _m_log_likelihood(post_prob, beta)

        # Track the convergence
        delta = np.abs(prev_nll - cur_nll) / (prev_nll + converge)

        if verbose:
            print '\t\tM-step delta: {0}'.format(delta)

        # Increment the step counter
        cur_step += 1

        # Update the negative log-likelihood tracker
        prev_nll = cur_nll

    return beta, initial_values

def _m_log_likelihood(post_prob, beta):
    '''Calculate the log-likelihood of the betas given the weights and data.'''
    return (np.log(1 + np.exp(beta)) - post_prob * beta).sum()

def ilogit(x):
    return 1. / (1. + np.exp(-x))


