import itertools
import numpy as np
from scipy import sparse
from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csc_matrix, linalg as sla
from functools import partial
from collections import deque
from pygfl.solver import TrailSolver

class GaussianKnown:
    '''
    A simple Gaussian distribution with known mean and stdev.
    '''
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def pdf(self, data):
        return norm.pdf(data, loc=self.mean, scale=self.stdev)

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.stdev)

    def __repr__(self):
        return 'N({0}, {1}^2)'.format(self.mean, self.stdev)


class SmoothedFdr(object):
    def __init__(self, signal_dist, null_dist, penalties_cross_x=None):
        self.signal_dist = signal_dist
        self.null_dist = null_dist

        if penalties_cross_x is None:
            self.penalties_cross_x = np.dot
        else:
            self.penalties_cross_x = penalties_cross_x

        self.w_iters = []
        self.beta_iters = []
        self.c_iters = []
        self.delta_iters = []

        # ''' Load the graph fused lasso library '''
        # graphfl_lib = cdll.LoadLibrary('libgraphfl.so')
        # self.graphfl_weight = graphfl_lib.graph_fused_lasso_weight_warm
        # self.graphfl_weight.restype = c_int
        # self.graphfl_weight.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
        #                     c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
        #                     c_double, c_double, c_double, c_int, c_double,
        #                     ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]
        self.solver = TrailSolver()

    def add_step(self, w, beta, c, delta):
        self.w_iters.append(w)
        self.beta_iters.append(beta)
        self.c_iters.append(c)
        self.delta_iters.append(delta)

    def finish(self):
        self.w_iters = np.array(self.w_iters)
        self.beta_iters = np.array(self.beta_iters)
        self.c_iters = np.array(self.c_iters)
        self.delta_iters = np.array(self.delta_iters)

    def reset(self):
        self.w_iters = []
        self.beta_iters = []
        self.c_iters = []
        self.delta_iters = []

    def solution_path(self, data, penalties, dof_tolerance=1e-4,
            min_lambda=0.20, max_lambda=1.5, lambda_bins=30,
            converge=0.00001, max_steps=100, m_converge=0.00001,
            m_max_steps=100, cd_converge=0.00001, cd_max_steps=100, verbose=0, dual_solver='admm',
            admm_alpha=1., admm_inflate=2., admm_adaptive=False, initial_values=None,
            grid_data=None, grid_map=None):
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
        flat_data = data.flatten()
        edges = penalties[3] if dual_solver == 'graph' else None
        if grid_data is not None:
                grid_points = np.zeros(grid_data.shape)
                grid_points[:,:] = np.nan
        for i, _lambda in enumerate(lambda_grid):
            if verbose:
                print '#{0} Lambda = {1}'.format(i, _lambda)

            # Clear out all the info from the previous run
            self.reset()

            # Fit to the final values
            results = self.run(flat_data, penalties, _lambda=_lambda, converge=converge, max_steps=max_steps,
                           m_converge=m_converge, m_max_steps=m_max_steps, cd_converge=cd_converge,
                           cd_max_steps=cd_max_steps, verbose=verbose, dual_solver=dual_solver,
                           admm_alpha=admm_alpha, admm_inflate=admm_inflate, admm_adaptive=admm_adaptive,
                           initial_values=initial_values)

            if verbose:
                print 'Calculating degrees of freedom'

            # Create a grid structure out of the vector of betas
            if grid_map is not None:
                grid_points[grid_map != -1] = results['beta'][grid_map[grid_map != -1]]
            else:
                grid_points = results['beta'].reshape(data.shape)

            # Count the number of free parameters in the grid (dof)
            plateaus = calc_plateaus(grid_points, dof_tolerance, edges=edges)
            dof_trace[i] = len(plateaus)
            #dof_trace[i] = (np.abs(penalties.dot(results['beta'])) >= dof_tolerance).sum() + 1 # Use the naive DoF

            if verbose:
                print 'Calculating AIC'

            # Get the negative log-likelihood
            log_likelihood_trace[i] = -self._data_negative_log_likelihood(flat_data, results['c'])

            # Calculate AIC = 2k - 2ln(L)
            aic_trace[i] = 2. * dof_trace[i] - 2. * log_likelihood_trace[i]
            
            # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
            aicc_trace[i] = aic_trace[i] + 2 * dof_trace[i] * (dof_trace[i]+1) / (flat_data.shape[0] - dof_trace[i] - 1.)

            # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
            bic_trace[i] = -2 * log_likelihood_trace[i] + dof_trace[i] * (np.log(len(flat_data)) - np.log(2 * np.pi))

            # Track the best model thus far
            if best_idx is None or bic_trace[i] < bic_trace[best_idx]:
                best_idx = i
                best_plateaus = plateaus

            # Save the final run parameters to use for warm-starting the next iteration
            initial_values = results

            # Save the trace of all the resulting parameters
            beta_trace.append(results['beta'])
            u_trace.append(results['u'])
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
                'beta': np.array(beta_trace),
                'u': np.array(u_trace),
                'w': np.array(w_trace),
                'c': np.array(c_trace),
                'lambda': lambda_grid,
                'best': best_idx,
                'plateaus': best_plateaus}


    def run(self, data, penalties, _lambda=0.1, converge=0.00001, max_steps=100, m_converge=0.00001,
            m_max_steps=100, cd_converge=0.00001, cd_max_steps=100, verbose=0, dual_solver='admm',
            admm_alpha=1., admm_inflate=2., admm_adaptive=False, initial_values=None):
        '''Runs the Expectation-Maximization algorithm for the data with the given penalty matrix.'''
        delta = converge + 1
        
        if initial_values is None:
            beta = np.zeros(data.shape)
            prior_prob = np.exp(beta) / (1 + np.exp(beta))
            u = initial_values
        else:
            beta = initial_values['beta']
            prior_prob = initial_values['c']
            u = initial_values['u']

        prev_nll = 0
        cur_step = 0
        
        while delta > converge and cur_step < max_steps:
            if verbose:
                print 'Step #{0}'.format(cur_step)

            if verbose:
                print '\tE-step...'

            # Get the likelihood weights vector (E-step)
            post_prob = self._e_step(data, prior_prob)

            if verbose:
                print '\tM-step...'

            # Find beta using an alternating Taylor approximation and convex optimization (M-step)
            beta, u = self._m_step(beta, prior_prob, post_prob, penalties, _lambda,
                                   m_converge, m_max_steps,
                                   cd_converge, cd_max_steps,
                                   verbose, dual_solver,
                                   admm_adaptive=admm_adaptive,
                                   admm_inflate=admm_inflate,
                                   admm_alpha=admm_alpha,
                                   u0=u)

            # Get the signal probabilities
            prior_prob = ilogit(beta)
            cur_nll = self._data_negative_log_likelihood(data, prior_prob)

            if dual_solver == 'admm':
                # Get the negative log-likelihood of the data given our new parameters
                cur_nll += _lambda * np.abs(u['r']).sum()
            
            # Track the change in log-likelihood to see if we've converged
            delta = np.abs(cur_nll - prev_nll) / (prev_nll + converge)

            if verbose:
                print '\tDelta: {0}'.format(delta)

            # Track the step
            self.add_step(post_prob, beta, prior_prob, delta)

            # Increment the step counter
            cur_step += 1

            # Update the negative log-likelihood tracker
            prev_nll = cur_nll

            # DEBUGGING
            if verbose:
                print '\tbeta: [{0:.4f}, {1:.4f}]'.format(beta.min(), beta.max())
                print '\tprior_prob:    [{0:.4f}, {1:.4f}]'.format(prior_prob.min(), prior_prob.max())
                print '\tpost_prob:    [{0:.4f}, {1:.4f}]'.format(post_prob.min(), post_prob.max())
                if dual_solver != 'graph':
                    print '\tdegrees of freedom: {0}'.format((np.abs(penalties.dot(beta)) >= 1e-4).sum())

        # Return the results of the run
        return {'beta': beta, 'u': u, 'w': post_prob, 'c': prior_prob}

    def _data_negative_log_likelihood(self, data, prior_prob):
        '''Calculate the negative log-likelihood of the data given the weights.'''
        signal_weight = prior_prob * self.signal_dist.pdf(data)
        null_weight = (1-prior_prob) * self.null_dist.pdf(data)
        return -np.log(signal_weight + null_weight).sum()

    def _e_step(self, data, prior_prob):
        '''Calculate the complete-data sufficient statistics (weights vector).'''
        signal_weight = prior_prob * self.signal_dist.pdf(data)
        null_weight = (1-prior_prob) * self.null_dist.pdf(data)
        post_prob = signal_weight / (signal_weight + null_weight)
        return post_prob

    def _m_step(self, beta, prior_prob, post_prob, penalties,
                _lambda, converge, max_steps,
                cd_converge, cd_max_steps,
                verbose, dual_solver, u0=None,
                admm_alpha=1., admm_inflate=2., admm_adaptive=False):
        '''
        Alternating Second-order Taylor-series expansion about the current iterate
        and coordinate descent to optimize Beta.
        '''
        prev_nll = self._m_log_likelihood(post_prob, beta)
        delta = converge + 1
        u = u0
        cur_step = 0
        while delta > converge and cur_step < max_steps:
            if verbose:
                print '\t\tM-Step iteration #{0}'.format(cur_step)
                print '\t\tTaylor approximation...'

            # Cache the exponentiated beta
            exp_beta = np.exp(beta)

            # Form the parameters for our weighted least squares
            if dual_solver != 'admm' and dual_solver != 'graph':
                # weights is a diagonal matrix, represented as a vector for efficiency
                weights = 0.5 * exp_beta / (1 + exp_beta)**2
                y = (1+exp_beta)**2 * post_prob / exp_beta + beta - (1 + exp_beta)
                if verbose:
                    print '\t\tForming dual...'
                x = np.sqrt(weights) * y
                A = (1. / np.sqrt(weights))[:,np.newaxis] * penalties.T
            else:
                weights = prior_prob * (1 - prior_prob)
                y = beta - (prior_prob - post_prob) / weights

            if dual_solver == 'cd':
                # Solve the dual via coordinate descent
                u = self._u_coord_descent(x, A, _lambda, cd_converge, cd_max_steps, verbose > 1, u0=u)
            elif dual_solver == 'sls':
                # Solve the dual via sequential least squares
                u = self._u_slsqp(x, A, _lambda, verbose > 1, u0=u)
            elif dual_solver == 'lbfgs':
                # Solve the dual via L-BFGS-B
                u = self._u_lbfgsb(x, A, _lambda, verbose > 1, u0=u)
            elif dual_solver == 'admm':
                # Solve the dual via alternating direction methods of multipliers
                #u = self._u_admm_1dfusedlasso(y, weights, _lambda, cd_converge, cd_max_steps, verbose > 1, initial_values=u)
                #u = self._u_admm(y, weights, _lambda, penalties, cd_converge, cd_max_steps, verbose > 1, initial_values=u)
                u = self._u_admm_lucache(y, weights, _lambda, penalties, cd_converge, cd_max_steps,
                                        verbose > 1, initial_values=u, inflate=admm_inflate,
                                        adaptive=admm_adaptive, alpha=admm_alpha)
                beta = u['x']
            elif dual_solver == 'graph':
                u = self._graph_fused_lasso(y, weights, _lambda, penalties[0], penalties[1], penalties[2], penalties[3], cd_converge, cd_max_steps, max(0, verbose - 1), admm_alpha, admm_inflate, initial_values=u)
                beta = u['beta']
            else:
                raise Exception('Unknown solver: {0}'.format(dual_solver))

            if dual_solver != 'admm' and dual_solver != 'graph':
                # Back out beta from the dual solution
                beta = y - (1. / weights) * penalties.T.dot(u)

            # Get the current log-likelihood
            cur_nll = self._m_log_likelihood(post_prob, beta)

            # Track the convergence
            delta = np.abs(prev_nll - cur_nll) / (prev_nll + converge)

            if verbose:
                print '\t\tM-step delta: {0}'.format(delta)

            # Increment the step counter
            cur_step += 1

            # Update the negative log-likelihood tracker
            prev_nll = cur_nll

        return beta, u
    
    def _m_log_likelihood(self, post_prob, beta):
        '''Calculate the log-likelihood of the betas given the weights and data.'''
        return (np.log(1 + np.exp(beta)) - post_prob * beta).sum()

    def _graph_fused_lasso(self, y, weights, _lambda, ntrails, trails, breakpoints, edges, converge, max_steps, verbose, alpha, inflate, initial_values=None):
        '''Solve for u using a super fast graph fused lasso library that has an optimized ADMM routine.'''
        if verbose:
            print '\t\tSolving via Graph Fused Lasso'
        # if initial_values is None:
        #     beta = np.zeros(y.shape, dtype='double')
        #     z = np.zeros(breakpoints[-1], dtype='double')
        #     u = np.zeros(breakpoints[-1], dtype='double')
        # else:
        #     beta = initial_values['beta']
        #     z = initial_values['z']
        #     u = initial_values['u']
        # n = y.shape[0]
        # self.graphfl_weight(n, y, weights, ntrails, trails, breakpoints, _lambda, alpha, inflate, max_steps, converge, beta, z, u)
        # return {'beta': beta, 'z': z, 'u': u }
        self.solver.alpha = alpha
        self.solver.inflate = inflate
        self.solver.maxsteps = max_steps
        self.solver.converge = converge
        self.solver.set_data(y, edges, ntrails, trails, breakpoints, weights=weights)
        if initial_values is not None:
            self.solver.beta = initial_values['beta']
            self.solver.z = initial_values['z']
            self.solver.u = initial_values['u']
        self.solver.solve(_lambda)
        return {'beta': self.solver.beta, 'z': self.solver.z, 'u': self.solver.u }
        

    def _u_admm_lucache(self, y, weights, _lambda, D, converge_threshold, max_steps, verbose, alpha=1.8, initial_values=None,
                            inflate=2., adaptive=False):
        '''Solve for u using alternating direction method of multipliers with a cached LU decomposition.'''
        if verbose:
            print '\t\tSolving u via Alternating Direction Method of Multipliers'

        n = len(y)
        m = D.shape[0]
        a = inflate * _lambda # step-size parameter

        # Initialize primal and dual variables from warm start
        if initial_values is None:
            # Graph Laplacian
            L = csc_matrix(D.T.dot(D) + csc_matrix(np.eye(n)))

            # Cache the LU decomposition
            lu_factor = sla.splu(L, permc_spec='MMD_AT_PLUS_A')
            
            x = np.array([y.mean()] * n) # likelihood term
            z = np.zeros(n) # slack variable for likelihood
            r = np.zeros(m) # penalty term
            s = np.zeros(m) # slack variable for penalty
            u_dual = np.zeros(n) # scaled dual variable for constraint x = z
            t_dual = np.zeros(m) # scaled dual variable for constraint r = s
        else:
            lu_factor = initial_values['lu_factor']
            x = initial_values['x']
            z = initial_values['z']
            r = initial_values['r']
            s = initial_values['s']
            u_dual = initial_values['u_dual']
            t_dual = initial_values['t_dual']

        primal_trace = []
        dual_trace = []
        converged = False
        cur_step = 0
        D_full = D
        while not converged and cur_step < max_steps:
            # Update x
            x = (weights * y + a * (z - u_dual)) / (weights + a)
            x_accel = alpha * x + (1 - alpha) * z # over-relaxation

            # Update constraint term r
            arg = s - t_dual
            local_lambda = (_lambda - np.abs(arg) / 2.).clip(0) if adaptive else _lambda
            r = _soft_threshold(arg, local_lambda / a)
            r_accel = alpha * r + (1 - alpha) * s

            # Projection to constraint set
            arg = x_accel + u_dual + D.T.dot(r_accel + t_dual)
            z_new = lu_factor.solve(arg)
            s_new = D.dot(z_new)
            dual_residual_u = a * (z_new - z)
            dual_residual_t = a * (s_new - s)
            z = z_new
            s = s_new

            # Dual update
            primal_residual_x = x_accel - z
            primal_residual_r = r_accel - s
            u_dual = u_dual + primal_residual_x
            t_dual = t_dual + primal_residual_r

            # Check convergence
            primal_resnorm = np.sqrt((np.array([i for i in primal_residual_x] + [i for i in primal_residual_r])**2).mean())
            dual_resnorm = np.sqrt((np.array([i for i in dual_residual_u] + [i for i in dual_residual_t])**2).mean())
            primal_trace.append(primal_resnorm)
            dual_trace.append(dual_resnorm)
            converged = dual_resnorm < converge_threshold and primal_resnorm < converge_threshold

            if primal_resnorm > 5 * dual_resnorm:
                a *= inflate
                u_dual /= inflate
                t_dual /= inflate
            elif dual_resnorm > 5 * primal_resnorm:
                a /= inflate
                u_dual *= inflate
                t_dual *= inflate

            # Update the step counter
            cur_step += 1

            if verbose and cur_step % 100 == 0:
                print '\t\t\tStep #{0}: dual_resnorm: {1:.6f} primal_resnorm: {2:.6f}'.format(cur_step, dual_resnorm, primal_resnorm)

        return {'x': x, 'r': r, 'z': z, 's': s, 'u_dual': u_dual, 't_dual': t_dual,
                'primal_trace': primal_trace, 'dual_trace': dual_trace, 'steps': cur_step,
                'lu_factor': lu_factor}

    def _u_admm(self, y, weights, _lambda, D, converge_threshold, max_steps, verbose, alpha=1.0, initial_values=None):
        '''Solve for u using alternating direction method of multipliers.'''
        if verbose:
            print '\t\tSolving u via Alternating Direction Method of Multipliers'

        n = len(y)
        m = D.shape[0]

        a = _lambda # step-size parameter

        # Set up system involving graph Laplacian
        L = D.T.dot(D)
        W_over_a = np.diag(weights / a)
        x_denominator = W_over_a + L
        #x_denominator = sparse.linalg.inv(W_over_a + L)

        # Initialize primal and dual variables
        if initial_values is None:
            x = np.array([y.mean()] * n)
            z = np.zeros(m)
            u = np.zeros(m)
        else:
            x = initial_values['x']
            z = initial_values['z']
            u = initial_values['u']

        primal_trace = []
        dual_trace = []
        converged = False
        cur_step = 0
        while not converged and cur_step < max_steps:
            # Update x
            x_numerator = 1.0 / a * weights * y + D.T.dot(a * z - u)
            x = np.linalg.solve(x_denominator, x_numerator)
            Dx = D.dot(x)

            # Update z
            Dx_relaxed = alpha * Dx + (1 - alpha) * z # over-relax Dx
            z_new = _soft_threshold(Dx_relaxed + u / a, _lambda / a)
            dual_residual = a * D.T.dot(z_new - z)
            z = z_new
            primal_residual = Dx_relaxed - z

            # Update u
            u = u + a * primal_residual

            # Check convergence
            primal_resnorm = np.sqrt((primal_residual ** 2).mean())
            dual_resnorm = np.sqrt((dual_residual ** 2).mean())
            primal_trace.append(primal_resnorm)
            dual_trace.append(dual_resnorm)
            converged = dual_resnorm < converge_threshold and primal_resnorm < converge_threshold

            # Update step-size parameter based on norm of primal and dual residuals
            # This is the varying penalty extension to standard ADMM
            a *= 2 if primal_resnorm > 10 * dual_resnorm else 0.5

            # Recalculate the x_denominator since we changed the step-size
            # TODO: is this worth it? We're paying a matrix inverse in exchange for varying the step size
            #W_over_a = sparse.dia_matrix(np.diag(weights / a))
            W_over_a = np.diag(weights / a)
            #x_denominator = sparse.linalg.inv(W_over_a + L)
            
            # Update the step counter
            cur_step += 1

            if verbose and cur_step % 100 == 0:
                print '\t\t\tStep #{0}: dual_resnorm: {1:.6f} primal_resnorm: {2:.6f}'.format(cur_step, dual_resnorm, primal_resnorm)

        dof = np.sum(Dx > converge_threshold) + 1.
        AIC = np.sum((y - x)**2) + 2 * dof

        return {'x': x, 'z': z, 'u': u, 'dof': dof, 'AIC': AIC}

    def _u_admm_1dfusedlasso(self, y, W, _lambda, converge_threshold, max_steps, verbose, alpha=1.0, initial_values=None):
        '''Solve for u using alternating direction method of multipliers. Note that this method only works for the 1-D fused lasso case.'''
        if verbose:
            print '\t\tSolving u via Alternating Direction Method of Multipliers (1-D fused lasso)'

        n = len(y)
        m = n - 1

        a = _lambda

        # The D matrix is the first-difference operator. K is the matrix (W + a D^T D)
        # where W is the diagonal matrix of weights. We use a tridiagonal representation
        # of K.
        Kd = np.array([a] + [2*a] * (n-2) + [a]) + W # diagonal entries
        Kl = np.array([-a] * (n-1)) # below the diagonal
        Ku = np.array([-a] * (n-1)) # above the diagonal

        # Initialize primal and dual variables
        if initial_values is None:
            x = np.array([y.mean()] * n)
            z = np.zeros(m)
            u = np.zeros(m)
        else:
            x = initial_values['x']
            z = initial_values['z']
            u = initial_values['u']

        primal_trace = []
        dual_trace = []
        converged = False
        cur_step = 0
        while not converged and cur_step < max_steps:
            # Update x
            out = _1d_fused_lasso_crossprod(a*z - u)
            x = tridiagonal_solve(Kl, Ku, Kd, W * y + out)
            Dx = np.ediff1d(x)

            # Update z
            Dx_hat = alpha * Dx + (1 - alpha) * z # Over-relaxation
            z_new = _soft_threshold(Dx_hat + u / a, _lambda / a)
            dual_residual = a * _1d_fused_lasso_crossprod(z_new - z)
            z = z_new
            primal_residual = Dx - z
            #primal_residual = Dx_hat - z

            # Update u
            u = (u + a * primal_residual).clip(-_lambda, _lambda)

            # Check convergence
            primal_resnorm = np.sqrt((primal_residual ** 2).mean())
            dual_resnorm = np.sqrt((dual_residual ** 2).mean())
            primal_trace.append(primal_resnorm)
            dual_trace.append(dual_resnorm)
            converged = dual_resnorm < converge_threshold and primal_resnorm < converge_threshold
            
            # Update step-size parameter based on norm of primal and dual residuals
            a *= 2 if primal_resnorm > 10 * dual_resnorm else 0.5
            Kd = np.array([a] + [2*a] * (n-2) + [a]) + W # diagonal entries
            Kl = np.array([-a] * (n-1)) # below the diagonal
            Ku = np.array([-a] * (n-1)) # above the diagonal

            cur_step += 1

            if verbose and cur_step % 100 == 0:
                print '\t\t\tStep #{0}: dual_resnorm: {1:.6f} primal_resnorm: {2:.6f}'.format(cur_step, dual_resnorm, primal_resnorm)

        dof = np.sum(Dx > converge_threshold) + 1.
        AIC = np.sum((y - x)**2) + 2 * dof

        return {'x': x, 'z': z, 'u': u, 'dof': dof, 'AIC': AIC}


    def _u_coord_descent(self, x, A, _lambda, converge, max_steps, verbose, u0=None):
        '''Solve for u using coordinate descent.'''
        if verbose:
            print '\t\tSolving u via Coordinate Descent'
        
        u = u0 if u0 is not None else np.zeros(A.shape[1])

        l2_norm_A = (A * A).sum(axis=0)
        r = x - A.dot(u)
        delta = converge + 1
        prev_objective = _u_objective_func(u, x, A)
        cur_step = 0
        while delta > converge and cur_step < max_steps:
            # Update each coordinate one at a time.
            for coord in xrange(len(u)):
                prev_u = u[coord]
                next_u = prev_u + A.T[coord].dot(r) / l2_norm_A[coord]
                u[coord] = min(_lambda, max(-_lambda, next_u))
                r += A.T[coord] * prev_u - A.T[coord] * u[coord]

            # Track the change in the objective function value
            cur_objective = _u_objective_func(u, x, A)
            delta = np.abs(prev_objective - cur_objective) / (prev_objective + converge)

            if verbose and cur_step % 100 == 0:
                print '\t\t\tStep #{0}: Objective: {1:.6f} CD Delta: {2:.6f}'.format(cur_step, cur_objective, delta)

            # Increment the step counter and update the previous objective value
            cur_step += 1
            prev_objective = cur_objective

        return u

    def _u_slsqp(self, x, A, _lambda, verbose, u0=None):
        '''Solve for u using sequential least squares.'''
        if verbose:
            print '\t\tSolving u via Sequential Least Squares'

        if u0 is None:
            u0 = np.zeros(A.shape[1])

        # Create our box constraints
        bounds = [(-_lambda, _lambda) for u0_i in u0]

        results = minimize(_u_objective_func, u0,
                           args=(x, A),
                           jac=_u_objective_deriv,
                           bounds=bounds,
                           method='SLSQP',
                           options={'disp': False, 'maxiter': 1000})

        if verbose:
            print '\t\t\t{0}'.format(results.message)
            print '\t\t\tFunction evaluations: {0}'.format(results.nfev)
            print '\t\t\tGradient evaluations: {0}'.format(results.njev)
            print '\t\t\tu: [{0}, {1}]'.format(results.x.min(), results.x.max())

        return results.x

    def _u_lbfgsb(self, x, A, _lambda, verbose, u0=None):
        '''Solve for u using L-BFGS-B.'''
        if verbose:
            print '\t\tSolving u via L-BFGS-B'

        if u0 is None:
            u0 = np.zeros(A.shape[1])

        # Create our box constraints
        bounds = [(-_lambda, _lambda) for _ in u0]

        # Fit
        results = minimize(_u_objective_func, u0, args=(x, A), method='L-BFGS-B', bounds=bounds, options={'disp': verbose})

        return results.x

    def plateau_regression(self, plateaus, data, grid_map=None, verbose=False):
        '''Perform unpenalized 1-d regression for each of the plateaus.'''
        weights = np.zeros(data.shape)
        for i,(level,p) in enumerate(plateaus):
            if verbose:
                print '\tPlateau #{0}'.format(i+1)
            
            # Get the subset of grid points for this plateau
            if grid_map is not None:
                plateau_data = np.array([data[grid_map[x,y]] for x,y in p])
            else:
                plateau_data = np.array([data[x,y] for x,y in p])

            w = single_plateau_regression(plateau_data, self.signal_dist, self.null_dist)
            for idx in p:
                weights[idx if grid_map is None else grid_map[idx[0], idx[1]]] = w
        posteriors = self._e_step(data, weights)
        weights = weights.flatten()
        return (weights, posteriors)


def _u_objective_func(u, x, A):
    return np.linalg.norm(x - A.dot(u))**2

def _u_objective_deriv(u, x, A):
    return 2*A.T.dot(A.dot(u) - x)

def _u_slsqp_constraint_func(idx, _lambda, u):
    '''Constraint function for the i'th value of u.'''
    return np.array([_lambda - np.abs(u[idx])])

def _u_slsqp_constraint_deriv(idx, u):
    jac = np.zeros(len(u))
    jac[idx] = -np.sign(u[idx])
    return jac

def _1d_fused_lasso_crossprod(x):
    '''Efficiently compute the cross-product D^T x, where D is the first-differences matrix.''' 
    return -np.ediff1d(x, to_begin=x[0], to_end=-x[-1])

def _soft_threshold(x, _lambda):
    return np.sign(x) * (np.abs(x) - _lambda).clip(0)

## Tri-Diagonal Matrix Algorithm (a.k.a Thomas algorithm) solver
## Source: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
def tridiagonal_solve(a,b,c,f):
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0] * n
 
    for i in range(n-1):
        alpha.append(-b[i]/(a[i]*alpha[i] + c[i]))
        beta.append((f[i] - a[i]*beta[i])/(a[i]*alpha[i] + c[i]))
 
    x[n-1] = (f[n-1] - a[n-2]*beta[n-1])/(c[n-1] + a[n-2]*alpha[n-1])
 
    for i in reversed(range(n-1)):
        x[i] = alpha[i+1]*x[i+1] + beta[i+1]
 
    return np.array(x)

def ilogit(x):
    return 1. / (1. + np.exp(-x))

def calc_plateaus(beta, rel_tol=1e-4, edges=None, verbose=0):
    '''Calculate the plateaus (degrees of freedom) of a 1d or 2d grid of beta values in linear time.'''
    to_check = deque(itertools.product(*[range(x) for x in beta.shape])) if edges is None else deque(xrange(len(beta)))
    check_map = np.zeros(beta.shape, dtype=bool)
    check_map[np.isnan(beta)] = True
    plateaus = []

    if verbose:
        print '\tCalculating plateaus...'

    if verbose > 1:
        print '\tIndices to check {0} {1}'.format(len(to_check), check_map.shape)

    # Loop until every beta index has been checked
    while to_check:
        if verbose > 1:
            print '\t\tPlateau #{0}'.format(len(plateaus) + 1)

        # Get the next unchecked point on the grid
        idx = to_check.popleft()

        # If we already have checked this one, just pop it off
        while to_check and check_map[idx]:
            try:
                idx = to_check.popleft()
            except:
                break

        # Edge case -- If we went through all the indices without reaching an unchecked one.
        if check_map[idx]:
            break

        # Create the plateau and calculate the inclusion conditions
        cur_plateau = set([idx])
        cur_unchecked = deque([idx])
        val = beta[idx]
        min_member = val - rel_tol
        max_member = val + rel_tol

        # Check every possible boundary of the plateau
        while cur_unchecked:
            idx = cur_unchecked.popleft()
            
            # neighbors to check
            local_check = []

            # Generic graph case
            if edges is not None:
                local_check.extend(edges[idx])

            # 1d case -- check left and right
            elif len(beta.shape) == 1:
                if idx[0] > 0:
                    local_check.append(idx[0] - 1) # left
                if idx[0] < beta.shape[0] - 1:
                    local_check.append(idx[0] + 1) # right

            # 2d case -- check left, right, up, and down
            elif len(beta.shape) == 2:
                if idx[0] > 0:
                    local_check.append((idx[0] - 1, idx[1])) # left
                if idx[0] < beta.shape[0] - 1:
                    local_check.append((idx[0] + 1, idx[1])) # right
                if idx[1] > 0:
                    local_check.append((idx[0], idx[1] - 1)) # down
                if idx[1] < beta.shape[1] - 1:
                    local_check.append((idx[0], idx[1] + 1)) # up

            # Only supports 1d and 2d cases for now
            else:
                raise Exception('Degrees of freedom calculation does not currently support more than 2 dimensions unless edges are specified explicitly. ({0} given)'.format(len(beta.shape)))

            # Check the index's unchecked neighbors
            for local_idx in local_check:
                if not check_map[local_idx] \
                    and beta[local_idx] >= min_member \
                    and beta[local_idx] <= max_member:
                        # Label this index as being checked so it's not re-checked unnecessarily
                        check_map[local_idx] = True

                        # Add it to the plateau and the list of local unchecked locations
                        cur_unchecked.append(local_idx)
                        cur_plateau.add(local_idx)

        # Track each plateau's indices
        plateaus.append((val, cur_plateau))

    # Returns the list of plateaus and their values
    return plateaus

def plateau_loss_func(c, data, signal_dist, null_dist):
    '''The negative log-likelihood function for a plateau.'''
    return -np.log(c * signal_dist.pdf(data) + (1. - c) * null_dist.pdf(data)).sum()

def single_plateau_regression(data, signal_dist, null_dist):
    '''Perform unpenalized 1-d regression on all of the points in a plateau.'''
    return minimize_scalar(plateau_loss_func, args=(data, signal_dist, null_dist), bounds=(0,1), method='Bounded').x




        


