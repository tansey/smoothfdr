fdrsmooth_cholcache = function(z, lambda, fl0 = NULL) {
	
	
}

lambda = .38
rel_tol = 1e-6
travel = 1
prior_prob = ilogit(beta_hat)
old_objective = sum(-log(prior_prob*f1 + (1-prior_prob)*f0)) + lambda * sum(abs(fl0$r))
converged = FALSE
while(!converged) {
	# E step
	m1 = prior_prob*f1
	m0 = (1-prior_prob)*f0
	post_prob = m1/(m1+m0)
	
	# Partial M step: one ADMM iteration, analogous to a single Newton iteration
	weights = prior_prob*(1-prior_prob)
	y = beta_hat - (prior_prob - post_prob)/weights
	fl0 = fit_graphfusedlasso_cholcache(y, lambda=lambda, D=D, chol_factor=chol_factor, weights=weights,
		initial_values = fl0, rel_tol = rel_tol, alpha=1.8, adaptive=FALSE)
	beta_hat = fl0$x
	prior_prob = ilogit(beta_hat)
	
	# Check relative convergence
	new_objective = sum(-log(prior_prob*f1 + (1-prior_prob)*f0)) + lambda * sum(abs(fl0$r))
	travel = abs(new_objective - old_objective)
	old_objective = new_objective
	converged = {travel/(old_objective + rel_tol) < rel_tol}
}