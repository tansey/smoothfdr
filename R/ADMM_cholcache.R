# Fits the weighted fused lasso by ADMM where D is the discrete difference operator on a graph
# D is a sparse matrix of class 'dgCMatrix' [package "Matrix"]

fit_graphfusedlasso_cholcache = function(y, lambda, D, chol_factor = NULL, weights=NULL, initial_values = NULL, iter_max = 10000, rel_tol = 1e-4, alpha=1.0, inflate=2, adaptive=FALSE) {
	require(Matrix)
	
	n = length(y)
	m = nrow(D)
	a = 2*lambda # step-size parameter
		
	if(missing(weights)) {
		weights = rep(1, n)
	}
	
	# Check if we need a Cholesky decomp of system involving graph Laplacian
	if(missing(chol_factor)) {
		L = Matrix::crossprod(D)
		chol_factor = Matrix::Cholesky(L + Matrix::Diagonal(n))
	}

	# Initialize primal and dual variables from warm start
	if(missing(initial_values)) {
		x = rep(0, n) # likelihood term
		z = rep(0, n) # slack variable for likelihood
		r = rep(0, m) # penalty term
		s = rep(0, m) # slack variable for penalty
		u_dual = rep(0,n) # scaled dual variable for constraint x = z
		t_dual = rep(0,m) # scaled dual variable for constraint r = s
	} else {
		x = initial_values$x
		z = initial_values$z
		r = initial_values$r
		s = initial_values$s
		t_dual = initial_values$t_dual
		u_dual = initial_values$u_dual
	}
	
	primal_trace = NULL
	dual_trace = NULL
	converged = FALSE
	counter = 0
	while(!converged & counter < iter_max) {
		
		# Update x
		x = {weights * y + a*(z - u_dual)}/{weights + a}
		x_accel = alpha*x + (1-alpha)*z
		
		# Update constraint term r
		arg = s - t_dual
		if(adaptive) {
			local_lambda = 1/{1+(lambda)*abs(arg)}  # Minimax-concave penalty instead?
		} else {
			local_lambda = lambda
		}
		r = softthresh(arg, local_lambda/a)
		r_accel = alpha*r + (1-alpha)*s
		
		# Projection to constraint set
		arg = x_accel + u_dual + Matrix::crossprod(D, r_accel + t_dual)
		z_new = drop(Matrix::solve(chol_factor, arg))
		s_new = as.numeric(D %*% z_new)
		dual_residual_u = a*(z_new - z)
		dual_residual_t = a*(s_new - s)
		z = z_new
		s = s_new
		
		# Dual update
		primal_residual_x = x_accel - z
		primal_residual_r = r_accel - s
		u_dual = u_dual + primal_residual_x
		t_dual = t_dual + primal_residual_r
		
		# Check convergence
		primal_resnorm = sqrt(mean(c(primal_residual_x, primal_residual_r)^2))
		dual_resnorm = sqrt(mean(c(dual_residual_u, dual_residual_t)^2))
		if(dual_resnorm < rel_tol && primal_resnorm < rel_tol) {
			converged=TRUE
		}
		primal_trace = c(primal_trace, primal_resnorm)
		dual_trace = c(dual_trace, dual_resnorm)
		counter = counter+1
		
		# Update step-size parameter based on norm of primal and dual residuals
		if(primal_resnorm > 5*dual_resnorm) {
			a = inflate*a
			u_dual = u_dual/inflate
			t_dual = t_dual/inflate
		} else if(dual_resnorm > 5*primal_resnorm) {
			a = a/inflate
			u_dual = inflate*u_dual
			t_dual = inflate*t_dual
		}
	}
	list(x=x, r=r, z=z, s=s, u_dual=u_dual, t_dual=t_dual,
		primal_trace = primal_trace, dual_trace=dual_trace, counter=counter)
}
