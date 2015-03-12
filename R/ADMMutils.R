library(limSolve)

ilogit = function(x) 1/{1+exp(-x)}

makeD1 = function(d) {
# Construct the first-difference matrix
	D1 = diag(1,d-1)
	D1 = rbind(D1,0)
	D1 = cbind(0,D1)
	diag(D1) = -1
	D1 = D1[-nrow(D1),]
	D1
}

Dcrossx = function(x) {
# efficient compute cross-product D^T x,
# where D is the first-difference matrix
	n = length(x) + 1
	out = rep(0, n)
	out[1] = -x[1]
	out[2:(n-1)] = rev(diff(rev(x)))
	out[n] = x[n-1]
	out
}

softthresh = function(x, lambda) {
	return(sign(x)*pmax(0, abs(x) - lambda))
}

fit_1dfusedlasso_ADMM = function(y, lambda, weights=NULL, initial_values = NULL, iter_max = 10000, rel_tol = 1e-4, alpha=1.0) {
	# Fits the 1D weighted fused lasso by ADMM
	n = length(y)
	m = n-1
	
	if(missing(weights)) {
		weights = rep(1, n)
	}
	
	# Step-size parameter
	a = lambda
	
	# the D matrix is the first-difference operator
	# K is the matrix (W + a D^T D)
	# where W is the diagonal matrix of weights.
	# We use a tridiagonal representation of K
	Kd = c(a, rep(2*a, n-2), a) + weights # diagonal entries
	Kl = rep(-a, n-1) # below the diagonal
	Ku = rep(-a, n-1) # above the diagonal

	# print("K:")
	# print(length(Kd))
	# print(length(Kl))
	# print(length(Ku))
	
	# Initialize primal and dual variables
	if(missing(initial_values)) {
		x = rep(mean(y), n)
		z = rep(0,m)
		u = rep(0,m)
	} else {
		x = initial_values$x
		z = initial_values$z
		u = initial_values$u
		u = pmin(lambda, pmax(-lambda, u)) # ensure feasibility of starting point
	}
	
	primal_trace = NULL
	dual_trace = NULL
	converged = FALSE
	counter = 0
	while(!converged & counter < iter_max) {
		
		# Update x
		out = Dcrossx(a*z - u)
		x = Solve.tridiag(Kl, Kd, Ku, weights*y + out)
		Dx = diff(x)

		# Update z: over-relaxation?
		Dx_hat = alpha*Dx + (1-alpha)*z
		z_new = softthresh(Dx_hat + u/a, lambda/a)
		dual_residual = a * Dcrossx(z_new - z)
		z = z_new
		primal_residual = Dx - z
		
		# Update u
		u = u + a*primal_residual
		
		# Check convergence
		primal_resnorm = sqrt(mean(primal_residual^2))
		dual_resnorm = sqrt(mean(dual_residual^2))
		primal_trace = c(primal_trace, primal_resnorm)
		dual_trace = c(dual_trace, dual_resnorm)
		if(dual_resnorm < rel_tol && primal_resnorm < rel_tol) {
			converged=TRUE
		}
		counter = counter+1
		
		# Update step-size parameter based on norm of primal and dual residuals
		if(primal_resnorm > 10*dual_resnorm) {
			a = 2*a
			Kd = c(a, rep(2*a, n-2), a) + weights # diagonal entries
			Kl = rep(-a, n-1) # below the diagonal
			Ku = rep(-a, n-1) # above the diagonal
		} else if(dual_resnorm > 10*primal_resnorm) {
			a = a/2
			Kd = c(a, rep(2*a, n-2), a) + weights # diagonal entries
			Kl = rep(-a, n-1) # below the diagonal
			Ku = rep(-a, n-1) # above the diagonal
		}

	}
	dof = sum(Dx > rel_tol) + 1
	AIC =  sum((y-x)^2) + 2*dof
	
	list(x=x, z=z, u=u, dof=dof, AIC=AIC)
}
