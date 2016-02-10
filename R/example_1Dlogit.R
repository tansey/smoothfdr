# must install glmgen, see Github page:
# https://github.com/statsmaths/glmgen
library(glmgen)

# logit utilities (forward and inverse transform)
flogit = function(x) log(x/(1-x))
ilogit = function(x) 1/{1+exp(-x)}

# Simulated data settings
n = 5000
beta_true = rep(-4,n) 
beta_true[1500:2000] = 0   # an enriched region 500 sites long
w_true = ilogit(beta_true)
gamma_true = rbinom(n, 1, w_true)
z = rnorm(n)
z[gamma_true == 1] = rnorm(sum(gamma_true), 0, 3)
plot(z, xlab='X', ylab='Z score', pch=19, col=rgb(0.2,0.2,0.2,0.2))

# We know the null and alternative densities
f0 = dnorm(z,0,1)
f1 = dnorm(z,0,3)

# Fixed lambda for now
lambda = 5

# Initial settings for algorithm
drift = 1
rel_tol = 1e-4
beta_hat = rep(0,n)
prior_prob = ilogit(beta_hat)
objective_old = sum(log(prior_prob*f1 + (1-prior_prob)*f0))

while(drift > rel_tol) {
	# E step
	prior_prob = ilogit(beta_hat)
	m1 = prior_prob*f1
	m0 = (1-prior_prob)*f0
	post_prob = m1/(m1+m0)
	
	# M step
	ebeta = exp(beta_hat)
	weights = ebeta/{(1+ebeta)^2}
	y = {(1+ebeta)^2}*post_prob/ebeta + beta_hat - (1+ebeta)
	weights = prior_prob*(1-prior_prob)
	y = beta_hat - (prior_prob - post_prob)/weights
	
	# Solve the 1D FL problem using the subroutine in glmgen
	fl0 = trendfilter(drop(y), weights=drop(weights), k = 0, family='gaussian', lambda=lambda)

	beta_hat = fl0$beta
	prior_prob = ilogit(beta_hat)

	objective_new = sum(log(prior_prob*f1 + (1-prior_prob)*f0))
	drift = abs(objective_old - objective_new)/(abs(objective_old) + rel_tol)
	objective_old = objective_new
}


# Plot the results

par(mfrow=c(1,2))
plot(z, pch=19, cex=0.6,col=rgb(0.2,0.2,0.2,0.2), las=1,
	cex.axis=0.8, xlab='x', ylab='z score')
rug({1:n}[which(gamma_true==1)])

prob_grid = c(0.01,0.05,0.1,0.25,0.5)
plot(flogit(w_true), las=1, ylim=range(flogit(prob_grid)),
	type='l', col='black', lwd=4, axes=FALSE)
lines(beta_hat, col='gray', lwd=2)
abline(h=flogit(mean(w_true)), lty='dotted')
axis(1, cex.axis=0.8, las=1)
axis(2, cex.axis=0.8, at = flogit(prob_grid), labels=prob_grid, las=1)
