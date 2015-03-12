library(genlasso)
library(FDRreg)
library(locfdr)
source("ADMMutils.R")
source("ADMM_cholcache.R")

n = 128
d = n^2
D = genlasso::getD2dSparse(n,n) # appears to fastcount on rows

# Construct a true blocky 2D image
beta_true = matrix(-10, nrow=n, ncol=n)
block1_i = 10:25
block1_j = 10:25
beta_true[block1_i, block1_j] = 0
block2_i = 40:50
block2_j = 50:60
beta_true[block2_i, block2_j] = 2

# Plot the truth
mybreaks = seq(0,1,length=21)
mycols = grey(seq(1,0,length=20)^0.5)
filled.contour(1:n, 1:n, ilogit(beta_true), levels=mybreaks, col=mycols)

beta_true_vec = as.numeric(beta_true)
w_true = ilogit(beta_true_vec)
gamma_true = rbinom(d, 1, w_true)
z = rnorm(d)
z[gamma_true == 1] = rnorm(sum(gamma_true), 0, 3)
image(1:128, 1:128, matrix(z, nrow=n, ncol=n), col=mycols, xlab='', ylab='')


# Pre-compute a sparse Cholesky factorization: L^t L = D^t D + I
# Use a fill-reducing permutation to keep L as sparse as possible
chol_factor = Matrix::Cholesky(Matrix::crossprod(D) + Matrix::Diagonal(d), perm=TRUE, super=FALSE)
#image(chol_factor)

# Quick timing test
arg = runif(ncol(chol_factor), -1,1)
system.time(replicate(100, (Matrix::solve(chol_factor, arg))))

# EM
beta_hat = rep(0,d)
f0 = dnorm(z,0,1)
f1 = dnorm(z,0,3)

#####
#### Russ data example
#####

# Load in a single slice from Russ Poldrack's data
Z_scores = read.csv("../data/zscores_russdata_hslice37.csv")
Z_scores = as.matrix(Z_scores)

par(mar=c(4,4,3,1))
# Color scale
mybreaks = c(seq(0,5, length=21), 20)
mycols = c(grey(seq(1,.3,length=20)^0.5), 'red')
image(1:nrow(Z_scores), 1:ncol(Z_scores), abs(Z_scores), breaks=mybreaks, col=mycols,
	main="Raw z scores from a single horizontal section", xlab='x', ylab='y',
	cex.main=0.9,
	xlim=c(20,110), ylim=c(0, 105), las=1)


# Very high z scores
image(1:nrow(Z_scores), 1:ncol(Z_scores), 0+{abs(Z_scores) > 8}, breaks=c(-0.5,0.5,1.5), col=c('white', 'black'),
	main="Findings under FDR control", xlab='x', ylab='y',
	cex.main=0.8,
	xlim=c(20,110), ylim=c(0, 105), las=1)


# Very high z scores
image(1:nrow(Z_scores), 1:ncol(Z_scores), 0+{abs(Z_scores) > 8}, breaks=c(-0.5,0.5,1.5), col=c('white', 'black'),
	main="Findings under FDR control", xlab='x', ylab='y',
	cex.main=0.8,
	xlim=c(20,110), ylim=c(0, 105), las=1)




x_length = 128
y_length = 128
xy_grid = expand.grid(1:x_length, 1:y_length)

z_full = as.numeric(Z_scores)
brain_area = which(z_full != 0)
z = z_full[brain_area]

d = length(brain_area)

# Sanity check on the area we think is the brain
plot(xy_grid[brain_area,], pch=15, cex=0.6)

# Get the oriented incidence matrix for the retained nodes
D = genlasso::getD2dSparse(x_length,y_length) # appears to fastcount on rows
D = D[,brain_area]
scrub_edges = which(rowSums(abs(D)) != 2)
D = D[-scrub_edges,]
chol_factor = Matrix::Cholesky(Matrix::crossprod(D) + Matrix::Diagonal(d), perm=TRUE, super=FALSE)


# Fit empirical null and nonparametric alternative

# e1 = efron(z, nulltype='empirical')

e2 = locfdr(z)
mu0 = e2$fp0[3,1]
sig0 = e2$fp0[3,2] 
pr1 = prfdr(z, mu0=mu0, sig0=sig0)
f0 = pr1$f0_z
f1 = pr1$f1_z

par(mar=c(2,4,3,1))
hist(z, 200, prob=TRUE, axes=FALSE,
	main='z scores from fMRI experiment', xlab='',
	col='lightgrey', border='grey', xlim=c(-6,6), las=1)
curve(0.745231088*dnorm(x), add=TRUE, lty='dotted')
curve(0.955349473*dnorm(x,mu0,sig0), add=TRUE)
legend('topright', legend=c('Theoretical Null: 0.75*N(0, 1)', 'Empirical Null: 0.96*N(-0.1, 1.3)'),
	cex=0.6, lty=c('dotted', 'solid'), bty='n')
axis(2, las=1, tick=FALSE)
axis(1, at=-5:5, las=1, tick=FALSE, line=-1)
mtext('z', side=1, line=1)

lines(pr1$x_grid, pr1$f1_grid, lty='dashed', col='grey')

# EM
beta_hat = rep(0,d)

# Initialization
fl0 = list(x = rep(mean(z), d), # likelihood term
			z = rep(0, d), # slack variable for likelihood
			r = rep(0, nrow(D)), # penalty term
			s = rep(0, nrow(D)), # slack variable for penalty
			u_dual = rep(0,length(z)), # scaled dual variable for constraint x = z
			t_dual = rep(0,nrow(D))) # scaled dual variable for constraint r = s
		

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


Pi_smoothed = matrix(0, nrow=x_length, ncol=y_length)
Pi_smoothed[brain_area] = prior_prob

Z_empirical = (abs(Z_scores-mu0))/sig0
Z_empirical[Z_scores == 0] = 0

par(mfrow=c(2,2), mar=c(1,1,3,1))
# Color scale
mybreaks = c(seq(0,5, length=21), 20)
mycols = c(grey(seq(0.98,.1,length=21)^0.5))
image(1:nrow(Z_scores), 1:ncol(Z_scores), Z_empirical, breaks=mybreaks, col=mycols,
	main="Raw z scores from a single horizontal section", xlab='x', ylab='y',
	cex.main=0.8, axes=FALSE,
	xlim=c(110,20), ylim=c(0, 105), las=1)


# Compare with BH
is_finding_BH = BenjaminiHochberg(z, 0.05)
findings_matrixBH = matrix(0, nrow=x_length, ncol=y_length)
findings_matrixBH[brain_area[which(is_finding_BH==1)]] = 1
image(1:nrow(Z_scores), 1:ncol(Z_scores), findings_matrixBH, breaks=c(-0.5,0.5,1.5), col=c('white', 'black'),
	main="Findings using the Benjamini-Hochberg method", xlab='x', ylab='y',
	cex.main=0.8, axes=FALSE,
	xlim=c(110,20), ylim=c(0, 105), las=1)




mybreaks = c(0, seq(0.3,0.75,length=19), 1)
n_levels = 10
mybreaks = c(0,seq(1e-5,1, length=n_levels))
mycols = grey(c(1,seq(0.97,0.1,length=n_levels-1)))
image(1:nrow(Z_scores), 1:ncol(Z_scores), Pi_smoothed, breaks=mybreaks, col=mycols,
	main="Estimated local fraction of signals", xlab='x', ylab='y',
	cex.main=0.8, axes=FALSE,
	xlim=c(110,20), ylim=c(0, 105), las=1)


local_fdr = (1-prior_prob)*pr1$f0_z
local_fdr = local_fdr / {local_fdr + prior_prob*pr1$f1_z}
is_finding = {getFDR(1-local_fdr)$FDR < 0.05}
findings_matrix = matrix(0, nrow=x_length, ncol=y_length)
findings_matrix[brain_area[is_finding]] = 1

image(1:nrow(Z_scores), 1:ncol(Z_scores), findings_matrix, breaks=c(-0.5,0.5,1.5), col=c('white', 'black'),
	main="Findings using FDR smoothing", xlab='x', ylab='y',
	cex.main=0.8, axes=FALSE,
	xlim=c(110,20), ylim=c(0, 105), las=1)





write.csv(Z_empirical, file='Z_empiricalnull.csv', row.names=FALSE)
write.csv(Pi_smoothed, file='Pi_smoothed.csv', row.names=FALSE)
write.csv(findings_matrix, file='findings_matrix.csv', row.names=FALSE)
write.csv(findings_matrixBH, file='findings_matrix_BH.csv', row.names=FALSE)


par(mar=c(3,3,3,1))
filled.contour(1:n, 1:n, ilogit(beta_true), levels=mybreaks, col=mycols, main="True prior probability of signal")

zlevels = c(seq(0,2,length=25), max(z))
zcolors = c(grey(seq(1,0,length=24)^0.75), 'red')
image(1:n, 1:n, matrix(abs(z), nrow=n, ncol=n),
	breaks=zlevels, col=zcolors,
	main="Observed z score")

filled.contour(1:n, 1:n, matrix(prior_prob, nrow=n, ncol=n), levels=mybreaks, col=mycols, main="Estimated prior probability")

filled.contour(1:n, 1:n, matrix(beta_hat, nrow=n, ncol=n),  col=mycols, main="Estimated prior probability")


