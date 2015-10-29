# Benjamini-Hochberg with two-sided p-values
BenjaminiHochberg = function(zscores, fdr_level) {
# zscores is a vector of z scores
# fdr_level is the desired level (e.g. 0.1) of control over FDR
# returns a binary vector where 0=nofinding, 1=finding at given FDR level
    N = length(zscores)
    pval2 = 2*pmin(pnorm(zscores), 1- pnorm(zscores))
    cuts = (1:N)*fdr_level/N
    bhdiff = sort(pval2)-cuts
    bhcutind2 = max(which(bhdiff < 0))
    bhcut2 = sort(pval2)[bhcutind2]
    0+{pval2 <= bhcut2}
}

args <- commandArgs(trailingOnly = TRUE)
z = as.numeric(t(as.matrix(read.csv(args[1], header=FALSE))))#testing
discoveries = BenjaminiHochberg(z, 0.1)
write.table(discoveries, args[2], row.names=FALSE, col.names=FALSE, sep=",")