import numpy as np
from scipy.constants import c #m/s
from math import pi
from scipy.stats import norm

def percentile(perc, pdf, xarray): #May not be perfect due to binning...
	sum = 0.
	idx = 0
	tot=pdf.sum()
	while sum<perc:
		sum = sum + pdf[idx]/tot
		idx=idx+1
	return xarray[idx]

def lnlike(H0, z, pb_gal, distmu, diststd, distnorm, pixarea, H0_min, H0_max):
    distgal = (c/1000.)*z/H0 #THIS NEEDS TO BECOME A DISTANCE WITH FULL COSMOLOGY!!! #cosmo = FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0=2.725)
    like_gals = pb_gal * distnorm * norm(distmu, diststd).pdf(distgal)*z**2 # /cosmo.comoving_volume(z).value #Maybe this is not clever as it takes a log of exp...
    normalization = H0**3
    return np.log(np.sum(like_gals)/normalization)

def lnprior(H0, H0_min, H0_max):
	if H0_min < H0 < H0_max:
		return 0.0
	return -np.inf

def lnprob(H0, z, pb_gal, distmu, diststd, distnorm, pixarea, H0_min, H0_max):
	lp = lnprior(H0, H0_min, H0_max)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(H0, z, pb_gal, distmu, diststd, distnorm, pixarea, H0_min, H0_max)
