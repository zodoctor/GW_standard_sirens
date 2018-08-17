import numpy as np
from scipy.constants import c #m/s
from math import pi
from scipy.stats import norm

########### H0 utils ############

def percentile(perc, pdf, xarray): #May not be perfect due to binning... Valid only for regularly spaced xarrays
    sum_pdf = 0.
    idx = 0
    tot = pdf.sum()
    while (sum_pdf<perc):
        sum_pdf = pdf[:idx].sum()/tot
        idx=idx+1
    return xarray[idx-1]

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
	return lp + lnlike(H0, z, pb_gal, distmu, diststd, distnorm, H0_min, H0_max)

########### Time delay utils ############

def sfh(t, tau):  #For now exp SFH
    return np.exp(-(t/tau))/tau

def lnlike_taud(taud, z, pb_gal, distmu, diststd, distnorm, H0, age_gal, tau_gal, norm_sfh_gal):
    distgal = (c/1000.)*z/H0 #THIS NEEDS TO BECOME A DISTANCE WITH FULL COSMOLOGY!!! #cosmo = FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0=2.725)
    t_gal = age_gal-taud
    ix_formed_gals =(t_gal>0)
    like_gals = norm_sfh_gal[ix_formed_gals] * sfh(t_gal[ix_formed_gals],tau_gal[ix_formed_gals]) * pb_gal[ix_formed_gals] * distnorm[ix_formed_gals] * norm(distmu[ix_formed_gals], diststd[ix_formed_gals]).pdf(distgal[ix_formed_gals])*z[ix_formed_gals]**2
    normalization = H0**3
    print np.log(np.sum(like_gals)/normalization)
    return np.log(np.sum(like_gals)/normalization)

def lnprior_taud(taud, age_gal, tau_gal, norm_sfh_gal , taud_min, taud_max):
    if taud_min < taud < taud_max:
        return 0.0 #  Eventually this should be either norm_sfh_gal*sfh(age_gal-taud,tau_gal) for each galaxy or the prior on cosmological parameters
    return -np.inf

def lnprob_taud(taud, z, pb_gal, distmu, diststd, distnorm, H0, age_gal, tau_gal, norm_sfh_gal, taud_min, taud_max):
    lp = lnprior_taud(taud, age_gal, tau_gal, norm_sfh_gal, taud_min, taud_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_taud(taud, z, pb_gal, distmu, diststd, distnorm, H0, age_gal, tau_gal, norm_sfh_gal)
