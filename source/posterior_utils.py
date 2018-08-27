import numpy as np
from scipy.constants import c #m/s
from math import pi
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
import scipy.integrate as integrate
from scipy.integrate import romb
import pickle
import os 

#################################
########### H0 utils ############
#################################

def gauss(mu, std, x):
    return np.exp(-(x-mu)*(x-mu)/(2*std*std))/(std*(2.*pi)**0.5)

def percentile(perc, pdf, xarray): #May not be perfect due to binning... Valid only for regularly spaced xarrays
    sum_pdf = 0.
    idx = 0
    tot = pdf.sum()
    while (sum_pdf<perc):
        sum_pdf = pdf[:idx].sum()/tot
        idx=idx+1
        if idx==len(pdf): return xarray[-1]
    return xarray[idx-1]

def perc_idx(perc, pdf): #May not be perfect due to binning... Valid only for regularly spaced xarrays
    sum_pdf = 0.
    idx = 0
    tot = pdf.sum()
    while (sum_pdf<perc):
        sum_pdf = pdf[:idx].sum()/tot
        idx=idx+1
        if idx==len(pdf): return -1
    return max(idx-1,0)

def make_blind(H0_true, outpath):
    '''
    This function check if the binary blinding factor file is already
    created. If it is: read and apply the blinding factor on H0. If not:
    generate the factor, save and apply.
    '''

    if os.path.isfile(outpath):
        B_H0 = pickle.load(open(outpath, "rb"))
        H0_blinded = H0_true*B_H0
    else:
        B_H0 = np.fabs(np.random.randn()*0.3 + 1.) #Normal with sigma=0.3, mu=1 
        outarr = np.array([B_H0])
        pickle.dump(outarr, open(outpath, "wb"))
        H0_blinded = H0_true*B_H0
    return H0_blinded

#### The likelihood has the option to have delta functions or Gaussians for redshifts, and the luminosity distance can be computed from Flat lambdaCDM or simple Hubble constant. All options are left so that the user can choose how quick the likelihood will be computed, depending also on the redshift range

def lnlike(H0, z, zerr, pb_gal, distmu, diststd, distnorm, H0_min, H0_max, z_min, z_max, zerr_use, cosmo_use, omegam=0.3):
    if ((zerr_use==False) & (cosmo_use==False)):
        distgal = (c/1000.)*z/H0
        like_gals = pb_gal * distnorm * gauss(distmu,diststd,distgal)*z**2
        normalization = H0**3
    elif ((zerr_use==False) & (cosmo_use==True)):
        cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)
        dV=cosmo.differential_comoving_volume(z).value / (1+z)**3
        distgal = cosmo.luminosity_distance(z).value 
        like_gals = pb_gal * distnorm * gauss(distmu,diststd,distgal) * dV
        normalization = 1.
    elif ((zerr_use==True) & (cosmo_use==False)):
        ngals = z.shape[0]
        like_gals = np.zeros(ngals)
        z_s = np.arange(z_min,z_max, step=0.02)
        const = (c/1000.)/H0
        normalization = H0**3
        for i in range(ngals):
            like_gals[i]= pb_gal[i] * distnorm[i] * romb( gauss(z[i], zerr[i],z_s) * gauss(distmu[i], diststd[i], const*z_s)*z[i]*z[i], dx=0.02)
    else:
        ngals = z.shape[0]
        cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)
        like_gals = np.zeros(ngals)
        z_s = np.arange(z_min,z_max, step=0.02)
        normalization = 1.
        for i in range(ngals):
            dist_gal = cosmo.luminosity_distance(z[i]).value
            dV=cosmo.differential_comoving_volume(z[i]).value / (1+z[i])**3
            like_gals[i] = pb_gal[i] * distnorm[i] * romb( gauss(z[i],zerr[i],z_s) * gauss(distmu[i],diststd[i],dist_gal) * dV, dx=0.02)

    lnlike_sum = np.log(max(1.e-300,np.sum(like_gals)/normalization))
    return lnlike_sum

#### Flat prior ####

def lnprior(H0, H0_min, H0_max):
	if H0_min < H0 < H0_max:
		return 0.0
	return -np.inf

##### Posterior ####

def lnprob(H0, z, zerr, pb_gal, distmu, diststd, distnorm, pixarea, H0_min, H0_max, z_min, z_max, zerr_use=False, cosmo_use=False,omegam=0.3):
	lp = lnprior(H0, H0_min, H0_max)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(H0, z, zerr, pb_gal, distmu, diststd, distnorm, H0_min, H0_max, z_min, z_max, zerr_use, cosmo_use,omegam)

#########################################
########### Time delay utils ############
#########################################

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
