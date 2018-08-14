from astropy. io import fits
import healpy as hp
from math import pi
c=3.e5 #kms, put more precision
import matplotlib.pyplot as plt
import numpy as np

def lnlike(H0, z, distmu, diststd, distnorm):
	distgal = c*z/H0
	lnlike_gals = distnorm/(diststd*(2.*pi)**0.5)*np.exp(-0.5*((distgal-distmu)**2)/diststd**2) #Maybe this is not clever as it takes a log of exp...
	return np.log(((c/H0)**3)*np.sum(lnlike_gals))

def lnprior(H0):
	if 60. < H0 < 80. : 
		return 0.0
	return -np.inf

def lnprob(H0, z, distmu, diststd, distnorm):
	lp = lnprior(H0)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(H0, z, distmu, diststd, distnorm)

 
pb_frac = 0.9
NSIDE=1024
H0bins = 30 

print "Reading skymap..."
map = fits.open('skymap.fits')[1].data
pb = map['PROB']
distmu = map['DISTMU']
distsigma = map['DISTSIGMA']
distnorm = map['DISTNORM']

#Only choose galaxies within pb_frac probability region

idx_sort = np.argsort(pb)
idx_sort_up = list(reversed(idx_sort))
sum = 0.
id = 0
print "Sorting pixels..."
while sum<pb_frac:
    this_idx = idx_sort_up[id]
    sum = sum+pb[this_idx]
    id = id+1

idx_sort_cut = idx_sort_up[:id]

print "Reading in galaxy catalog..."

gals=fits.open('Y3A2_GW170814area_Jul18_good.fits')[1].data
ra_gal = gals['RA']
dec_gal = gals['DEC']
phi_gal = ra_gal*pi/180.
theta_gal = (90.-dec_gal)*pi/180.
pix_gal = hp.pixelfunc.ang2pix(NSIDE, phi_gal, theta_gal,nest=True)
pb_gal = pb[pix_gal]
distmu_gal = distmu[pix_gal]
distsigma_gal = distsigma[pix_gal]
distnorm_gal = distnorm[pix_gal]

print "Defining good galaxies"
idx_goodgals = []
for idx_gal in range(pix_gal.shape[0]): 
	if (np.sum(np.where(idx_sort_cut == pix_gal[idx_gal]))>0.): idx_goodgals.append(idx_gal)

print idx_goodgals
pb_gal = pb_gal[idx_goodgals]
distmu_gal = distmu_gal[idx_goodgals]
distsigma_gal = distsigma_gal[idx_goodgals]
distnorm_gal = distnorm_gal[idx_goodgals]

#Posterior without normalization at the moment, and with a delta function for z

Ngoodgals=pb_gal.shape[0]

H0_array = np.linspace(40.,140.,num=H0bins)
lnposterior=[]

print "Estimating Posterior for H0 values:"
for i in range(H0bins):
	print H0[i]
	lnposterior.append(lnprob(H0[i], z, distmu, diststd, distnorm))

plt.ion()
plt.plot(H0_array, lnposterior)
plt.xlabel('$H_0$')
plt.ylabel('$p$')
plt.savefig('H0_posterior.png')







