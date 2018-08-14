from astropy. io import fits
import healpy as hp
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c #m/s
from scipy.stats import norm

def lnlike(H0, z, pb_gal, distmu, diststd, distnorm, pixarea):
	distgal = (c/1000.)*z/H0
	like_gals = pb_gal*distnorm* norm(distmu, diststd).pdf(distgal)*z**2 #/pixarea*distgal**2 #/(diststd*(2.*pi)**0.5)*distgal**2*np.exp(-0.5*((distgal-distmu)**2)/diststd**2) #Maybe this is not clever as it takes a log of exp...
	normalization = H0**3
	return np.log(np.sum(like_gals)/normalization)

def lnprior(H0):
	if 10. < H0 < 220. : 
		return 0.0
	return -np.inf

def lnprob(H0, z, pb_gal, distmu, diststd, distnorm, pixarea):
	lp = lnprior(H0)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(H0, z, pb_gal, distmu, diststd, distnorm, pixarea)

 
pb_frac = 0.9
NSIDE=1024
H0bins = 100 
INPUT_DIR = "./DES_GW170814_galaxies/"

#For now I am degrading the map to match nside frm the DES catalog

print "Reading skymap..."
map = fits.open('skymapGW170817.fits')[1].data
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


print "Only choose one galaxy at 40 Mpc and H0=70."


pb_gal = 0.9 #map[idx_sort_cut[0]]['PROB']
distmu_gal = 40.
z_gal = 70.*distmu_gal/(c/1000.)
distsigma_gal = map[idx_sort_cut[0]]['DISTSIGMA']/10.
distnorm_gal = map[idx_sort_cut[0]]['DISTNORM']

#Posterior without normalization at the moment, and with a delta function for z


H0_array = np.linspace(10.,220.,num=H0bins)
lnposterior=[]
pixarea = hp.nside2pixarea(NSIDE )


print "Estimating Posterior for H0 values:"
for i in range(H0bins):
	#print H0_array[i]
	lnposterior.append(lnprob(H0_array[i], z_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, pixarea))

posterior = np.exp(lnposterior)

plt.ion()
#plt.clf()
plt.plot(H0_array, posterior/norm, label='GW170817 fixed')
plt.xlabel('$H_0$ [km/s/Mpc]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.tight_layout()
plt.xlim(10.,220.)
plt.show()
#plt.savefig('H0_posterior.png')




