from astropy. io import fits
import healpy as hp
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c #m/s
from scipy.stats import norm
import posterior_utils as pos

def lnlike(H0, z, pb_gal, distmu, diststd, distnorm, pixarea):
	distgal = (c/1000.)*z/H0
	like_gals = pb_gal * distnorm * norm(distmu, diststd).pdf(distgal)*z**2  #*distgal**2  #Maybe this is not clever as it takes a log of exp...
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
INPUT_DIR = "../catalogs/DES_GW170814_galaxies/" #/data/des60.b/data/palmese/GW_reject_cat/DES_GW170814

#For now I am degrading the map to match nside frm the DES catalog

print "Reading skymap..."
map = fits.open('../skymaps/skymap.fits')[1].data
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
theta,phi=hp.pixelfunc.pix2ang(NSIDE,idx_sort_cut,nest=True) #Check if it should have been nest=True
idx_sort_cut_nside32=hp.pixelfunc.ang2pix(32,theta,phi)
idx_sort_cut_nside32=np.unique(idx_sort_cut_nside32)

print "Reading in galaxy catalogs..."
z_gal , zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal = [], [], [], [], [], []

for thisField in idx_sort_cut_nside32:
	if thisField < 10000:
		filename  = INPUT_DIR + "GW_cat_hpx_0" + str(thisField) + ".fits"
	else:
		filename  = INPUT_DIR + "GW_cat_hpx_" + str(thisField) + ".fits"
	#print filename+'\n'
	h = fits.open(filename)[1].data
	mask_good = ((h['DNF_ZMEAN_MOF']>0.04) & (h['DNF_ZMEAN_MOF']<0.17))
	ra_gal=h['RA_1'][mask_good]
	dec_gal=h['DEC_1'][mask_good]
	phi_gal = ra_gal*pi/180.
	theta_gal = (90.-dec_gal)*pi/180.
	pix_gal = hp.pixelfunc.ang2pix(NSIDE, theta_gal, phi_gal,nest=True) #Check if it should have been nest=True
	z_gal.append(h['DNF_ZMEAN_MOF'][mask_good])
	zerr_gal.append(h['DNF_ZSIGMA_MOF'][mask_good])
	pb_gal.append(pb[pix_gal])
	distmu_gal.append(distmu[pix_gal])
	distsigma_gal.append(distsigma[pix_gal])
	distnorm_gal.append(distnorm[pix_gal])

pb_gal = np.concatenate(pb_gal)
z_gal = np.concatenate(z_gal)
zerr_gal = np.concatenate(zerr_gal)
distmu_gal = np.concatenate(distmu_gal)
distsigma_gal = np.concatenate(distsigma_gal)
distnorm_gal = np.concatenate(distnorm_gal)

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
plt.clf()
plt.plot(H0_array, np.exp(lnposterior))
#plt.yscale('log')
plt.xlabel('$H_0$ [km/s/Mpc]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.tight_layout()
#plt.show()
plt.savefig('../plots/H0_posterior_GW170814.png')

idx_max = np.argmax(lnposterior)

perc_max = posterior[:idx_max].sum()/posterior.sum()

maxposterior = posterior[np.argmax(lnposterior)]

print "ML percentile: ", perc_max
print "H0 ML: ", H0_array[idx_max], "+", pos.percentile(perc_max+0.34, posterior, H0_array)-H0_array[idx_max], "-", H0_array[idx_max] - pos.percentile(perc_max-0.34, posterior, H0_array)
print "H0 Median: ", pos.percentile(0.50, posterior, H0_array)




