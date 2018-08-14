#Test code for GW170817 with and without counterpart
from astropy. io import fits
import healpy as hp
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c #m/s
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
import os


#cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def percentile(perc, pdf, xarray): #May not be perfect due to binning...
	sum = 0.
	idx = 0
	tot=pdf.sum()
	while sum<perc:
		sum = sum + pdf[idx]/tot
		idx=idx+1
	return xarray[idx]

def lnlike(H0, z, pb_gal, distmu, diststd, distnorm, pixarea):
    distgal = (c/1000.)*z/H0
    like_gals = pb_gal * distnorm * norm(distmu, diststd).pdf(distgal)*z**2 # /cosmo.comoving_volume(z).value #Maybe this is not clever as it takes a log of exp...
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
DIR_SOURCE = os.getcwd()
DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]
DIR_CATALOG = DIR_MAIN+'/catalogs/GW170817_galaxies/'
DIR_SKYMAP = DIR_MAIN+'/skymaps/'
DIR_PLOTS = DIR_MAIN+'/plots/'

#For now I am degrading the map to match nside frm the DES catalog

print "Reading skymap..."
map = fits.open(DIR_SKYMAP+'skymapGW170817.fits')[1].data
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
theta,phi=hp.pixelfunc.pix2ang(NSIDE,idx_sort_cut,nest=True) 
idx_sort_cut_nside32=hp.pixelfunc.ang2pix(32,theta,phi)
idx_sort_cut_nside32=np.unique(idx_sort_cut_nside32)

print "Reading in galaxy catalogs..."
z_gal , zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, ra_gal, dec_gal = [], [], [], [], [], [], [], []

for thisField in idx_sort_cut_nside32:
	if thisField < 10000:
		filename  = DIR_CATALOG + "GW_cat_hpx_0" + str(thisField) + ".fits"
	else:
		filename  = DIR_CATALOG + "GW_cat_hpx_" + str(thisField) + ".fits"
	#print filename+'\n'
	h = fits.open(filename)[1].data
	mask_good = ((h['ZPHOTO']>0.00) & (h['ZPHOTO']<0.012) & (h['CATALOG']=="2MASS"))
	ra_g=h['RA'][mask_good]
	dec_g=h['DEC'][mask_good]
	phi_gal = ra_g*pi/180.
	theta_gal = (90.-dec_g)*pi/180.
	pix_gal = hp.pixelfunc.ang2pix(NSIDE, theta_gal, phi_gal,nest=True) 
	ra_gal.append(ra_g)
	dec_gal.append(dec_g)
	z_gal.append(h['ZPHOTO'][mask_good])
	zerr_gal.append(h['ZPHOTO_ERR'][mask_good])
	pb_gal.append(pb[pix_gal])		# THESE should have the same size as z_gal being appended!!!
	distmu_gal.append(distmu[pix_gal])
	distsigma_gal.append(distsigma[pix_gal])
	distnorm_gal.append(distnorm[pix_gal])

pb_gal = np.concatenate(pb_gal)
z_gal = np.concatenate(z_gal)
zerr_gal = np.concatenate(zerr_gal)
distmu_gal = np.concatenate(distmu_gal)
distsigma_gal = np.concatenate(distsigma_gal)
distnorm_gal = np.concatenate(distnorm_gal)
ra_gal = np.concatenate(ra_gal)
dec_gal = np.concatenate(dec_gal)

#Posterior without normalization at the moment, and with a delta function for z


H0_array = np.linspace(10.,220.,num=H0bins)
lnposterior, lnposterior_ngc=[],[]
pixarea = hp.nside2pixarea(NSIDE )


print "Estimating Posterior for H0 values:"
for i in range(H0bins):
	lnposterior_bin = lnprob(H0_array[i], z_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, pixarea)
	#print i, H0_array[i],lnposterior_bin
	lnposterior.append(lnposterior_bin)

plt.ion()
plt.clf()
posterior=np.exp(lnposterior)
norm_all = np.trapz(posterior, H0_array)
plt.plot(H0_array, posterior/norm_all, label='GW170817 2MASS galaxies')

print "Estimating Posterior for H0 for NGC 4993:"

z_ngc = 0.009680
ra_ngc = 197.448750
dec_ngc = -23.383889

pix_ngc = hp.pixelfunc.ang2pix(NSIDE, (90.-dec_ngc)*pi/180., ra_ngc*pi/180.,nest=True)
distmu_ngc = distmu[pix_ngc]
distsigma_ngc = distsigma[pix_ngc]

lnposterior_ngc=[]

for i in range(H0bins):
    lnposterior_bin = lnprob(H0_array[i], z_ngc, 1, distmu_ngc, distsigma_ngc, 1., pixarea) #43.8 Mpc or 41.1?
    #print i, H0_array[i],lnposterior_bin
    lnposterior_ngc.append(lnposterior_bin)
posterior_ngc=np.exp(lnposterior_ngc)
norm_ngc = np.trapz(posterior_ngc, H0_array)
plt.plot(H0_array, posterior_ngc/norm_ngc, label='NGC 4993, skymap pixel info')


#plt.yscale('log')
plt.xlabel('$H_0$ [km/s/Mpc]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig('H0_posterior_GW170817.png')

idx_max = np.argmax(lnposterior)
perc_max = posterior[:idx_max].sum()/posterior.sum()
maxposterior = posterior[np.argmax(lnposterior)]

print "No counterpart:"
print "ML percentile: ", perc_max
print "H0 ML: ", H0_array[np.argmax(lnposterior)], "+", H0_array[np.argmax(lnposterior)]-percentile(perc_max-0.34, posterior, H0_array), "-", H0_array[np.argmax(lnposterior)]- percentile(perc_max+0.34, posterior, H0_array)
print "H0 Median: ", percentile(0.50, posterior, H0_array)

idx_max = np.argmax(lnposterior_ngc)
perc_max = posterior_ngc[:idx_max].sum()/posterior_ngc.sum()
maxposterior = posterior_ngc[np.argmax(lnposterior_ngc)]

print "Assuming NGC 4993"
print "ML percentile: ", perc_max
print "H0 ML: ", H0_array[np.argmax(lnposterior_ngc)], "+", H0_array[np.argmax(lnposterior_ngc)]-percentile(perc_max-0.34, posterior_ngc, H0_array), "-", H0_array[np.argmax(lnposterior_ngc)]- percentile(perc_max+0.34, posterior_ngc, H0_array)
print "H0 Median: ", percentile(0.50, posterior_ngc, H0_array)




