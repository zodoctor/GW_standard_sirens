from astropy. io import fits
import healpy as hp
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c #m/s
from scipy.stats import norm
import posterior_utils as pos
import argparse
import os


parser = argparse.ArgumentParser(description='Compute H0 posterior given a GW event and a galaxy catalog given in fits files for each healpy pixel')

parser.add_argument('--incat', default='DES_GW170814_galaxies/',
                    help='Input galaxy catalog directory path')
parser.add_argument('--skymap', default='skymap.fits',
                    help='Input skymap')
parser.add_argument('--zmin', default=0.07,
                    help='Minimum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--zmax', default=0.14,
                    help='Maximum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--Hmin', default=10.,
                    help='Minimum H0 for a flat prior.')
parser.add_argument('--Hmax', default=220.,
                    help='Maximum H0 for a flat prior.')
parser.add_argument('--cosmo_use', default=False, type=bool,
                    help='Use full cosmology for dL.')
parser.add_argument('--zerr_use', default=False,  type=bool,
                    help='Galaxy redshift is a Gaussian instead of delta function.')


args = parser.parse_args()

incat = args.incat
skymap = args.skymap
z_min = args.zmin
z_max = args.zmax
H0_min = args.Hmin
H0_max = args.Hmax
cosmo_use = args.cosmo_use
zerr_use = args.zerr_use


pb_frac = 0.9 #Fraction of the skymap prbability to consider, decrease for speed
NSIDE = 1024     #skymap nside
NSIDE_gal = 32     #Galaxy catalog nside
H0bins = 100
out_plot = 'H0_posterior_GW170814.png'
z_column_name = 'DNF_ZMEAN_MOF'
zerr_column_name = 'DNF_ZSIGMA_MOF'


DIR_SOURCE = os.getcwd()
DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]
DIR_CATALOG = incat #DIR_MAIN+'/catalogs/'+incat
DIR_SKYMAP = DIR_MAIN+'/skymaps/'
DIR_PLOTS = DIR_MAIN+'/plots/'


#For now I am degrading the map to match nside frm the DES catalog

print "Reading skymap..."
map = fits.open(DIR_SKYMAP+skymap)[1].data
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
idx_sort_cut_nside32=hp.pixelfunc.ang2pix(NSIDE_gal,theta,phi)
idx_sort_cut_nside32=np.unique(idx_sort_cut_nside32)

print "Reading in galaxy catalogs..."
z_gal , zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal = [], [], [], [], [], []

for thisField in idx_sort_cut_nside32:
	if thisField < 10000:
		filename  = DIR_CATALOG + "GW_cat_hpx_0" + str(thisField) + ".fits"
	else:
		filename  = DIR_CATALOG + "GW_cat_hpx_" + str(thisField) + ".fits"
	#print filename+'\n'
	h = fits.open(filename)[1].data
	mask_good = ((h[z_column_name]>z_min) & (h[z_column_name]<z_max))
	ra_gal=h['RA_1'][mask_good]
	dec_gal=h['DEC_1'][mask_good]
	phi_gal = ra_gal*pi/180.
	theta_gal = (90.-dec_gal)*pi/180.
	pix_gal = hp.pixelfunc.ang2pix(NSIDE, theta_gal, phi_gal,nest=True) #Check if it should have been nest=True
	z_gal.append(h[z_column_name][mask_good])
	zerr_gal.append(h[zerr_column_name][mask_good])
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
	lnposterior.append(pos.lnprob(H0_array[i], z_gal, zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, pixarea, H0_min, H0_max, z_min, z_max, zerr_use=zerr_use, cosmo_use=cosmo_use))

posterior = np.exp(lnposterior)
plt.ion()
plt.clf()
plt.plot(H0_array, np.exp(lnposterior))
#plt.yscale('log')
plt.xlabel('$H_0$ [km/s/Mpc]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.tight_layout()
plt.savefig(DIR_PLOTS+out_plot)

idx_max = np.argmax(posterior)

perc_max = posterior[:idx_max].sum()/posterior.sum()

maxposterior = posterior[idx_max]

print "ML percentile: ", perc_max
print "H0 max posterior: ", H0_array[idx_max], "+", pos.percentile(perc_max+0.34, posterior, H0_array)-H0_array[idx_max], "-", H0_array[idx_max] - pos.percentile(perc_max-0.34, posterior, H0_array)
print "H0 Median: ", pos.percentile(0.50, posterior, H0_array)




