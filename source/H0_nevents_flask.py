from astropy. io import fits
import healpy as hp
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c #m/s
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM
import argparse
import os
import posterior_utils as pos


parser = argparse.ArgumentParser(description='Compute H0 posterior given N events')

parser.add_argument('--infile', default='flask_sims_Jul18_hpix.fits',
                    help='Input galaxy catalog name with hpix numbers')
parser.add_argument('--skymap', default='rotated_skymap',
                    help='Input skymap')
parser.add_argument('--nevents', default=1, type=int,
                    help='Number of events to be simulated (for now it is the same skymap being rotated)')
parser.add_argument('--zmin', default=0.05,
                    help='Minimum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--zmax', default=0.22,
                    help='Maximum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--Hmin', default=10.,
                    help='Minimum H0 for a flat prior.')
parser.add_argument('--Hmax', default=220.,
                    help='Maximum H0 for a flat prior.')


args = parser.parse_args()

infile = args.infile
skymap = args.skymap
nevents = args.nevents
z_min = args.zmin
z_max = args.zmax
H0_min = args.Hmin
H0_max = args.Hmax

pb_frac = 0.7 #Fraction of the skymap prbability to consider, decrease for speed
NSIDE = 1024     #skymap nside, corresponding also to the nside of the hpix column in the galaxy catalog
H0bins = 100

DIR_SOURCE = os.getcwd()
DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]
DIR_CATALOG = DIR_MAIN+'/catalogs/'
DIR_SKYMAP = DIR_MAIN+'/skymaps/'
DIR_PLOTS = DIR_MAIN+'/plots/'


print "Reading in galaxy catalogs..."

h = fits.open(DIR_CATALOG+infile)[1].data
mask_z = ((h['z']>0.05) & (h['z']<0.15))
#mask_hpixs = [h['hpix1024']== i for i in idx_sort_cut]
#mask_hpix = np.zeros(mask_z.shape[0], dtype='bool')
#for i in range(len(mask_hpixs)):
#    mask_hpix = mask_hpix | mask_hpixs[i]
#mask_good = np.logical_and(mask_z, mask_hpix)
ra_g=h['ra'][mask_z]
dec_g=h['dec'][mask_z]

try:
    pix_g = h['hpix1024'][mask_z]
except:
    phi_g = ra_g*pi/180.
    theta_g = (90.-dec_g)*pi/180.
    pix_g = hp.pixelfunc.ang2pix(1024, theta_g, phi_g)
    print "No hpix column in catalog"

z_g = h['z'][mask_z]

H0_array = np.linspace(H0_min,H0_max,num=H0bins)
posterior = np.zeros((H0_array.shape[0],nevents))

for nevent in range(nevents):

    print "Reading skymap for event ", str(nevent+1)
    skymap_name = DIR_SKYMAP+skymap+str(nevent)+".fits"
    map = fits.open(skymap_name)[1].data
    pb = map['PROB']
    distmu = map['DISTMU']
    distsigma = map['DISTSIGMA']
    distnorm = map['DISTNORM']

    #Only choose galaxies within pb_frac probability region

    idx_sort = np.argsort(pb)
    idx_sort_up = list(reversed(idx_sort))
    sum = 0.
    id = 0
    print "Sorting pixels by PROB value..."
    while sum<pb_frac:
        this_idx = idx_sort_up[id]
        sum = sum+pb[this_idx]
        id = id+1

    idx_sort_cut = idx_sort_up[:id]
    #theta,phi=hp.pixelfunc.pix2ang(NSIDE,idx_sort_cut,nest=True)

    z_gal , zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, ra_gal, dec_gal = [], [], [], [], [], [], [], []

    print "Assigning probabilities to galaxies.."

    for idx_hpix in idx_sort_cut:
        idx_this_hpix = (pix_g == idx_hpix)
        z_gal.append(z_g[idx_this_hpix])
        ra_gal.append(ra_g[idx_this_hpix])
        dec_gal.append(dec_g[idx_this_hpix])
        #zerr_gal.append()
        #print idx_hpix, pb[idx_hpix], distmu[idx_hpix], distsigma[idx_hpix], distnorm[idx_hpix]
        ngals_this_hpix = z_g[idx_this_hpix].shape[0]
        pb_gal.append(np.full(ngals_this_hpix,pb[idx_hpix]))
        distmu_gal.append(np.full(ngals_this_hpix,distmu[idx_hpix]))
        distsigma_gal.append(np.full(ngals_this_hpix,distsigma[idx_hpix]))
        distnorm_gal.append(np.full(ngals_this_hpix,distnorm[idx_hpix]))

    pb_gal = np.concatenate(pb_gal)
    z_gal = np.concatenate(z_gal)
    distmu_gal = np.concatenate(distmu_gal)
    distsigma_gal = np.concatenate(distsigma_gal)
    distnorm_gal = np.concatenate(distnorm_gal)
    ra_gal = np.concatenate(ra_gal)
    dec_gal = np.concatenate(dec_gal)
    #zerr_gal = np.concatenate(zerr_gal)

    #Posterior without normalization at the moment, and with a delta function for z

    lnposterior=[]
    pixarea = hp.nside2pixarea(NSIDE)

    print "Estimating Posterior for H0 values:"
    for i in range(H0bins):
        lnposterior_bin = pos.lnprob(H0_array[i], z_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, pixarea, H0_min, H0_max)
        #print i, H0_array[i],lnposterior_bin
        lnposterior.append(lnposterior_bin)

    posterior[:,nevent]=np.exp(lnposterior)

    idx_max = np.argmax(lnposterior)

    perc_max = posterior[:idx_max].sum()/posterior.sum()

    maxposterior = posterior[np.argmax(lnposterior)]

    print "ML percentile: ", perc_max
    print "H0 ML: ", H0_array[idx_max], "+", H0_array[idx_max]-pos.percentile(perc_max-0.34, posterior[:,nevent], H0_array), "-", H0_array[idx_max] - pos.percentile(perc_max+0.34, posterior[:,nevent], H0_array)
    print "H0 Median: ", pos.percentile(0.50, posterior[:,nevent], H0_array)


plt.clf()
for nevent in range(nevents):
    norm = np.trapz(posterior[:,nevent], H0_array)
    plt.plot(H0_array, posterior[:,nevent]/norm, label="Event "+str(nevent))

if nevents>1:
    posterior_final = np.prod(posterior, axis=1)
    norm = np.trapz(posterior_final, H0_array)
    plt.plot(H0_array, np.prod(posterior, axis=1)/norm, label="Final")
    idx_max = np.argmax(posterior_final)
    perc_max = posterior_final[:idx_max].sum()/posterior_final.sum()
    maxposterior = posterior_final[idx_max]
    print "-------- Final H0 estimate ---------"
    print "ML percentile: ", perc_max
    print "H0 ML: ", H0_array[idx_max], "+", H0_array[idx_max]-pos.percentile(perc_max-0.34, posterior_final, H0_array), "-", H0_array[idx_max] - pos.percentile(perc_max+0.34, posterior_final, H0_array)
    print "H0 Median: ", pos.percentile(0.50, posterior_final, H0_array)

plt.legend()
plt.xlabel('$H_0$ [km/s/Mpc]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.tight_layout()
#plt.show()
plt.savefig(DIR_PLOTS+'H0_flask_posterior_'+str(nevents)+'.png')






