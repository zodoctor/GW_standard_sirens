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

parser.add_argument('--infile', default='GW170817_lephare_radec.fits',
                    help='Input galaxy catalog name with hpix numbers')
parser.add_argument('--skymap', default='skymap',
                    help='Input skymap')
parser.add_argument('--zmin', default=0.05,
                    help='Minimum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--zmax', default=0.22,
                    help='Maximum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--taumin', default=0.01,
                    help='Minimum delay time in Gyr')
parser.add_argument('--taumax', default=13.5,
                    help='Maximum delay time in Gyr')
parser.add_argument('--nevents', default=1, type=int,
                    help='Number of events to be simulated (for now it is the same skymap being rotated)')
parser.add_argument('--H0', default=70.,
                    help='H0 to be used in estimating distances')

args = parser.parse_args()

infile = args.infile
skymap = args.skymap
z_min = args.zmin
z_max = args.zmax
taud_min = args.taumin
taud_max = args.taumax
nevents = args.nevents # To be implemented
H0 = args.H0

pb_frac = 0.9 #Fraction of the skymap prbability to consider, decrease for speed
NSIDE = 1024     #skymap nside, corresponding also to the nside of the hpix column in the galaxy catalog
taudbins = 100

DIR_SOURCE = os.getcwd()
DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]
DIR_CATALOG = DIR_MAIN+'/catalogs/'
DIR_SKYMAP = DIR_MAIN+'/skymaps/'
DIR_PLOTS = DIR_MAIN+'/plots/'


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
theta,phi=hp.pixelfunc.pix2ang(NSIDE,idx_sort_cut,nest=True)

print "Reading in galaxy catalogs..."

h = fits.open(DIR_CATALOG+infile)[1].data
mask = ((h['Z']>0.05) & (h['Z']<0.15) & (h['AGE_BEST']>0.) & (h['SFR_BEST']>-90.) )
#mask_hpixs = [h['hpix1024']== i for i in idx_sort_cut]
#mask_hpix = np.zeros(mask_z.shape[0], dtype='bool')
#for i in range(len(mask_hpixs)):
#    mask_hpix = mask_hpix | mask_hpixs[i]
#mask_good = np.logical_and(mask_z, mask_hpix)
ra_g=h['RA'][mask]
dec_g=h['DEC'][mask]

try:
    pix_g = h['hpix1024'][mask]
except:
    phi_g = ra_g*pi/180.
    theta_g = (90.-dec_g)*pi/180.
    pix_g = hp.pixelfunc.ang2pix(1024, theta_g, phi_g, nest=True)
    print "No hpix column in catalog, computing healpy pixel numbers..."

z_g = h['Z'][mask]
sfr_g = h['SFR_BEST'][mask]
mass_g = h['MASS_BEST'][mask]
MODs = h['MOD_BEST'][mask]
age_g = h['AGE_BEST'][mask]/1e9

zmet_g, tau_g = [],[]

for MOD in MODs:
    if MOD==1: zmet_g.append(0.004); tau_g.append(0.1);
    elif MOD==2: zmet_g.append(0.004); tau_g.append(0.3);
    elif MOD==3: zmet_g.append(0.004); tau_g.append(1.);
    elif MOD==4: zmet_g.append(0.004); tau_g.append(2.);
    elif MOD==5: zmet_g.append(0.004); tau_g.append(3.);
    elif MOD==6: zmet_g.append(0.004); tau_g.append(5.);
    elif MOD==7: zmet_g.append(0.004); tau_g.append(10.);
    elif MOD==8: zmet_g.append(0.004); tau_g.append(15.);
    elif MOD==9: zmet_g.append(0.004); tau_g.append(30.);
    elif MOD==10: zmet_g.append(0.008); tau_g.append(0.1);
    elif MOD==11: zmet_g.append(0.008); tau_g.append(0.3);
    elif MOD==12: zmet_g.append(0.008); tau_g.append(1.);
    elif MOD==13: zmet_g.append(0.008); tau_g.append(2.);
    elif MOD==14: zmet_g.append(0.008); tau_g.append(3.);
    elif MOD==15: zmet_g.append(0.008); tau_g.append(5.);
    elif MOD==16: zmet_g.append(0.008); tau_g.append(10.);
    elif MOD==17: zmet_g.append(0.008); tau_g.append(15.);
    elif MOD==18: zmet_g.append(0.008); tau_g.append(30.);
    elif MOD==19: zmet_g.append(0.02); tau_g.append(0.1);
    elif MOD==20: zmet_g.append(0.02); tau_g.append(0.3);
    elif MOD==21: zmet_g.append(0.02); tau_g.append(1.);
    elif MOD==22: zmet_g.append(0.02); tau_g.append(2.);
    elif MOD==23: zmet_g.append(0.02); tau_g.append(3.);
    elif MOD==24: zmet_g.append(0.02); tau_g.append(5.);
    elif MOD==25: zmet_g.append(0.02); tau_g.append(10.);
    elif MOD==26: zmet_g.append(0.02); tau_g.append(15.);
    elif MOD==27: zmet_g.append(0.02); tau_g.append(30.);
    else: print "Model does not exist"

tau_g = np.array(tau_g)
zmet_g = np.array(zmet_g)

taud_array = np.linspace(taud_min,taud_max,num=taudbins)
posterior = np.zeros((taud_array.shape[0],nevents))

z_gal , zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, ra_gal, dec_gal, tau_gal, age_gal, norm_sfh_gal  = [], [], [], [], [], [], [], [], [], [], []

for idx_hpix in idx_sort_cut:
    idx_this_hpix = (pix_g == idx_hpix)
    z_gal.append(z_g[idx_this_hpix])
    ra_gal.append(ra_g[idx_this_hpix])
    dec_gal.append(dec_g[idx_this_hpix])
    tau_gal.append(tau_g[idx_this_hpix])
    age_gal.append(age_g[idx_this_hpix])
    norm_sfh_gal.append(10**sfr_g[idx_this_hpix]/pos.sfh(age_g[idx_this_hpix], tau_g[idx_this_hpix]))
    #zerr_gal.append()
    #print idx_hpix,10**sfr_g[idx_this_hpix]/pos.sfh(age_g[idx_this_hpix], tau_g[idx_this_hpix])
    ngals_this_hpix = z_g[idx_this_hpix].shape[0]
    pb_gal.append(np.full(ngals_this_hpix,pb[idx_hpix]))
    distmu_gal.append(np.full(ngals_this_hpix,distmu[idx_hpix]))
    distsigma_gal.append(np.full(ngals_this_hpix,distsigma[idx_hpix]))
    distnorm_gal.append(np.full(ngals_this_hpix,distnorm[idx_hpix]))



pb_gal = np.concatenate(pb_gal)
z_gal = np.concatenate(z_gal)
#zerr_gal = np.concatenate(zerr_gal)
distmu_gal = np.concatenate(distmu_gal)
distsigma_gal = np.concatenate(distsigma_gal)
distnorm_gal = np.concatenate(distnorm_gal)
tau_gal = np.concatenate(tau_gal)
age_gal = np.concatenate(age_gal)
norm_sfh_gal = np.concatenate(norm_sfh_gal)

print "There are ",pb_gal.shape[0], " galaxies"

#Posterior without normalization at the moment, and with a delta function for z


taud_array = np.linspace(taud_min,taud_max,num=taudbins)
lnposterior=[]
pixarea = hp.nside2pixarea(NSIDE )


print "Estimating Posterior for tau_d values:"
for i in range(taudbins):
    lnposterior_bin =pos.lnprob_taud(taud_array[i], z_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, H0, age_gal, tau_gal, norm_sfh_gal, taud_min, taud_max)
    print lnposterior_bin
    lnposterior.append(lnposterior_bin)

posterior = np.exp(lnposterior)
plt.ion()
plt.clf()
plt.plot(taud_array, np.exp(lnposterior))
plt.yscale('log')
plt.xlabel(r'$\tau_d$ [Gyr]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.tight_layout()
plt.show()
plt.savefig('../plots/taud_posterior_GW170814.png')

idx_max = np.argmax(lnposterior)

perc_max = posterior[:idx_max].sum()/posterior.sum()

maxposterior = posterior[np.argmax(lnposterior)]

print "ML percentile: ", perc_max
print "H0 ML: ", taud_array[idx_max], "+", pos.percentile(perc_max+0.34, posterior, taud_array)-taud_array[idx_max], "-", taud_array[idx_max] - pos.percentile(perc_max-0.34, posterior, taud_array)
print "H0 Median: ", pos.percentile(0.50, posterior, taud_array)




