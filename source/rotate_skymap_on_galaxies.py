#Rotates an input skymap by putting the highest PROB value on top of the N galaxies
#which are the closest to the redshist at the peak of the DIST posterior, with H0 fixed
#given in input

from astropy. io import fits
import healpy as hp
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c #m/s
from scipy.stats import norm
import argparse
import os
import hp2np
import rotate
import scipy.interpolate
import rm


parser = argparse.ArgumentParser(description='Rotate input skymap to density peaks')
parser.add_argument('--infile', default='flask_sims_Jul18_hpix.fits',
                   help='Input galaxy catalog name with hpix numbers')
parser.add_argument('--skymap', default='skymap.fits',
                   help='Input skymap to be rotated')
parser.add_argument('--Nevents', default=1,
                   help='Number of events to be simulated (for now it is the same skymap being rotated)')
parser.add_argument('--H0', default=70.,
                   help='Input H0 (km/s/Mpc)')
parser.add_argument('--zmin', default=0.05,
                    help='Minimum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--zmax', default=0.22,
                    help='Maximum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--dist_min', default=300.,
                    help='Minimum luminosity distance to be considered when computing the line of sight posterior from the skymap. Default is for GW170814. [Mpc]')
parser.add_argument('--dist_max', default=600.,
                    help='Minimum luminosity distance to be considered when computing the line of sight posterior from the skymap. Default is for GW170814. [Mpc]')
parser.add_argument('--out_skymap', default='rotated_skymap_on_galaxy_',
                   help='Output skymap(s)')
parser.add_argument('--nevents', default=1,type=int,
                    help='Number of events (i.e. skymaps) to be simulated.')

args = parser.parse_args()

infile = args.infile
skymap_file = args.skymap
Nevents = args.Nevents #To be implemented for more events (both here and in the likelihood)
H0 = args.H0
out_skymap = args.out_skymap
z_min = args.zmin
z_max = args.zmax
dist_min = args.dist_min
dist_max = args.dist_max
nevents = args.nevents

DIR_SOURCE = os.getcwd()
DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]
DIR_CATALOG = DIR_MAIN+'/catalogs/'
DIR_SKYMAP = DIR_MAIN+'/skymaps/'
DIR_PLOTS = DIR_MAIN+'/plots/'


print "Reading skymap..."
map = fits.open(DIR_SKYMAP+skymap_file)[1].data
pb = map['PROB']
distmu = map['DISTMU']
distsigma = map['DISTSIGMA']
distnorm = map['DISTNORM']

nside = hp.get_nside(pb)

idx_sort = np.argsort(pb)
idx_sort_up = list(reversed(idx_sort))

idx_max = idx_sort_up[0]
dist = np.linspace(dist_min, dist_max) #Mpc
posterior_los = pb[idx_max] * distnorm[idx_max] * norm(distmu[idx_max], distsigma[idx_max]).pdf(dist)

dist_ML = dist[np.argmax(posterior_los)]
#print dist_ML

# Read in catalog with hpix numbers
d = fits.open(DIR_CATALOG+infile)[1].data

# DEFININING REDSHIFT OF EVENT!! Pick the nevents galaxies which are closest in redshift to this
#redshift, given H0

z_event = H0*dist_ML/(c/1000.)
delta_z = np.abs(d['z']-z_event)
idxs_host = np.argsort(delta_z)
idxs_best_host = idxs_host[:nevents]

nevent=0

#Read in skymap to be rotated
debug = False
if debug :
    res = 64
else:
    res = False

rax,decx, map_ringx_pb = hp2np.hp2np(DIR_SKYMAP+skymap_file,degrade=res, field=0)
rax,decx, map_ringx_d = hp2np.hp2np(DIR_SKYMAP+skymap_file,degrade=res, fluxConservation=False, field=1)
rax,decx, map_ringx_dstd = hp2np.hp2np(DIR_SKYMAP+skymap_file,degrade=res, fluxConservation=False, field=2)
rax,decx, map_ringx_dn = hp2np.hp2np(DIR_SKYMAP+skymap_file,degrade=res, field=3)
map_ringx_d[(np.isinf(map_ringx_d))|(map_ringx_d<0.)|(np.isnan(map_ringx_d))] = 0.

#Loop over Nevents

for idx_best_host in idxs_best_host:
    
    print "The event number ",nevent," is at z=",d['z'][idx_best_host], " and distance=", dist_ML, " Mpc"

    ra_new = d['ra'][idx_best_host]
    dec_new = d['dec'][idx_best_host]

    ligo_max = np.argmax(map_ringx_pb)
    ligo_max_ra = rax[ligo_max]
    ligo_max_dec = decx[ligo_max]

    # Rotation happens here
    ra,dec, rmap_pb, rmap_d, rmap_dstd, rmap_dn  =\
        rm.rot_all_maps(rax,decx, map_ringx_pb, map_ringx_d, map_ringx_dstd, map_ringx_dn,ligo_max_ra, ligo_max_dec, ra_new, dec_new)
        
    if debug :
        up_rmap_pb = hp.ud_grade(rmap_pb, nside)
        up_rmap_d = hp.ud_grade(rmap_d, nside, power=-2)
        up_rmap_dstd = hp.ud_grade(rmap_dstd, nside, power=-2)
        up_rmap_dn = hp.ud_grade(rmap_dn, nside)
    else:
        up_rmap_pb = rmap_pb
        up_rmap_d = rmap_d
        up_rmap_dstd = rmap_dstd
        up_rmap_dn = rmap_dn

    #Is it right to put the nans to 0?? maybe not

    up_rmap_pb[np.isnan(up_rmap_pb)] = 0.
    up_rmap_d[np.isnan(up_rmap_d)] = 0.
    up_rmap_dn[np.isnan(up_rmap_dn)] = 0.
    up_rmap_dstd[np.isnan(up_rmap_dstd)] = 0.

    c1 = fits.Column(name='PROB', array=up_rmap_pb, format='E')
    c2 = fits.Column(name='DISTMU', array=up_rmap_d, format='E')
    c3 = fits.Column(name='DISTSIGMA', array=up_rmap_dstd, format='E')
    c4 = fits.Column(name='DISTNORM', array=up_rmap_dn, format='E')
    t = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
    t.writeto(DIR_SKYMAP+out_skymap+str(nevent)+".fits", overwrite=True)

    nevent = nevent+1
    #plt.clf()
    #hp.mollview(up_rmap_pb, cmap='Reds',nest=False)
    #hp.graticule()
    #plt.savefig(DIR_PLOTS+"rot_skymap.png")

