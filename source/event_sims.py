import numpy as np
import random
import astropy.units as u
from astropy.cosmology import z_at_value 
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM, funcs
import scipy.interpolate as interpolate
from astropy.io import fits
from astropy.table import Table
import os
import argparse
import mockmaps
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--outname', type=str, default='mockmapsO2', help='Name of output file, no file extension')
parser.add_argument('--ligorun', choices=['o3', 'aligo', 'none'], default='none', help='What distribution to pick runs from')
parser.add_argument('--catalog', default='truth_hpix_Chinchilla-0Y3_v1.6_truth.31.fits', help='What catalog to match RA/DEC/z to')
parser.add_argument('--rootdir', type=str, default='/data/des41.a/data/marcelle/GW170814/dark-siren', help='Path to top level work dir')
parser.add_argument('--nevents', type=int, default=7, help='Number of events to generate')
args=parser.parse_args()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def angular_apperture(dec,sky_area):
    if dec == 0.: 
        return 2 * np.pi * (u.radian).to(u.degree)
    factor = 0.0935 # magic factor to go from 90% target area to 1-sigma width of 2D Gaussian
    s = sky_area * (u.degree**2).to(u.steradian) * factor
    d = dec * (u.degree).to(u.radian)  
    if isinstance(d, np.ndarray):
        x = np.divide(np.full_like(d,s), np.sin(d), out=np.full_like(d,2*np.pi), where=d!=0)
    else:
        x = s/np.sin(d)
    return np.sqrt(abs(x)) * (u.radian).to(u.degree) 

if args.ligorun == 'o3':
    ligo_z_distrib = np.loadtxt(args.rootdir+'/catalogs/stat_redshift_o3_120_60_hlv_bbh_10_10.txt', unpack=True)

if args.ligorun == 'aligo':
    ligo_z_distrib = np.loadtxt(args.rootdir+'/catalogs/stat_redshift_aLIGO_hlv_bbh_10_10.txt', unpack=True)

if args.ligorun != 'none': 
    randomized_z_distrib = np.random.shuffle(ligo_z_distrib)

hdul = fits.open(args.rootdir+'/catalogs/'+args.catalog)
evt_data = hdul[1].data

# RHALO is the comoving distance to nearest halo, in Mpc/h
# It is set to 1000 for galaxies that are farther than 3Mpc/h from any halo
# Use this mask to put events preferrably in overdense regions
# mask = (evt_data['RHALO']<1000)
# CENTRAL == 0 if the galaxy is the BCG of that halo, otherwise CENTRAL == 1
# The mask CENTRAL>=0 means that we are not requiring that the host be a BCG
mask = (evt_data['CENTRAL']>=0) 

gal_z = evt_data['Z'][mask]
gal_ra = evt_data['RA'][mask]
gal_dec = evt_data['DEC'][mask]
gal_id = evt_data['ID'][mask]

H_0=70.
Omega_m=0.3
cosmo = FlatLambdaCDM(H0=H_0, Om0=Omega_m)

sky_area = 60. # area enclosing 90% localization prob, in sq-deg
dist_err_frac = 0.15

outfile = open(args.rootdir+'/catalogs/'+args.outname+'.txt','w')
outfile.write('# cosmo = FlatLambdaCDM\n')
outfile.write('# H_0  = '+str(H_0)+'\n')
outfile.write('# Om0  = '+str(Omega_m)+'\n')
outfile.write('# EVENT_ID HOST_ID RA DEC Z DIST DIST_ERR ROI_SIZE\n')

nevents=args.nevents
dist = np.linspace(50., 950., num=nevents) 
dist_err = dist * dist_err_frac
ra=np.zeros(nevents)
dec=np.zeros(nevents)
host_id=np.zeros(nevents)
angular_size=np.zeros(nevents)

for i in range(nevents):
    if args.ligorun == 'none':
        event_z = z_at_value(cosmo.luminosity_distance, dist[i] * u.Mpc)
    else:
        event_z = randomized_z_distrib[i]
        dist[i] = cosmo.luminozity_distance(event_z).value
        dist_err[i] = dist[i]*dist_err_frac

    nearest_gal = find_nearest(gal_z, event_z)    
    ra[i] = gal_ra[nearest_gal]
    dec[i] = gal_dec[nearest_gal]
    host_id[i] = gal_id[nearest_gal]
    angular_size[i] = angular_apperture(dec[i],sky_area)

    outfile.write('%i %i %f %f %f %f %f %f\n' % (i, host_id[i], ra[i], dec[i], event_z, dist[i], dist_err[i], angular_size[i]))

outfile.close()

mockmaps.gaussian2d_from_sample_map(ra_coord=ra,
                                    dec_coord=dec,
                                    sigma_ra=angular_size,
                                    sigma_dec=angular_size,
                                    distance=dist,
                                    distance_err=dist_err,
                                    nside=128,
                                    sample_map_file_name=args.rootdir+'/skymaps/skymap.fits',
                                    mock_map_file_name=args.rootdir+'/skymaps/'+args.outname+'.fits')
