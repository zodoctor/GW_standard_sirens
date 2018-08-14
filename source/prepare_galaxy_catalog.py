#Look into galaxy catalog and write it with hpix number

from astropy. io import fits
import healpy as hp
from math import pi
import numpy as np
from scipy.constants import c #m/s
from scipy.stats import norm
import argparse
import os


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--infile', default='flask_sims_Jul18.fits',
                   help='Input galaxy catalog name')
parser.add_argument('--outfile', default='flask_sims_Jul18_hpix.fits',
                   help='Output galaxy catalog name')
parser.add_argument('--nside', default=1024,
                   help='Skymap Nside')

args = parser.parse_args()

infile = args.infile
outfile = args.outfile
nside = args.nside

DIR_SOURCE = os.getcwd()
DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]

h=fits.open(DIR_MAIN+'/catalogs/'+infile)[1].data
ra_gal=h['ra']
dec_gal=h['dec']
phi_gal = ra_gal*pi/180.
theta_gal = (90.-dec_gal)*pi/180.
pix_gal = hp.pixelfunc.ang2pix(nside, theta_gal, phi_gal)
pix_gal_32 = hp.pixelfunc.ang2pix(32, theta_gal, phi_gal)
z_gal= h['z']

c1 = fits.Column(name='ra', array=ra_gal, format='F')
c2 = fits.Column(name='dec', array=dec_gal, format='F')
c3 = fits.Column(name='z', array=z_gal, format='F')
c4 = fits.Column(name='hpix1024', array=pix_gal, format='K')
c5 = fits.Column(name='hpix32', array=pix_gal_32, format='K')
t = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5])
t.writeto(DIR_MAIN+'/catalogs/'+outfile, overwrite=True)
