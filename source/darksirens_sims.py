import matplotlib.pyplot as plt
import numpy as np
import random
import math
import astropy.units as u
from astropy.cosmology import z_at_value , WMAP7
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
import scipy.interpolate as interpolate
from astropy.io import fits
from astropy.table import Table


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

#hdul = fits.open('truth_hpix_Chinchilla-0Y3_v1.6_truth.31.fits')
hdul = fits.open('../catalogs/truth_hpix_Chinchilla-0Y3_v1.6_truth.31.fits')
evt_data = Table(hdul[1].data)
photoz = evt_data['Z']
ra = evt_data['RA']
dec = evt_data['DEC']
galaxy_ID = evt_data['ID']
#print(galaxy_ID)

outfile = open('gauss_dist_609Mpc_sims_o3.txt', 'w')
outfile.write('#ID, galaxyID, RA, DEC, DIST, DIST_ERR\n')

cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)

D = [] 
i = 100
for i in range(0,10): #how many events you want 
    D_evento3=random.gauss(609.9,200) #mean dist from https://arxiv.org/pdf/1709.08079.pdf, O3 avg dist.
    #print(D_evento3)
    z=z_at_value(cosmo.luminosity_distance, D_evento3*u.megaparsec)
    
    
    bcc_nearest = find_nearest(photoz, z) #find the z in the catalog that is nearest to the calculated z from event
    bcc_z = bcc_nearest[1]
    #print(bcc_z)
    RA = np.asarray(ra)[bcc_nearest[0]]
    DEC = np.asarray(dec)[bcc_nearest[0]]
    galaxyID = np.asarray(galaxy_ID)[bcc_nearest[0]]
    #print(z, bcc_z, RA, DEC, galaxyID)
    
    D.append(D_evento3)
    DIST_ERR = 0.25*D_evento3
        
    #Columns: ID, galaxyID, RA, DEC, DIST, DIST_ERR           
    outfile.write('%i %i %f %f %f %f\n' % (i, galaxyID, RA, DEC, D_evento3, DIST_ERR))

outfile.close()    
#print(D)
