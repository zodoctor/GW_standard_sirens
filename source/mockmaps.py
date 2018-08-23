import matplotlib
matplotlib.use('Agg')
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import erf
from astropy.table import Table

# example of how to use mockmaps.py in your code:
#import mockmaps
#mockmaps.gaussian2d_from_sample_map(ra_coord=0.,dec_coord=0.,sigma_ra=5.,sigma_dec=5.,distance=100.,distance_err=25.,nside=128,sample_map_file_name="skymap.fits",mock_map_file_name="mockmap.fits")



# Helper function to get the ra dec coords of a healpix pixel
def pix2radec(ipix,nside=512):
    theta, phi = hp.pix2ang(nside,ipix,nest=False)
    dec = np.rad2deg(0.5 * np.pi - theta) 
    ra = np.rad2deg(phi)
    return ra,dec

# Helper function to compute the 4th layer values, using formula by Juan Garcia-Bellido:
def norm(m,s):
    return (np.exp(-0.5*(m/s)**2)*m*s)/(np.sqrt(2*np.pi)) + 0.5*(m**2+s**2)*(1+erf(m/(np.sqrt(2)*s)))

# Use this function to make your mockmaps.
# This function takes values for one event, or arrays of many events.
# The coordinates are assumed to be in degrees.
# The sample map has 4 layers of data: prob, dist, dist_err, norm
def gaussian2d_from_sample_map(ra_coord=0.,dec_coord=0.,
                               sigma_ra=5.,sigma_dec=5.,
                               distance=100.,distance_err=25.,
                               nside=128,
                               sample_map_file_name="data/skymap.fits",
                               mock_map_file_name="data/mockmap.fits"):

    # read sample file
    m = hp.read_map(sample_map_file_name,hdu=1,verbose=False,field=[0,1,2,3])

    # if ra_coord is an array, assume dec_coord, sigma_ra, sigma_dec, distance, distance_err are also arrays of same length:
    if isinstance(ra_coord, (list, tuple, np.ndarray)):
        basename=mock_map_file_name.split('.')[0]
        extension=".fits"
        for i in range(len(ra_coord)):
            mock_map=one_gaussian2d_from_sample_map(m,ra_coord[i],dec_coord[i],sigma_ra[i],sigma_dec[i],distance[i],distance_err[i],nside)
            ##hp.write_map(mock_map_file_name[i],mock_map,column_names=["PROB","DISTMU","DISTSIGMA","DISTNORM"],nest=True,coord='C')
            t=Table(mock_map,names=("PROB","DISTMU","DISTSIGMA","DISTNORM"))
            mock_map_file_name=basename+'_'+str(i)+extension
            t.write(mock_map_file_name,format='fits')
        
    # if ra_coord is a scalar, assume they are all scalars:
    if isinstance(ra_coord, (float,int)):
        mock_map=one_gaussian2d_from_sample_map(m,ra_coord,dec_coord,sigma_ra,sigma_dec,distance,distance_err,nside)

        t=Table(mock_map,names=("PROB","DISTMU","DISTSIGMA","DISTNORM"))
        t.write(mock_map_file_name,format='fits')

#        hp.write_map(mock_map_file_name,mock_map,column_names=["PROB","DISTMU","DISTSIGMA","DISTNORM"],nest=True,coord='C')

    return 


# this function is to make one map from a healpix sample map with four layers:
def one_gaussian2d_from_sample_map(hpx_layers,ra,dec,sigma_ra,sigma_dec,dist,dist_err,nside_p):

    npix = hp.nside2npix(nside_p)

    hpx = hpx_layers[0]

    s = np.random.normal(ra,sigma_ra,len(hpx))
    counts, bins, patches = plt.hist(s, int(np.sqrt(npix)), density=False)
    ra_prob = counts/np.sum(counts)
    ra_bin_edges = bins[:-1]

    s = np.random.normal(dec,sigma_dec,len(hpx))
    counts, bins, patches = plt.hist(s, int(np.sqrt(npix)), density=False)
    dec_prob = counts/np.sum(counts)
    dec_bin_edges = bins[:-1]

    hpx_p=hpx_layers[0]
    hpx_d=hpx_layers[1]
    hpx_e=hpx_layers[2]
    hpx_n=hpx_layers[3]

    nside = hp.npix2nside(len(hpx))

    if nside > nside_p : 
        hpx_p=hp.ud_grade(hpx_p, nside_p)
        hpx_d=hp.ud_grade(hpx_d, nside_p)
        hpx_e=hp.ud_grade(hpx_e, nside_p)
        hpx_n=hp.ud_grade(hpx_n, nside_p)

    for pix in range(len(hpx_p)):
        ra_pix,dec_pix=pix2radec(pix,nside_p)
        if ra_pix>180. : ra_pix = ra_pix - 360.
        ra_idx=(np.abs(ra_bin_edges-ra_pix)).argmin()
        dec_idx=(np.abs(dec_bin_edges-dec_pix)).argmin()
        ra_p=ra_prob[ra_idx]
        dec_p=dec_prob[dec_idx]
        hpx_p[pix]=hpx_p[pix] * 0.0 + (ra_p * dec_p) 
        hpx_d[pix]=dist
        hpx_e[pix]=dist_err
        hpx_n[pix]=norm(dist,dist_err)

    norm_factor = hpx_p.sum()
    hpx_p = hpx_p/norm_factor

    if nside > nside_p : 
        hpx_p=hp.ud_grade(hpx_p, nside, power=-2)
        hpx_d=hp.ud_grade(hpx_d, nside)
        hpx_e=hp.ud_grade(hpx_e, nside)
        hpx_n=hp.ud_grade(hpx_n, nside)

    return hpx_p,hpx_d,hpx_e,hpx_n


