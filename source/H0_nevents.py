import matplotlib
matplotlib.use('Agg')
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
parser.add_argument('--skymap', default='rotated_skymap_on_galaxy_0.fits',
                    help='Input skymap')
parser.add_argument('--nevents', default=1, type=int,
                    help='Number of events to be simulated (for now it is the same skymap being rotated)')
parser.add_argument('--zmin', default=0.05, type=float,
                    help='Minimum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--zmax', default=0.22, type=float,
                    help='Maximum redshift to be considered in the galaxy catalog. Default is for GW170814.')
parser.add_argument('--Hmin', default=10., type=float,
                    help='Minimum H0 for a flat prior.')
parser.add_argument('--Hmax', default=220., type=float,
                    help='Maximum H0 for a flat prior.')
parser.add_argument('--cosmo_use', default=False, type=bool,
                    help='Use full cosmology for dL.')
parser.add_argument('--zerr_use', default=False,  type=bool,
                    help='Galaxy redshift is a Gaussian instead of delta function.')
parser.add_argument('--blind', default=False,  type=bool,
                    help='Blinding of the H0 results.')


args = parser.parse_args()

infile = args.infile
skymap = args.skymap
nevents = args.nevents
z_min = args.zmin
z_max = args.zmax
H0_min = args.Hmin
H0_max = args.Hmax
cosmo_use = args.cosmo_use
zerr_use = args.zerr_use
blind = args.blind

pb_frac = 0.7 #Fraction of the skymap probability to consider, decrease for speed
H0bins = 100

# Add by hand an error test_photozs_err to the galaxy redshifts. Note this is fixed for all galaxies for now!

test_photozs = True
test_photozs_err = 0.01

# Names of the input galaxy catalog columns

ra_column_name = 'ra'
dec_column_name = 'dec'
z_column_name = 'z'
zerr_column_name = 'zerr'
hpix_column_name = 'hpix1024' #If it exists in the galaxy catalog, if not, it is computed for the skymap given its NSIDE, default is ring
nest = False #Watch out! mock maps have nest=True but BCC has False at the moment

DIR_SOURCE = os.getcwd()
DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]
DIR_CATALOG = DIR_MAIN+'/catalogs/'
DIR_SKYMAP = DIR_MAIN+'/skymaps/'
DIR_PLOTS = DIR_MAIN+'/plots/'
DIR_OUT = DIR_MAIN+'/out/'


print "Reading in galaxy catalogs..."

h = fits.open(DIR_CATALOG+infile)[1].data
mask_z = ((h[z_column_name]>z_min) & (h[z_column_name]<z_max))
#mask_hpixs = [h['hpix1024']== i for i in idx_sort_cut]
#mask_hpix = np.zeros(mask_z.shape[0], dtype='bool')
#for i in range(len(mask_hpixs)):
#    mask_hpix = mask_hpix | mask_hpixs[i]
#mask_good = np.logical_and(mask_z, mask_hpix)
ra_g=h[ra_column_name][mask_z]
dec_g=h[dec_column_name][mask_z]

#Read in the first skymap to get nside
if (nevents==1):
    skymap_name = DIR_SKYMAP+skymap
else:
    skymap_name = DIR_SKYMAP+skymap+str(0)+".fits"
map = fits.open(skymap_name)[1].data
pb = map['PROB'].flatten()
NSIDE = hp.pixelfunc.get_nside(pb)

try:
    pix_g = h[hpix_column_name][mask_z]
except:
    phi_g = ra_g*pi/180.
    theta_g = (90.-dec_g)*pi/180.
    pix_g = hp.pixelfunc.ang2pix(NSIDE, theta_g, phi_g, nest=nest)
    print "No hpix column in catalog"

z_g = h[z_column_name][mask_z]

H0_array = np.linspace(H0_min,H0_max,num=H0bins)
posterior = np.zeros((H0_array.shape[0],nevents))

for nevent in range(nevents):

    print "Reading skymap for event ", str(nevent+1)
    if (nevents==1):
        skymap_name = DIR_SKYMAP+skymap
    else:
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
    zerr_gal = np.zeros(z_gal.shape[0]) #np.concatenate(zerr_gal) # Here zerr is 0 because it is not given in the sims

    if (test_photozs==True):
        zerr_gal.fill(test_photozs_err)

    #Posterior without normalization at the moment, and with a delta function for z
    print "There are ", str(ra_gal.shape[0]), " galaxies within ", str(pb_frac*100.), "%, and z between ", z_min, z_max
    lnposterior=[]
    pixarea = hp.nside2pixarea(NSIDE)

    print "Estimating Posterior for H0 values:\n"
    for i in range(H0bins):
        lnposterior_bin = pos.lnprob(H0_array[i], z_gal, zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, pixarea, H0_min, H0_max, z_min, z_max, zerr_use=zerr_use, cosmo_use=cosmo_use)
        #print i, H0_array[i],lnposterior_bin
        lnposterior.append(lnposterior_bin)


    posterior[:,nevent]=np.exp(lnposterior)

    idx_max = np.argmax(lnposterior)

    perc_max = posterior[:idx_max].sum()/posterior.sum()

    maxposterior = posterior[np.argmax(lnposterior)]


    if blind:
        #Output path for blinding file
        blindpath = DIR_MAIN+"/blinding_file.p"
        H0_blinded_array = pos.make_blind(H0_array, blindpath)
        print 'Applying blinding factor. Saving value on ', blindpath
        print "\nBlinded ML percentile: ", perc_max
#        print "Blinded H0 ML: ", H0_blinded_array[idx_max], "+", pos.percentile(perc_max+0.34, posterior[:,nevent], H0_blinded_array)-H0_blinded_array[idx_max], "-", H0_blinded_array[idx_max] - pos.percentile(perc_max-0.34, posterior[:,nevent], H0_blinded_array)
        print "Blinded H0 Median: ", pos.percentile(0.50, posterior[:,nevent], H0_blinded_array)
    else:
        print 'No blinding applied!'
        print "\nML percentile: ", perc_max
#        print "H0 ML: ", H0_array[idx_max], "+", pos.percentile(perc_max+0.34, posterior[:,nevent], H0_array)-H0_array[idx_max], "-", H0_array[idx_max] - pos.percentile(perc_max-0.34, posterior[:,nevent], H0_array)
        print "H0 Median: ", pos.percentile(0.50, posterior[:,nevent], H0_array)


plt.clf()
for nevent in range(nevents):
    norm = np.trapz(posterior[:,nevent], H0_array)
    posterior[:,nevent] = posterior[:,nevent]/norm
    if blind:
        plt.plot(H0_blinded_array, posterior[:,nevent], label="Event "+str(nevent)+" - Blinded")
    else:
        plt.plot(H0_array, posterior[:,nevent], label="Event "+str(nevent))

if nevents>1:
    posterior_final = np.prod(posterior, axis=1)
    norm = np.trapz(posterior_final, H0_array)
    posterior_final = posterior_final/norm

    idx_max = np.argmax(posterior_final)
    perc_max = posterior_final[:idx_max].sum()/posterior_final.sum()
    maxposterior = posterior_final[idx_max]

    fmt = "%10.5f "
    if blind:
        plt.plot(H0_blinded_array, posterior_final, label="Final - Blinded")
        
        print "-------- Final Blinded H0 estimate ---------"
        print "ML percentile: ", perc_max
        H0_blinded_errp = pos.percentile(perc_max+0.34, posterior_final, H0_blinded_array)-H0_blinded_array[idx_max]
        H0_blinded_errm = H0_blinded_array[idx_max] - pos.percentile(perc_max-0.34, posterior_final, H0_blinded_array) 
        print "Blinded H0 ML: ", H0_blinded_array[idx_max], "+", H0_errp, "-", H0_errm
        print "Blinded H0 Median: ", pos.percentile(0.50, posterior_final, H0_blinded_array)
        cols = np.column_stack((H0_blinded_array,posterior, posterior_final))
        header = "H0 Blinded"
        for i in range(nevents+1):
            fmt=fmt+"%10.6e "
            header = header+" Posterior_"+str(i)
        header = header+" Final "
    else:
        plt.plot(H0_array, posterior_final, label="Final")

        print "-------- Final H0 estimate ---------"
        print "ML percentile: ", perc_max
        H0_errp = pos.percentile(perc_max+0.34, posterior_final, H0_array)-H0_array[idx_max]
       	H0_errm	= H0_array[idx_max] - pos.percentile(perc_max-0.34, posterior_final, H0_array)
        print "Blinded H0 ML: ", H0_blinded_array[idx_max], "+", H0_errp, "-", H0_errm   
        print "H0 Median: ", pos.percentile(0.50, posterior_final, H0_array)

        cols = np.column_stack((H0_array,posterior, posterior_final))
        header = "H0 "
        for i in range(nevents+1):
            fmt=fmt+"%10.6e "
            header = header+" Posterior_"+str(i)
        header = header+" Final "

else:
    if blind: 
        header = "Blinded H0 posterior"
        cols = np.column_stack((H0_array,posterior))
        fmt = "%10.5f %10.6e"
    else:
        header = "H0 posterior"
        cols = np.column_stack((H0_array,posterior))
        fmt = "%10.5f %10.6e"

plt.legend()
plt.xlabel('$H_0$ [km/s/Mpc]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.tight_layout()

if blind:
    plt.savefig(DIR_PLOTS+'H0_flask_posterior_'+str(nevents)+'_blinded.png')
    np.savetxt(DIR_OUT+'posterior_'+infile.rsplit('.')[0]+'_'+str(nevents)+'_blinded.txt',cols, header=header, fmt=fmt)
else:
    plt.savefig(DIR_PLOTS+'H0_flask_posterior_'+str(nevents)+'.png')
    np.savetxt(DIR_OUT+'posterior_'+infile.rsplit('.')[0]+'_'+str(nevents)+'.txt',cols, header=header, fmt=fmt)




