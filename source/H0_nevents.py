import matplotlib
matplotlib.use('Agg')
from astropy. io import fits
import healpy as hp
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c #m/s
from scipy.stats import norm
from astropy.cosmology import FlatLambdaCDM, z_at_value
import argparse
import os
import posterior_utils as pos
import astropy.units as u 

parser = argparse.ArgumentParser(description='Compute H0 posterior given N events')

parser.add_argument('--infile', default='truthChinchillaY3.fits',
                    help='Input galaxy catalog name with hpix numbers.')
parser.add_argument('--skymap', default='simevents',
                    help='Input skymap')
parser.add_argument('--nevents', type=int,  
                    help='Number of events to run. If not given, will combine all events in the event_list.')
parser.add_argument('--zmin', default=0.0, type=float,
                    help='Minimum redshift to be considered in the galaxy catalog. Default is 0.0')
parser.add_argument('--zmax', default=0.3, type=float,
                    help='Maximum redshift to be considered in the galaxy catalog. Default is 0.3')
parser.add_argument('--Hmin', default=0.0, type=float,
                    help='Minimum H0 for a flat prior. Default is 0.0')
parser.add_argument('--Hmax', default=100., type=float,
                    help='Maximum H0 for a flat prior. Default is 100.0')
parser.add_argument('--cosmo_use', default=True, type=bool,
                    help='Use full cosmology for dL.')
parser.add_argument('--zerr_use', default=False,  type=bool,
                    help='Galaxy redshift is a Gaussian instead of delta function.')
parser.add_argument('--colnames', default=['z','zerr','ra','dec'], type=str,
                    nargs=4,
                    help='the three column names in the galaxy catalog for redshift, redshift error, RA, Dec')
parser.add_argument('--blind', default=False,  type=bool,
                    help='Blinding of the H0 results.')
parser.add_argument('--zerr', default=0.0, type=float,
                    help='Redshift uncertainty, assuming gaussian.')
parser.add_argument('--maglim', default=999., type=float,
                    help='Limiting magnitude of the galaxy catalog.')
parser.add_argument('--rootdir', #default='/data/des41.a/data/marcelle/GW170814/dark-siren',
                    help='Path to top level work dir')
parser.add_argument('--p', default=0.90, type=float, 
                    help='Fraction of the skymap probability to consider. Default is 0.9 (decrease for speed)')
parser.add_argument('--test', default=0, type=int, choices=range(1,2),
                    help='Pick one of the simple pre-defined sanity checks to run.\n Test 1: scrambled galaxies')
parser.add_argument('--Hbins', default=200, type=int,
                    help='Number of H0 bins to evaluate. Default is 200 (decrease for speed)')

args = parser.parse_args()

infile = args.infile
glxcat = infile.split('.fits')[0]
skymap = (args.skymap).split('.fits')[0]
#nevents = args.nevents
event_list = skymap+'.txt'
z_min = args.zmin
z_max = args.zmax
H0_min = args.Hmin
H0_max = args.Hmax
cosmo_use = args.cosmo_use
zerr_use = args.zerr_use
blind = args.blind
zerr = args.zerr
maglim = args.maglim

outlabel='posterior_'+glxcat+'_'+str(args.maglim)+'_'+str(args.zerr)+'_'+skymap
if blind: outlabel=outlabel+'_blinded'


# Add by hand an error test_photozs_err to the galaxy redshifts. Note this is fixed for all galaxies for now!

test_photozs = True
test_photozs_err = 0.01
pb_frac = args.p #Fraction of the skymap probability to consider, decrease for speed
H0bins = args.Hbins

# Names of the input galaxy catalog columns

colnames = args.colnames
ra_column_name = colnames[2]
dec_column_name = colnames[3]
z_column_name = colnames[0]
zerr_column_name = colnames[1]

DIR_SOURCE = os.getcwd()
DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]
#DIR_CATALOG = "/Users/palmese/work/GW/Dark_sirens/catalogs/buzzard/" # DIR_MAIN+'/catalogs/'

hpix_column_name = 'hpix1024' #If it exists in the galaxy catalog, if not, it is computed for the skymap given its NSIDE, default is ring
nest = False #Watch out! mock maps have nest=True but BCC has False at the moment
mag_column_name = 'OMAG'

if args.rootdir == None:
    DIR_SOURCE = os.getcwd()
    DIR_MAIN = DIR_SOURCE.rsplit('/', 1)[0]
else:
    DIR_MAIN = args.rootdir

DIR_CATALOG = DIR_MAIN+'/catalogs/'
DIR_SKYMAP = DIR_MAIN+'/skymaps/'
DIR_PLOTS = DIR_MAIN+'/plots/'
DIR_OUT = DIR_MAIN+'/out/'

plot_axvlines = True

if os.path.isfile(DIR_CATALOG+event_list):

    print "Reading list of events file..."

    events=np.genfromtxt(DIR_CATALOG+event_list, 
                         dtype="i8,i8,f8,f8,f8,f8,f8,f8",
                         names=['id','host_id','ra','dec','z','d','derr','roi_size'], 
                         comments = "#")

    if events.shape == ():
        nevents = 1
    else:
        if args.nevents == None: 
            nevents = events.shape[0]
        else:
            nevents = min(args.nevents,events.shape[0])
        
    if nevents == 1: skymap = skymap+'0'
    
    with open(DIR_CATALOG+event_list) as fp:  
        cosmo=fp.readline().split(" ")[-1].strip('\n')
        H0=float(fp.readline().split(" ")[-1])
        Omega_m=float(fp.readline().split(" ")[-1])
        print "Input cosmology:", cosmo 
        print " H0:", H0 
        print " Omega_m:", Omega_m

else:
    nevents=1
    cosmo=None
    H0=70.0 
    Omega_m=0.3

print "Number of events to process:", nevents

print "Reading in galaxy catalogs..."

h = fits.open(DIR_CATALOG+infile)[1].data

try:
    mask_z = ( (h[z_column_name]>z_min) & (h[z_column_name]<z_max) & (h[mag_column_name][:,1]<maglim) )
except KeyError:
    mask_z = ( (h[z_column_name]>z_min) & (h[z_column_name]<z_max))

if (args.test==1):
    print "Running Test Mode 1: Scrambling galaxies ra,dec,z values"
    nvalues=h[ra_column_name].size
    minvalue=min(h[ra_column_name])
    maxvalue=max(h[ra_column_name])
    h[ra_column_name]=np.random.uniform(minvalue,maxvalue,nvalues)
    minvalue=np.deg2rad(min(h[dec_column_name])+90)
    maxvalue=np.deg2rad(max(h[dec_column_name])+90)
    h[dec_column_name]=np.rad2deg(np.arccos(1-2*np.random.uniform(minvalue,maxvalue,nvalues)))-90.
    h[z_column_name]=np.zeros(nvalues)
    nmaskedvalues=h[z_column_name][mask_z].size
    hz=np.random.uniform(z_min,z_max,nmaskedvalues)    
    h[z_column_name]=hz.resize(nvalues)
    maglim = 999.
    zerr_use = False
    outlabel=outlabel+'_test'+str(args.test)
    cosmo=None
    plot_axvlines=False
    H0bins=25


ra_g=h[ra_column_name][mask_z]
dec_g=h[dec_column_name][mask_z]


z_g = h[z_column_name][mask_z]

if zerr_column_name in h.dtype.names:
    zerr_g = h[zerr_column_name][mask_z]
    zerr = np.mean(zerr_g)
else:
    print "No zerr provided in input catalog. Setting errors to "+str(zerr)+" * (1+z)"
    zerr_g = zerr * (1+z_g)
    if zerr > 0.:
        z_g = np.random.normal(loc=z_g, scale=zerr_g)

if zerr<=0.:
    zerr_use = False

H0_array = np.linspace(H0_min,H0_max,num=H0bins)
posterior = np.zeros((H0_array.shape[0],nevents))

distmu_average = np.zeros(nevents)
distsigma_average = np.zeros(nevents)


for nevent in range(nevents):

    print "Reading skymap for event ", str(nevent+1)
    if (nevents==1):
        skymap_name = DIR_SKYMAP+skymap+".fits"
    else:
        skymap_name = DIR_SKYMAP+skymap+str(nevent)+".fits"
    #map = fits.open(skymap_name)[1].data
    pb,distmu,distsigma,distnorm = hp.read_map(skymap_name, field=range(4))
    NSIDE = hp.npix2nside(len(pb))
    pixarea = hp.nside2pixarea(NSIDE)
    pixarea_deg2 = hp.pixelfunc.nside2pixarea(NSIDE,degrees=True) 
    try:
        pix_g = h[hpix_column_name][mask_z]
    except:
        phi_g = ra_g*pi/180.
        theta_g = (90.-dec_g)*pi/180.
        pix_g = hp.pixelfunc.ang2pix(NSIDE, theta_g, phi_g, nest=nest)
        print "No hpix column in catalog"


    #pb = map['PROB']
    #distmu = map['DISTMU']
    #distsigma = map['DISTSIGMA']
    #distnorm = map['DISTNORM']

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

    roi_area = pixarea_deg2 * id
    print "ROI total area: ", roi_area, " , npix:", id

    distmu_average[nevent] = np.average(distmu[idx_sort_cut],weights=pb[idx_sort_cut])
    distsigma_average[nevent] = np.average(distsigma[idx_sort_cut],weights=pb[idx_sort_cut])

    z_gal , zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, ra_gal, dec_gal = [], [], [], [], [], [], [], []

    print "Assigning probabilities to galaxies.."

    for idx_hpix in idx_sort_cut:
        idx_this_hpix = (pix_g == idx_hpix)
        z_gal.append(z_g[idx_this_hpix])
        ra_gal.append(ra_g[idx_this_hpix])
        dec_gal.append(dec_g[idx_this_hpix])
        zerr_gal.append(zerr_g[idx_this_hpix])
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
    zerr_gal = np.concatenate(zerr_gal) 

    #Posterior without normalization at the moment, and with a delta function for z
    print "There are ", str(ra_gal.shape[0]), " galaxies within ", str(pb_frac*100.), "%, and z between ", z_min, z_max
    lnposterior=[]

    print "Estimating Posterior for H0 values:"
    for i in range(H0bins):
        lnposterior_bin = pos.lnprob(H0_array[i], z_gal, zerr_gal, pb_gal, distmu_gal, distsigma_gal, distnorm_gal, pixarea, H0_min, H0_max, z_min, z_max, zerr_use=zerr_use, cosmo_use=cosmo_use,omegam=Omega_m)
        #print i, H0_array[i],lnposterior_bin
        lnposterior.append(lnposterior_bin)


    posterior[:,nevent]=np.exp(lnposterior)

    # incorporate \beta(H0), ie. GW selection effect provided
    # by Maya
    betafile = np.genfromtxt('H0_normalization_zmax035.txt',names='H0,beta',dtype=float)
    inv_beta_of_H0 = np.interp(H0_array,betafile['H0'],betafile['beta'])
    posterior[:,nevent] = np.multiply(posterior[:,nevent],inv_beta_of_H0)
    
    idx_max = np.argmax(lnposterior)

    perc_max = posterior[:idx_max].sum()/posterior.sum()

    maxposterior = posterior[np.argmax(lnposterior)]
    cl=0.68
    idx_err_p = pos.perc_idx(perc_max+cl/2,posterior[:,nevent])
    idx_err_m = pos.perc_idx(perc_max-cl/2,posterior[:,nevent])

    if blind:
        #Output path for blinding file
        blindpath = DIR_MAIN+"/blinding_file.p"
        H0_blinded_array = pos.make_blind(H0_array, blindpath)
        print 'Applying blinding factor. Saving value on ', blindpath
        H0_array_out = H0_blinded_array
        print "Blinded results:"
    else:
        print 'No blinding applied!'
        H0_array_out = H0_array

    H0_maxlike=H0_array_out[idx_max]
    H0_err_p=abs(H0_array_out[idx_err_p] - H0_array_out[idx_max])
    H0_err_m=abs(H0_array_out[idx_err_m] - H0_array_out[idx_max])
    H0_median=pos.percentile(0.50, posterior[:,nevent], H0_array_out)

    print " ML percentile: ", perc_max
    print " H0 ML: ", H0_maxlike, "+", H0_err_p, "-", H0_err_m 
    print " H0 Median: ", H0_median

fmt = "%10.5f"
if blind:        
    header = "H0_Blinded"
else:
    header = "H0"

for nevent in range(nevents):
    norm = np.trapz(posterior[:,nevent], H0_array_out)
    posterior[:,nevent] = posterior[:,nevent]/norm
    dl=int(distmu_average[nevent])
    plt.plot(H0_array_out, posterior[:,nevent], label="Event "+str(nevent)+": "+str(dl)+" Mpc")
    fmt=fmt+" %10.6e"
    header = header+" Posterior_"+str(nevent)

if nevents == 1: 
    cols = np.column_stack((H0_array_out,posterior))
    header = header.split()[0]+" Posterior"
    plt.clf()
    dl=int(distmu_average[0])
    plt.plot(H0_array_out, posterior[:,nevent], color='k', label=str(dl)+" Mpc")

if nevents>1:
    posterior_final = np.prod(posterior, axis=1)
    norm = np.trapz(posterior_final, H0_array_out)
    posterior_final = posterior_final/norm

    idx_max = np.argmax(posterior_final)
    perc_max = posterior_final[:idx_max].sum()/posterior_final.sum()
    maxposterior = posterior_final[idx_max]
    idx_err_p = pos.perc_idx(perc_max+cl/2,posterior[:,nevent])
    idx_err_m = pos.perc_idx(perc_max-cl/2,posterior[:,nevent])

    H0_maxlike=H0_array_out[idx_max]
    HO_err_p=abs(H0_array_out[idx_err_p] - H0_array_out[idx_max])
    H0_err_m=abs(H0_array_out[idx_err_m] - H0_array_out[idx_max])  
    H0_median=pos.percentile(0.50, posterior[:,nevent], H0_array_out)

    if blind: 
        print "-------- Final Blinded H0 estimate ---------"
    else:
        print "-------- Final H0 estimate ---------"

    print " ML percentile: ", perc_max
    print " H0 ML: ", H0_maxlike, "+", H0_err_p, "-", H0_err_m 
    print " H0 Median: ", H0_median
    if cosmo != None: print " H0 true: ", H0

    cols = np.column_stack((H0_array_out,posterior, posterior_final))

    plt.plot(H0_array_out, posterior_final, label="Final",color='k')

    fmt=fmt+" %10.6e"
    header = header+" Posterior_Final"

if plot_axvlines:
    plt.axvline(x=H0_array_out[idx_max],color='k')
    plt.axvline(x=H0_array_out[idx_err_p],color='k',linestyle='dotted')
    plt.axvline(x=H0_array_out[idx_err_m],color='k',linestyle='dotted')
    if cosmo != None : plt.axvline(x=H0,color='k',linestyle='dashed')
plt.legend()
plt.xlabel('$H_0$ [km/s/Mpc]',fontsize=20)
plt.ylabel('$p$',fontsize=20)
plt.title(outlabel)
plt.tight_layout()

outfile = open(DIR_OUT+outlabel+'_final_summary.txt','w')
outfile.write('# nevents maglim zerr H0_maxlike H0_errp H0_errm H0_median H0_true\n')
outfile.write('%i %f %f %f %f %f %f %f\n' % ( nevents, maglim, zerr, H0_maxlike, H0_median, H0_err_p, H0_err_m, H0 ))
outfile.close()

if os.path.isfile(DIR_CATALOG+event_list):
    outfile = open(DIR_OUT+outlabel+'_nevents_summary.txt','w')
    outfile.write('# id host_id ra dec z d derr roi_size roi_area maglim zerr H0_maxlike H0_errp H0_errm H0_median H0_true\n')
    if events.shape == () :
        outfile.write('%i %i %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % ( events['id'], events['host_id'], events['ra'], events['dec'], events['z'], events['d'], events['derr'], events['roi_size'], roi_area, maglim, zerr, H0_maxlike, H0_err_p, H0_err_m, H0_median, H0 ))
    else:
        for i in range(nevents):
            outfile.write('%i %i %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % ( events['id'][i], events['host_id'][i], events['ra'][i], events['dec'][i], events['z'][i], events['d'][i], events['derr'][i], events['roi_size'][i], roi_area, maglim, zerr, H0_maxlike, H0_err_p, H0_err_m, H0_median, H0 ))
    outfile.close()

plt.savefig(DIR_PLOTS+outlabel+'.png')
np.savetxt(DIR_OUT+outlabel+'.txt',cols,header=header,fmt=fmt)





