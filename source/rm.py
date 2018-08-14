import hp2np
import rotate
import numpy as np
import scipy.interpolate

def rotate_map ( target_ra=0.0, target_dec = 0.0) :
    ra_o,dec_o, map_o= hp2np.hp2np("skymap.fits",degrade=64)

    ra,dec,rotated_map = rot_map(ra_o,dec_o,map_o, target_ra, target_dec)

    return ra,dec,rotated_map

#
def rot_map (ra,dec,map, ra_max,dec_max, target_ra=0.0, target_dec = 0.0) :
    map_ring = map
    # Find the maximum of the LIGO skymap that one wishes to rotate
    #max = np.argmax(map_ring)

    # rotate the maximum pixel to the origin
    ra_0, dec_0 = rotate.rotateSkyToOrigin (ra,dec, ra_max, dec_max)
    # rotate the center pixel to the target ra,dec
    ra_t,dec_t = rotate.rotateSkyAwayFromOrigin(ra_0,dec_0, target_ra, target_dec)

    # covert the new map into a healpy ring map
    interp = scipy.interpolate.LinearNDInterpolator(zip(ra_t, dec_t), map)
    rotated_map = interp(ra,dec)

    return ra,dec,rotated_map

#   ra_max, dec_max is the peak of the ligo map
#   target_ra, target_dec is the peak of the DM map
def rot_all_maps (ra,dec,spatial, distance,sigma, norm, ra_max,dec_max, target_ra=0.0, target_dec = 0.0) :
    
    # rotate the maximum pixel to the origin
    ra_0, dec_0 = rotate.rotateSkyToOrigin (ra,dec, ra_max, dec_max)
    # rotate the center pixel to the target ra,dec
    ra_t,dec_t = rotate.rotateSkyAwayFromOrigin(ra_0,dec_0, target_ra, target_dec)
    
    # covert the new map into a healpy ring map
    interp = scipy.interpolate.LinearNDInterpolator(zip(ra_t, dec_t), spatial)
    rotated_spatial = interp(ra,dec)
    
    interp = scipy.interpolate.LinearNDInterpolator(zip(ra_t, dec_t), distance)
    rotated_distance = interp(ra,dec)
    
    interp = scipy.interpolate.LinearNDInterpolator(zip(ra_t, dec_t), sigma)
    rotated_sigma = interp(ra,dec)
    
    interp = scipy.interpolate.LinearNDInterpolator(zip(ra_t, dec_t), norm)
    rotated_norm = interp(ra,dec)
    
    
    return ra,dec,rotated_spatial, rotated_distance, rotated_sigma, rotated_norm


