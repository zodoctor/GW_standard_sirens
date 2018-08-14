# GW_standard_sirens
Estimate H0 from GW measurements and galaxy catalogs.

## Introduction

This code estimates a posterior probability for the Hubble constant using a skymap from LIGO and a galaxy catalog.

Remember to uncompress the skymaps before running.

This code has been written by Antonella Palmese and Jim Annis.

## Usage on simulations
```
python prepare_galaxy_catalog.py
python rotate_skymap_for_catalog.py
python H0_onevent_hpix_search_flask.py
```

## Usage on data

H0_onevent_GW170817.py : Reads in a catalog in the GW170817 area, and compares the posterior to 
