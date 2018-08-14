# GW_standard_sirens
Estimate H0 from GW measurements and galaxy catalogs.

## Introduction

This code estimates a posterior probability for the Hubble constant using a skymap from LIGO and a galaxy catalog. The formalism is similar to the one described in Del Pozzo et al. 2012, Chen et al. 2017. It can be run on data, or it can rotate skymaps on density peaks for simulations.

This code has been written by Antonella Palmese and Jim Annis.

## Before running 

1. Uncompress the skymaps in the skymaps folder before running. skymap.fits is the LAL inference map for GW170814, skymapGW170817.fits is for GW170817.

2. Create a "catalogs" folder with the necessary catalogs. Flask simulations and DES/2MASS catalogs for the GW1708 events can be found at ```/data/des60.b/data/palmese/Dark_sirens/catalogs```

3. Create a plots directory (These steps will be put in a Makefile).

## Usage on simulations

```
python prepare_galaxy_catalog.py
python rotate_skymap_on_galaxies.py
python H0_nevents_flask.py
```

Step 1: prepare_galaxy_catalog.py adds healpix pixel information for each galaxy in the catalog, and makes it quicker to match catalogs to skymaps later on for hundreds of events.

Step 2: rotate_skymap_on_galaxies.py rotates the peak of the skymap to the position of galaxies which are found to be the closest to the redshift computed from the peak of the skymap luminosity distance and some input H0 provided by the user. The map can be rotated to nevents different galaxies. 

Step 3: H0_nevents_flask.py computes a posterior for N events, provided the skymaps from Step 2 and a galaxy catalog.

See the codes for a full list of user defined inputs (e.g. H0, redshift range considered...).

## Usage on data

H0_onevent_GW170817.py : Reads in a catalog in the GW170817 area, and compares the H0 posterior to the one computed when considering NGC 4993 only.

H0_onevent_hpix_search.py  : Reads in a catalog in healpix pixel files (as in the usual DES format) and a skymap, and produces a posterior. The default version uses the GW170814 skymap and a value added DES catalog, that can be downloaded from the DES machines.
