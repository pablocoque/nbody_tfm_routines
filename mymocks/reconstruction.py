from nbodykit.lab import *
import numpy as np
import dask.array as da

import sys
sys.path.append('../general_routines/')
from general_tools import *
from iterative_reconstruction import *

### MAIN

paired = sys.argv[1]

cosmo = cosmology.Planck15

### Import matter catalog
matter = BigFileCatalog('Matterpaired'+str(paired)+'_catalog.bigfile')

# Define global variables
Length = matter.attrs['BoxSize'][0]
Nc = matter.attrs['Nmesh'][0]
zobs = 0.3
zinit = 3.
r_s = 2.*(Length/Nc) # smoothing radius
print('Smoothing radius=', r_s)

### Import galaxy catalog
galaxy = BigFileCatalog('Galaxypaired'+str(paired)+'_catalog.bigfile')
ngal = galaxy.csize

# Evaluate bias in real and redshift space
bg = evaluate_bias(galaxy, matter)
bgRSD = evaluate_bias(galaxy, matter, tracer_pos='PositionRSD')

print('bias (real) PRE= {:.2f} +- {:.2f}'.format(bg, np.std(np.sqrt(Pkg[mask]/Pkdm[mask]))))
print('bias (redshift) PRE= {:.2f} +- {:.2f}'.format(bgRSD, np.std(np.sqrt(PkgRSD[mask]/Pkdm[mask]))))

# Valores de formulas
breczinitt = (bg-1)*(D(zobs)/D(zinit)) + 1
breczobst = (bg-1) + (D(zinit)/D(zobs))

print('Expected bias brez(zinit) = {:.2f}'.format(breczinitt))
print('Expected bias brez(zobs) = {:.2f}'.format(breczobst))

# Compute celestial coordinates pre-reconstruction
galaxy['SkyCoordz'] = transform.CartesianToSky(pos=galaxy['Position'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
galaxy['SkyCoordzpec'] = transform.CartesianToSky(pos=galaxy['PositionRSD'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = galaxy['SkyCoordz'].compute()
coord_array[:, 3] = galaxy['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('paired'+str(paired)+'_cat_pre_z0.3.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.5f')

# Apply reconstruction
iterative_reconstruction(3, Length, Nc, zobs, zinit, galaxy, matter, np.array([Length/2, Length/2, Length/2]), real_space=False, test=False

# Bias in real space
bgreconr2 = evaluate_bias(galaxy, matter, tracer_pos='PositionQ')
bgreconr1 = bgreconr2*D(zobs)/D(zinit)
print('bias (real) POST= {:.2f}(z={:.2f}),{:.2f}(z={:.2f})'.format(bgreconr1, zinit,bgreconr2, zobs))

# Bias in redshift space
bgrecons2 = evaluate_bias(galaxy, matter, tracer_pos='PositionQS')
bgrecons1 = bgrecons2*D(zobs)/D(zinit)
print('bias (redshift) POST= {:.2f}(z={:.2f}),{:.2f}(z={:.2f})'.format(bgrecons1, zinit,bgrecons2, zobs))

# Compute celestial coordinates post-reconstruction
galaxy['SkyCoordz'] = transform.CartesianToSky(pos=galaxy['PositionQ'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
galaxy['SkyCoordzpec'] = transform.CartesianToSky(pos=galaxy['PositionQS'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = galaxy['SkyCoordz'].compute()
coord_array[:, 3] = galaxy['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('paired'+str(paired)+'_cat_post_z3.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.5f')

galaxy.save('Galaxypaired'+str(paired)+'_catalog_POST.bigfile')