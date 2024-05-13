from nbodykit.lab import *
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
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

# Compute matter density field
delta_dm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k,z)
r = FFTPower(delta_dm, mode="1d")
Pkdm = r.power['power'].real - r.power.attrs['shotnoise']
k = r.power['k']

### Import galaxy catalog
galaxy = BigFileCatalog('Galaxypaired'+str(paired)+'_catalog.bigfile')

# Compute galaxy density field
delta_g = galaxy.to_mesh(resampler='cic', interlaced=True, compensated=True)
delta_gRSD = galaxy.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
ngal = galaxy.csize

# Compute galaxy P(k,z)
r = FFTPower(delta_g, mode='1d')
Pkg = r.power['power'].real - r.attrs['shotnoise']

r = FFTPower(delta_gRSD, mode='1d')
PkgRSD = r.power['power'].real - r.attrs['shotnoise']

# Calculate bias pre reconstruction
mask = (k <= 0.09)*(k > 0.03)
bg = np.mean(np.sqrt(Pkg[mask]/Pkdm[mask]))
bgRSD = np.mean(np.sqrt(PkgRSD[mask]/Pkdm[mask]))

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

reconstructedQ, reconstructedQS = iteration(3, Length, Nc, zobs, zinit, galaxy, np.array([Length/2, Length/2, Length/2]), k, Pkg)#, delta_dm)

# Bias in real space
r = FFTPower(reconstructedQ, mode='1d')
Pkgreconr = r.power['power'].real - r.attrs['shotnoise']
bgreconr1 = np.mean(np.sqrt(Pkgreconr[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))

bgreconr2 = np.mean(np.sqrt(Pkgreconr[mask]/Pkdm[mask]))
print('bias (real) POST= {:.2f}(z={:.2f}),{:.2f}(z={:.2f})'.format(bgreconr1, zinit,bgreconr2, zobs))

# Bias in redshift space
r = FFTPower(reconstructedQS, mode='1d')
Pkgrecons = r.power['power'].real - r.attrs['shotnoise']

bgrecons1 = np.mean(np.sqrt(Pkgrecons[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))

bgrecons2 = np.mean(np.sqrt(Pkgrecons[mask]/Pkdm[mask]))

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