import sys
from nbodykit.lab import *
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

def D(z):
    '''
    Obtain growth factor
    '''
    return cosmology.background.MatterDominated(cosmology.Planck15.Omega0_m).D1(1/(1+z))

######### MAIN

paired = sys.argv[1]

cosmo = cosmology.Planck15

### Import DM catalog
matter = BigFileCatalog('Matter_paired'+str(paired)+'.bigfile')

# Define global variables
Length = matter.attrs['BoxSize'][0]
Nc = matter.attrs['Nmesh'][0]
zobs = 0.3
zinit = 3.

# Compute matter density field
delta_dm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k,z)
r = FFTPower(delta_dm, mode="1d", Nmesh=Nc)
Pkdm = r.power['power'].real - r.power.attrs['shotnoise']
k = r.power['k']
mask = (k <= 0.04)*(k > 0.02)

### Import Halos catalog
halos = BigFileCatalog('Halos_paired'+str(paired)+'_reconvr.bigfile')
ngal = halos.csize

# Compute galaxy density field
delta_h = halos.to_mesh(resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_h, mode="1d", Nmesh=Nc)
Pkh = r.power['power'].real - r.power.attrs['shotnoise']

delta_hRSD = halos.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
delta_hQ = halos.to_mesh(position='PositionQ', resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_hQ, mode="1d", Nmesh=Nc)
PkhQ = r.power['power'].real - r.power.attrs['shotnoise']

bg = np.sqrt(np.mean(Pkh[mask]/Pkdm[mask]))
bgrecon1 = np.mean(np.sqrt(PkhQ[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))
bgrecon2 = np.mean(np.sqrt(PkhQ[mask]/Pkdm[mask]))

print('bias halos PRE= {:.2f}'.format(bg))
print('bias halos POST= {:.2f},{:.2f}'.format(bgrecon1,bgrecon2))

# Compute celestial coordinates pre-reconstruction
halos['SkyCoordz'] = transform.CartesianToSky(pos=halos['Position'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
halos['SkyCoordzpec'] = transform.CartesianToSky(pos=halos['Position'], cosmo=cosmo, velocity=halos['Velocity'], \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = halos['SkyCoordz'].compute()
coord_array[:, 3] = halos['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('pairedsim'+str(paired)+'_cat_pre_z0.3.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.2f')

# Compute celestial coordinates post-reconstruction
halos['SkyCoordz'] = transform.CartesianToSky(pos=halos['PositionQ'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
halos['SkyCoordzpec'] = transform.CartesianToSky(pos=halos['PositionQ'], cosmo=cosmo, velocity=halos['Velocity'], \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = halos['SkyCoordz'].compute()
coord_array[:, 3] = halos['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('pairedsim'+str(paired)+'_cat_post_z3.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.2f')

fig = plt.figure()
plt.loglog(k, Pkh, label='Pre-reconstruction, z=0.3')
plt.loglog(k, PkhQ, label='Post-reconstruction, z=3')
plt.legend()
plt.xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$')
plt.ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^{3}]$')
plt.savefig('Pkhalosrecon_pairedsim'+str(paired)+'.pdf')

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(delta_hRSD.paint(mode='real').preview(axes=[0,1]))
ax[0].set_title('Pre-reconstruction')
ax[1].imshow(delta_hQ.paint(mode='real').preview(axes=[0,1]))
ax[1].set_title('Post-reconstruction')
plt.savefig('reconstructed_pairedsim'+str(paired)+'.pdf')