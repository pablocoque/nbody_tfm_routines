# import modules
import sys
from nbodykit.lab import *
from fastpm.nbkit import FastPMCatalogSource
import numpy as np

# Files to read from
paired = sys.argv[1]

# Setup initial conditions
cosmo = cosmology.Planck15

# Generate IC density field
linear = BigFileMesh('Initialrealization.bigfile', dataset='Field')

# Global Parameters
Nc = linear.attrs['Nmesh'][0]
Length = linear.attrs['BoxSize'][0]
zobs = float(linear.attrs['plin.redshift'])

# P(k) of initial field
r = FFTPower(linear, mode="1d")
Pkdelta = r.power['power'].real
k_array = r.power['k']

deltaIC = BigFileMesh('pairedICs'+str(paired)+'.bigfile', dataset='Field')

r = FFTPower(deltaIC, mode='1d')
PkIC = r.power['power'].real

matter = FastPMCatalogSource(deltaIC, cosmo=cosmo, Nsteps=10)
matterfield = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)
matter.save('Matter_paired'+str(paired)+'.bigfile')

r = FFTPower(matterfield, mode='1d')
Pk = r.power['power'].real - r.power.attrs['shotnoise']

# Run FOF to identify halo groups
DM_part_mass = .5e12
fof = FOF(matter, linking_length=0.4, nmin=5)
halos = fof.to_halos(cosmo=cosmo, redshift=zobs, particle_mass=DM_part_mass, mdef='vir')

nhalos = halos.csize
delta_halos = halos.to_mesh(resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_halos, mode="1d", Nmesh=Nc)
Pkhalos = r.power['power'].real - r.attrs['shotnoise']

print('Halo number density = {:.3e}'.format(nhalos/(Length**3)))

# move halos to redshift space

## RSD formula
# vr = 1/(Ha) * (vec(v).vec(r))vec(r)

## Box centered
# Simulation 1
position_origin = halos['Position'] - 0.5*Length

projection_norm = np.linalg.norm(position_origin, axis=1)

line_of_sight = np.zeros_like(position_origin)
line_of_sight = position_origin/projection_norm[:, np.newaxis]

rsd_factor = (1+zobs) / (100 * cosmo.efunc(zobs))

dot_prod = np.sum(halos['Velocity']*line_of_sight, axis=1)

halos['PositionRSD'] = position_origin + rsd_factor*dot_prod[:, np.newaxis]*line_of_sight + 0.5*Length

delta_hRSD = halos.to_mesh(resampler='cic', position='PositionRSD', interlaced=True, compensated=True)
r = FFTPower(delta_hRSD, mode='1d', Nmesh=Nc)
PkhRSD = r.power['power'].real - r.attrs['shotnoise']

# Bias from halos PRE
mask = (k_array <= 0.09)*(k_array > 0.03)
bhalos = np.mean(np.sqrt(Pkhalos[mask]/Pk[mask]))

print('bias halos PRE= {:.2f}'.format(bhalos))

halos.save('Halos_paired'+str(paired)+'.bigfile')