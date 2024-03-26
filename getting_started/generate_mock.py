from nbodykit.lab import *
from fastpm.nbkit import FastPMCatalogSource
import numpy as np

# Global Parameters
Nc = 256
Length = 500 # Mpc  h-1
zobs = 0.

# Setup initial conditions
cosmo = cosmology.Planck15
power = cosmology.LinearPower(cosmo, redshift=zobs)
linear = LinearMesh(power, BoxSize=Length, Nmesh=Nc, seed=42)
linear.save('linear_ic_mesh.bigfile')

# P(k) of initial field
# r = FFTPower(linear, mode='1d')
# r.save('linear-power.json')


# Run the FastPM particle mesh simulation
matter = FastPMCatalogSource(linear, Nsteps=10)

# Save FastPM DM catalog and power spectrum to file
matter.save('fastpm_dmcatalog.bigfile')
# r = FFTPower(matter, mode='1d', Nmesh=512)
# r.save('matter-power.json')

# Compute matter density field
# delta_dm = matter.to_mesh(resampler='cic', interlaced=False, compensated=False)
# delta_dm.save('delta_dm_mesh.bigfile')

ndm = matter.csize

# Run FOF to identify halo groups
cellsize = Length/Nc
fof = FOF(matter, linking_length=0.4, nmin=20)

# DM particle mass
h = cosmology.Planck15.h
Omega_M = cosmology.Planck15.Omega0_cdm + cosmology.Planck15.Omega0_b
H0Mpc = 100.0*h*(3.24078e-20)
GMsunMpcS = 4.5182422e-48
rhomean = (3.0/(8.0*np.pi*GMsunMpcS))*Omega_M*(H0Mpc**2)
Mdm = rhomean*(Length**3)*h**3
DM_part_mass = Mdm/ndm

# Create halos
halos = fof.to_halos(DM_part_mass, cosmo, redshift=zobs)

# Populate halos with galaxies
hod = halos.populate(Zheng07Model)
hod.save('hod_gcatalog.bigfile')

# Compute galaxy density field
# delta_g = hod.to_mesh(resampler='cic', interlaced=True, compensated=True)
# delta_g.save('delta_g_mesh.bigfile')

# Compute and save galaxy P(k,z)
# r = FFTPower(delta_g, mode="1d", Nmesh=Nc)
# r.save('galaxy-power.json')