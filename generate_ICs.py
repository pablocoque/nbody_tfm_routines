# import modules
from nbodykit.lab import *
from fastpm.nbkit import FastPMCatalogSource
import numpy as np

import sys
sys.path.append('../general_routines/')
from general_tools import *
from catalogue_generation import *

# Global Parameters
Nc = 256
Length = 499.2 # Mpc  h-1
zobs = 0.3

# Setup initial conditions
cosmo = cosmology.Planck15

# Generate IC density field
power = cosmology.LinearPower(cosmo, redshift=zobs)
# Gaussian realization of linear theory power spectrum
linear = LinearMesh(power, BoxSize=Length, Nmesh=Nc)
linear.save('Initialrealization.bigfile')

# Generate and save paired ICs
linearkfix = linear.apply(obtain_paired, mode='complex', kind='wavenumber')

deltafix = linearkfix.to_field(mode='real')
deltafix = FieldMesh(deltafix)
deltafix.save('paired1_Lmesh.bigfile')

minusdeltafix = -1*linearkfix.to_field(mode='real')
minusdeltafix = FieldMesh(minusdeltafix)
minusdeltafix.save('paired2_Lmesh.bigfile')

# Generate and save NL field
matter1 = FastPMCatalogSource(deltafix, cosmo=cosmo, Nsteps=10)
matterfield1 = matter1.to_mesh(resampler='cic', interlaced=True, compensated=True)
matterfield1.save('paired1_NLmesh.bigfile')

matter2 = FastPMCatalogSource(minusdeltafix, cosmo=cosmo, Nsteps=10)
matterfield2 = matter2.to_mesh(resampler='cic', interlaced=True, compensated=True)
matterfield2.save('paired2_NLmesh.bigfile')