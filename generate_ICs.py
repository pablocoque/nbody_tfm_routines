# import modules
from nbodykit.lab import *
import numpy as np
# import matplotlib.pyplot as plt

# function to obtain initial Gaussian field for paired simulations
def obtain_paired(k, v):
    kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
    Pklinth = power.__call__(kk)
    Pdelta = v.real**2 + v.imag**2
    mask = Pdelta==0
    Pdelta[mask] = 1
    return v*np.sqrt(Pklinth)/np.sqrt(Pdelta)

# Global Parameters
Nc = 256
Length = 499.2 # Mpc  h-1
zobs = 0.3

# Setup initial conditions
cosmo = cosmology.Planck15

# Generate IC density field
power = cosmology.LinearPower(cosmo, redshift=zobs)
linear = LinearMesh(power, BoxSize=Length, Nmesh=Nc) # Gaussian realization of linear theory power spectrum
linear.save('Initialrealization.bigfile')
# # P(k) of initial field
# r = FFTPower(linear, mode="1d")
# Pkdelta = r.power['power'].real
# k_array = r.power['k']

# ################################
# fig = plt.figure(figsize=(5, 5))
# plt.imshow(linear.paint(mode='real').preview(axes=[0,1]))
# plt.title('Gaussian Realization')
# plt.savefig('gaussianfield.pdf')
# ################################

linearkfix = linear.apply(obtain_paired, mode='complex', kind='wavenumber')

deltafix = linearkfix.to_field(mode='real')
deltafix = FieldMesh(deltafix)
deltafix.save('pairedICs1.bigfile')

# r = FFTPower(deltafix, mode='1d')
# PkIC1 = r.power['power'].real

minusdeltafix = -1*linearkfix.to_field(mode='real')
minusdeltafix = FieldMesh(minusdeltafix)
minusdeltafix.save('pairedICs2.bigfile')

# r = FFTPower(minusdeltafix, mode='1d')
# PkIC2 = r.power['power'].real

# ################################
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(deltafix.paint(mode='real').preview(axes=[0,1]))
# ax[0].set_title('Simulation 1, ICs')
# ax[1].imshow(minusdeltafix.paint(mode='real').preview(axes=[0,1]))
# ax[1].set_title('Simulation 2, ICs')

# plt.savefig('paired_input_fields.pdf')
# ################################
# fig = plt.figure()
# plt.loglog(k_array, Pkdelta, label='Initial Gaussian realization')
# plt.loglog(k_array, PkIC1, 'x', label='ICs 1')
# plt.loglog(k_array, PkIC2, '+', label='ICs 2')
# plt.legend()
# plt.xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$')
# plt.ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^{3}]$')
# plt.savefig('Pk_ICs.pdf')
################################