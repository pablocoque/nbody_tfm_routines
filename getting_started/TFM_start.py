from nbodykit.lab import *
from fastpm.nbkit import FastPMCatalogSource
from nbodykit import setup_logging, style
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# def gaussian_filter(k, v, bg):
#     R = 5 # Mpc h-1
#     kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
#     kk[kk == 0] = 1
#     return v * np.exp(-0.5*kk*R**2)/bg

Nc=200
Length=1000 # 1 Gpc

#Set up IC
print("Computing P(k) linear with Planck15 cosmology")
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, 0)
linear = LinearMesh(Plin, BoxSize=Length, Nmesh=Nc,seed=42)

#Plot linear density
# one_plus_delta = linear.paint(mode='real')
#Compute P(k) of the IC
# print("-----> Saving linear P(k) in Lin_power_spectrum.txt")
# r = FFTPower(linear, mode="1d", Nmesh=Nc)
# k = r.power['k']
# Pk = r.power['power'].real

#Run simulation
print("Running FastPM simulation")
sim = FastPMCatalogSource(linear, Nsteps=10)

#Plot DM density
mesh = sim.to_mesh(resampler='tsc')
one_plus_delta_dm = mesh.paint(mode='real')

#Compute P(k) of the DM
# print("-----> Saving DM P(k) in DM_power_spectrum.txt")
r = FFTPower(sim, mode="1d", Nmesh=Nc)
k = r.power['k']
Pk_dm = r.power['power'].real - r.power.attrs['shotnoise']

#Compute the mass of each DM particle
print("Computing the mass of DM particle")
h = cosmology.Planck15.h
Omega_M = cosmology.Planck15.Omega0_cdm + cosmology.Planck15.Omega0_b
H0Mpc = 100.0*h*(3.24078e-20)
GMsunMpcS = 4.5182422e-48
pi = 3.14159265359
rhomean = 3.0/(8.0*pi*GMsunMpcS)*Omega_M*(H0Mpc*H0Mpc)
Mdm = rhomean*Length*Length*Length/h/h
DM_part_mass = Mdm/(Nc*Nc*Nc)
# print(DM_part_mass)

# run FOF to identify halo groups
print("Running FoF to compute halos")
fof = FOF(sim, 0.2*(Length/Nc), nmin=20, absolute=True)
halos = fof.to_halos(DM_part_mass, cosmo, 0.)

#Plot Halos density
# mesh = halos.to_mesh(resampler='tsc')
# one_plus_delta_halos = mesh.paint(mode='real')

# #Compute P(k) of the Halos
# r = FFTPower(halos, mode="1d", Nmesh=Nc)
# k = r.power['k']
# nhal=halos.csize
# print("-----> Number of halos = ",nhal)
# Pk = r.power['power'].real - r.power.attrs['shotnoise']

# populate halos with galaxies
print("Computing HOD with Zheng07 model")
hod = halos.populate(Zheng07Model)

#Plot HOD density
mesh = hod.to_mesh(resampler='tsc')
one_plus_delta_hod = mesh.paint()

# Compute and save Pk
r = FFTPower(hod, mode="1d", Nmesh=Nc)
k = r.power['k']
# ngal=hod.csize
# print("-----> Number of galaxies = ",ngal)
Pk_g = r.power['power'].real - r.power.attrs['shotnoise']

print('Calculating bias')
mask = (k <= 0.1)*(k > 0.03)
bg = np.sqrt(np.mean(Pk_g[mask]/Pk_dm[mask]))
print('bias = {:.2f}'.format(bg))

print('Applying Gaussian Smoothing')
# filtered_mesh = delta_hod.apply(gaussian_filter(bg), kind='wavenumber', mode='complex')
# delta_dm_tilde = filtered_mesh.paint(mode='real')
one_plus_delta_dm_tilde = ArrayMesh(gaussian_filter(one_plus_delta_hod/bg, sigma = 10*Nc/Length), BoxSize = Length)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(one_plus_delta_hod.preview(axes=[0,1]))
ax[1].imshow(one_plus_delta_dm_tilde.preview(axes=[0,1]))
ax[2].imshow(one_plus_delta_dm.preview(axes=[0,1]))
plt.show()

print('Shifting to RS')
line_of_sight = [0,0,1]
hod['RSDPosition'] = hod['Position'] + hod['VelocityOffset'] * line_of_sight