from nbodykit.lab import *
import matplotlib.pyplot as plt
import numpy as np

linear = BigFileMesh('paired1_Lmesh.bigfile', dataset='Field')
r = FFTPower(linear, mode="1d")
Pklin = r.power['power'].real
k_array = r.power['k']

matter1 = BigFileCatalog('Matterpaired1_catalog.bigfile')
matterfield1 = matter1.to_mesh(resampler='cic', interlaced=True, compensated=True)
r = FFTPower(matterfield1, mode='1d')
Pk1 = r.power['power'].real - r.power.attrs['shotnoise']

matter2 = BigFileCatalog('Matterpaired2_catalog.bigfile')
matterfield2 = matter2.to_mesh(resampler='cic', interlaced=True, compensated=True)
r = FFTPower(matterfield2, mode='1d')
Pk2 = r.power['power'].real - r.power.attrs['shotnoise']

galaxy1 = BigFileCatalog('Galaxypaired1_catalog.bigfile')
delta_g1 = galaxy1.to_mesh(resampler='cic', interlaced=True, compensated=True)
delta_gRSD1 = galaxy1.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_g1, mode='1d')
Pkg1 = r.power['power'].real - r.power.attrs['shotnoise']
r = FFTPower(delta_gRSD1, mode='1d')
PkgRSD1 = r.power['power'].real - r.power.attrs['shotnoise']

galaxy2 = BigFileCatalog('Galaxypaired2_catalog.bigfile')
delta_g2 = galaxy2.to_mesh(resampler='cic', interlaced=True, compensated=True)
delta_gRSD2 = galaxy1.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_g2, mode='1d')
Pkg2 = r.power['power'].real - r.power.attrs['shotnoise']
r = FFTPower(delta_gRSD2, mode='1d')
PkgRSD2 = r.power['power'].real - r.power.attrs['shotnoise']

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.loglog(k_array, Pklin, label='Initial Gaussian realization')
ax.loglog(k_array, Pk1, '--', label='Matter, sim 1')
ax.loglog(k_array, Pk2, ':', label='Matter, sim 2')
ax.loglog(k_array, Pkg1, '--', label='Galaxy, sim 1')
ax.loglog(k_array, Pkg2, ':', label='Galaxy, sim 2')
ax.legend(fontsize='16')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$', fontsize=18)
ax.set_ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$', fontsize=18)
plt.tight_layout()
plt.savefig('Pks_psimulation.pdf')

# Compare densifty field in physical and redshift space
fig, ax = plt.subplots(2, 3, figsize=(18, 12))

ax[0,0].loglog(k_array, Pklin, label='ICs, sim 1')
ax[0,0].loglog(k_array, Pkg1, label='Real Space, sim 1')
ax[0,0].loglog(k_array, PkgRSD1, label='Redshift Space, sim 1')
ax[0,0].legend(fontsize='16')
ax[0,0].set_xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$', fontsize='18')
ax[0,0].set_ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$', fontsize='18')
ax[0,0].set(ylim=(1e2, 2e5))
ax[0,0].tick_params(axis='both', which='major', labelsize=14)

ax[0,1].imshow(delta_g1.paint(mode='real').preview(axes=[0,1]))
ax[0,1].set_xlabel('Simulation 1, Real Space', fontsize='18')#r'$1 + \delta_{g}$')
ax[0,1].tick_params(axis='both', which='major', labelsize=14)

ax[0,2].imshow(delta_gRSD1.paint(mode='real').preview(axes=[0,1]))
ax[0,2].set_xlabel('Simulation 1, Redshift Space', fontsize='18')#r'$1 + \delta_{g}^{s}$')
ax[0,2].tick_params(axis='both', which='major', labelsize=14)

ax[1,0].loglog(k_array, Pklin, label='ICs, sim 2')
ax[1,0].loglog(k_array, Pkg2, label='Real Space, sim 2')
ax[1,0].loglog(k_array, PkgRSD2, label='Redshift Space, sim 2')
ax[1,0].legend(fontsize='16')
ax[1,0].set_xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$', fontsize='18')
ax[1,0].set_ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$', fontsize='18')
ax[1,0].set(ylim=(1e2, 2e5))
ax[1,0].tick_params(axis='both', which='major', labelsize=14)

ax[1,1].imshow(delta_g2.paint(mode='real').preview(axes=[0,1]))
ax[1,1].set_xlabel('Simulation 2, Real Space', fontsize='18')#r'$1 + \delta_{g}$')
ax[1,1].tick_params(axis='both', which='major', labelsize=14)

ax[1,2].imshow(delta_gRSD2.paint(mode='real').preview(axes=[0,1]))
ax[1,2].set_xlabel('Simulation 2, Redshift Space', fontsize='18')#r'$1 + \delta_{g}^{s}$')
ax[1,2].tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('realredshift_psimulations.pdf')