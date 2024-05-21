import matplotlib.pyplot as plt
from nbodykit.lab import *

# Generate IC density field
linear = BigFileMesh('../paired_simulations/Initialrealization.bigfile', dataset='Field')
# P(k) of initial field
r = FFTPower(linear, mode="1d")
Pkdelta = r.power['power'].real
k_array = r.power['k']

matter1 = BigFileCatalog('../paired_simulations/Matterpaired1_catalog.bigfile')
matterfield1 = matter1.to_mesh(resampler='cic', interlaced=True, compensated=True)
r = FFTPower(matterfield1, mode='1d')
Pk1 = r.power['power'].real - r.power.attrs['shotnoise']

matter2 = BigFileCatalog('../paired_simulations/Matterpaired2_catalog.bigfile')
matterfield2 = matter2.to_mesh(resampler='cic', interlaced=True, compensated=True)
r = FFTPower(matterfield2, mode='1d')
Pk2 = r.power['power'].real - r.power.attrs['shotnoise']

################################
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(matterfield1.paint(mode='real').preview(axes=[0,1]))
ax[0].set_title('Simulation 1, Matter')
ax[1].imshow(matterfield2.paint(mode='real').preview(axes=[0,1]))
ax[1].set_title('Simulation 2, Matter')

plt.savefig('paired_matter.pdf')
################################
fig = plt.figure()
plt.loglog(k_array, Pkdelta, label='Initial Gaussian realization')
plt.loglog(k_array, Pk1, '--', label='Matter, sim 1')
plt.loglog(k_array, Pk2, ':', label='Matter, sim 2')
plt.legend()
plt.xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$')
plt.ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$')
plt.savefig('Pkmatter_paired.pdf')
################################

halos1 = BigFileCatalog('../paired_simulations/Galaxypaired1.bigfile')
delta_halos1 = halos1.to_mesh(resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_halos1, mode='1d')
Pkhalos1 = r.power['power'].real - r.power.attrs['shotnoise']
delta_hRSD1 = halos1.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_hRSD1, mode='1d')
PkhRSD1 = r.power['power'].real - r.power.attrs['shotnoise']

halos2 = BigFileCatalog('../paired_simulations/Galaxypaired2.bigfile')
delta_halos2 = halos1.to_mesh(resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_halos2, mode='1d')
Pkhalos2 = r.power['power'].real - r.power.attrs['shotnoise']
delta_hRSD2 = halos2.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
r = FFTPower(delta_hRSD2, mode='1d')
PkhRSD2 = r.power['power'].real - r.power.attrs['shotnoise']

################################
fig = plt.figure()
plt.plot(k_array, k_array**1.5 * Pkhalos1, label='Tracers, sim 1')
plt.plot(k_array, k_array**1.5 * Pkhalos2, ':', label='Tracers, sim 2')
plt.legend()
plt.xscale('log')
plt.xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$')
plt.ylabel(r'$k^{1.5} P(k)$ $[h^{-1/5} \mathrm{Mpc}^{1.5}]$')
plt.savefig('Pkhalos_pairedsim.pdf')
################################

################################
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(delta_halos1.paint(mode='real').preview(axes=[0,1]))
ax[0].set_title('Simulation 1, Halos')
ax[1].imshow(delta_halos2.paint(mode='real').preview(axes=[0,1]))
ax[1].set_title('Simulation 2, Halos')
plt.savefig('paired_tracers.pdf')
################################

################################
# Compare densifty field in physical and redshift space
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(delta_hRSD1.paint(mode='real').preview(axes=[0,1]))
ax[0].set_title('Simulation 1, Redshift space')#r'$1 + \delta_{g}$')
ax[1].imshow(delta_hRSD2.paint(mode='real').preview(axes=[0,1]))
ax[1].set_title('Simulation 2, Redshift space')#r'$1 + \delta_{g}^{s}$')
plt.savefig('paired_tracers_redshift.pdf')
################################