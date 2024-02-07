from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt

def D(z):
    return cosmology.background.MatterDominated(cosmology.Planck15.Omega0_m).D1(1/(1+z))

### Import DM catalog
matter1 = BigFileCatalog('Matter_paired1.bigfile')
matter2 = BigFileCatalog('Matter_paired2.bigfile')

# Define global variables
Length = matter1.attrs['BoxSize'][0]
Nc = matter1.attrs['Nmesh'][0]
zobs = 0.3
zinit = 3.
r_s = 5. # smoothing radius

# Compute matter density field
delta_dm1 = matter1.to_mesh(resampler='cic', interlaced=True, compensated=True)
delta_dm2 = matter2.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k)
r = FFTPower(delta_dm1, mode="1d", Nmesh=Nc)
# shot-noise from nbodykit
Pkdm1 = r.power['power'].real - r.power.attrs['shotnoise']
k = r.power['k']
r = FFTPower(delta_dm2, mode="1d", Nmesh=Nc)
# shot-noise from nbodykit
Pkdm2 = r.power['power'].real - r.power.attrs['shotnoise']

### Import Halos catalog
halos1 = BigFileCatalog('Halos_paired1_reconvr.bigfile')
halos2 = BigFileCatalog('Halos_paired2_reconvr.bigfile')

# Compute galaxy density field
delta_h1 = halos1.to_mesh(position='PositionQ', resampler='cic', interlaced=True, compensated=True)
delta_h2 = halos2.to_mesh(position='PositionQ', resampler='cic', interlaced=True, compensated=True)
# delta_hRSD = halos.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)

r = FFTPower(delta_h1, mode="1d", Nmesh=Nc)
Pkh1 = r.power['power'].real - r.attrs['shotnoise']
r = FFTPower(delta_h2, mode="1d", Nmesh=Nc)
Pkh2 = r.power['power'].real - r.attrs['shotnoise']

# r = FFTPower(delta_hRSD, mode="1d", Nmesh=Nc)
# PkhRSD = r.power['power'].real - r.attrs['shotnoise']

mask = (k <= 0.09)*(k > 0.03)

bgzobs1 = np.sqrt(np.mean(Pkh1[mask]/Pkdm1[mask]))
bgzrec1 = np.sqrt(np.mean(Pkh1[mask]*(D(zobs)**2)/(Pkdm1[mask]*(D(zinit)**2))))
bgzobs2 = np.sqrt(np.mean(Pkh2[mask]/Pkdm2[mask]))
bgzrec2 = np.sqrt(np.mean(Pkh2[mask]*(D(zobs)**2)/(Pkdm2[mask]*(D(zinit)**2))))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(k, np.sqrt(Pkh1/Pkdm1), 'b')#, label='z=0.3, sim 1')
ax.plot(k, np.sqrt(Pkh1*(D(zobs)**2)/(Pkdm1*(D(zinit)**2))), 'c')#, label='z=3.0, sim 1')
ax.plot(k, np.sqrt(Pkh2/Pkdm2), 'r')#, label='z=0.3, sim 2')
ax.plot(k, np.sqrt(Pkh2*(D(zobs)**2)/(Pkdm2*(D(zinit)**2))), 'm')#, label='z=3.0, sim 2')
ax.hlines(bgzobs1, k.min(), k.max(), 'b', linestyles='dotted', label=r'$b^{{\mathrm{{rec,}}z={:.1f}}}_{{1, \mathrm{{sim 1}}}} = {:.2f}$'.format(zobs, bgzobs1))
ax.hlines(bgzrec1, k.min(), k.max(), 'c', linestyles='dotted', label=r'$b^{{\mathrm{{rec,}}z={:.1f}}}_{{1, \mathrm{{sim 1}}}} = {:.2f}$'.format(zobs, bgzrec1))
ax.hlines(bgzobs2, k.min(), k.max(), 'r', linestyles='dotted', label=r'$b^{{\mathrm{{rec,}}z={:.1f}}}_{{1, \mathrm{{sim 2}}}} = {:.2f}$'.format(zinit, bgzobs2))
ax.hlines(bgzrec2, k.min(), k.max(), 'm', linestyles='dotted', label=r'$b^{{\mathrm{{rec,}}z={:.1f}}}_{{1, \mathrm{{sim 2}}}} = {:.2f}$'.format(zinit, bgzrec2))
ax.set(xscale='log', ylim=(0, 6.5))
ax.legend(fontsize=16)
ax.set_xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$', fontsize=18)
ax.set_ylabel(r'$\sqrt{P^{\mathrm{rec}}_h(k,z)/P_m(k,z)}$', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)

# plt.show()
plt.savefig('bias_plot_rec.pdf')