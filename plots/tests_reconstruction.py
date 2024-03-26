from nbodykit.lab import *
import numpy as np
import dask.array as d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

######### MAIN

cosmo = cosmology.Planck15

### Import DM catalog
matter = BigFileCatalog('fastpm_dmcatalog.bigfile')

# Define global variables
Length = matter.attrs['BoxSize'][0]
Nc = matter.attrs['Nmesh'][0]
zobs = matter.attrs['plin.redshift']
zinit = 3.

# Compute matter density field
delta_dm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k,z)
r = FFTPower(delta_dm, mode="1d", Nmesh=Nc)
# shot-noise from nbodykit
Pkdm = r.power['power'].real - r.power.attrs['shotnoise']

### Import Galaxy catalog
hod = BigFileCatalog('hod_gcatalog.bigfile')

# Compute galaxy density field
delta_g = hod.to_mesh(resampler='cic', interlaced=True, compensated=True)
ngal = hod.csize

# Compute galaxy P(k,z)
r = FFTPower(delta_g, mode="1d", Nmesh=Nc)
k = r.power['k']
# shot-noise from nbodykit
Pkg = r.power['power'].real - r.attrs['shotnoise']

## RSD formula
# vr = 1/(Ha) * (vec(v).vec(r_unit))vec(r_unit)

# Compute redshift space distortions and shift galaxies
## Box centered
position_origin = hod['Position'] - 0.5*Length

projection_norm = np.linalg.norm(position_origin, axis=1)

line_of_sight = np.zeros_like(position_origin)
line_of_sight = position_origin/projection_norm[:, np.newaxis]

rsd_factor = (1+zobs) / (100 * cosmo.efunc(zobs))

dot_prod = np.sum(hod['Velocity']*line_of_sight, axis=1)

hod['PositionRSD'] = position_origin + rsd_factor*dot_prod[:, np.newaxis]*line_of_sight + 0.5*Length

# Compute density field at redshift space
delta_gRSD = hod.to_mesh(resampler='cic', position='PositionRSD', interlaced=True, compensated=True)
r = FFTPower(delta_gRSD, mode='1d', Nmesh=Nc)
PkgRSD = r.power['power'].real - r.attrs['shotnoise']

# Compare Pk's in real and redshift space
# Visualize them
plt.loglog(k, Pkg, label='Real space')
plt.loglog(k, PkgRSD, label='Redshift space')
plt.legend()
plt.xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$')
plt.ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$')
# plt.ylim(3e3, 2e5)
plt.tight_layout()
plt.savefig('Pk_real_n_redshift.pdf')
print('----------- P(k) figure saved')

# Obtain bias
mask = (k <= 0.04)*(k > 0.02)
bg = np.sqrt(np.mean(Pkg[mask]/Pkdm[mask]))
print('bias = {:.2f}'.format(bg))

# Compare densifty field in physical and redshift space
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

ax[0].imshow(delta_g.paint(mode='real').preview(axes=[0,1]))
ax[0].set_title('Real space')#r'$1 + \delta_{g}$')

ax[1].imshow(delta_gRSD.paint(mode='real').preview(axes=[0,1]))
ax[1].set_title('Redshift space')#r'$1 + \delta_{g}^{s}$')

plt.savefig('box_projection_pre.pdf')
print('----------- Box figure saved')