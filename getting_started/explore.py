import fastpm
from fastpm.nbkit import FastPMCatalogSource
from nbodykit.lab import *
from nbodykit import setup_logging, style
import matplotlib.pyplot as plt
import numpy as np

Nc=256
Length=1000 # 1 Gpc

#Set up IC
print("Computing P(k) linear with Planck15 cosmology")
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, 0)
linear = LinearMesh(Plin, BoxSize=Length, Nmesh=Nc,seed=42)
r = FFTPower(linear, mode="1d", Nmesh=Nc)
k = r.power['k']
Pk = r.power['power'].real

sim = FastPMCatalogSource(linear, Nsteps=10)
mesh = sim.to_mesh(resampler='tsc')
one_plus_delta = mesh.paint(mode='real')
r = FFTPower(sim, mode="1d", Nmesh=Nc)
Pk1 = r.power
Pk_dm = Pk1['power'].real-Pk1.attrs['shotnoise']

sim_dm = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=Length, Nmesh=Nc, bias=1.0, seed=42)
mesh = sim_dm.to_mesh(resampler='tsc')
one_plus_delta_dm = mesh.paint(mode='real')
r = FFTPower(sim_dm, mode="1d", Nmesh=Nc)
Pk2 = r.power
Pk_dm1 = Pk2['power'].real-Pk2.attrs['shotnoise']

sim_gal = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=Length, Nmesh=Nc, bias=2.0, seed=42)
mesh = sim_gal.to_mesh(resampler='tsc')
one_plus_delta_gal= mesh.paint(mode='real')
r = FFTPower(sim_gal, mode="1d", Nmesh=Nc)
Pk3 = r.power
Pk_gal = Pk3['power'].real-Pk3.attrs['shotnoise']

print('Calculating bias')
mask = (k <= 0.15)*(k > 0.03)
bg = np.sqrt(np.mean(Pk_gal[mask]/Pk_dm[mask]))
print('bias = {:.2f}'.format(bg))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(one_plus_delta.preview(axes=[0,1]))
ax[1].imshow(one_plus_delta_dm.preview(axes=[0,1]))
ax[2].imshow(one_plus_delta_gal.preview(axes=[0,1]))
plt.show()

plt.loglog(k, Pk, label = 'IC')
plt.loglog(Pk1['k'], Pk_dm, label = 'DM ref')
plt.loglog(Pk2['k'], Pk_dm1, label = 'DM log-normal')
plt.loglog(Pk3['k'], Pk_gal, label = 'Gal log-normal')
plt.legend()
plt.show()