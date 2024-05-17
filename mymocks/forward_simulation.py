from nbodykit.lab import *
import numpy as np
import sys

sys.path.append('../general_routines/')
from general_tools import *
from catalogue_generation import *

### MAIN

# Setup initial conditions
cosmo = cosmology.Planck15
L_mesh = BigFileMesh('Lmesh.bigfile', dataset='Field')
NL_mesh = BigFileMesh('NLmesh.bigfile', dataset='Field')

Nc = L_mesh.attrs['Nmesh'][0]
Length = L_mesh.attrs['BoxSize'][0] # Mpc  h-1
zobs = 0.3

deltaICNL = convolve_NL(Length, Nc, (L_mesh.paint(mode='real') - 1.), (NL_mesh.paint(mode='real') - 1.))

Nlinear = ArrayMesh(deltaICNL + 1., Length)

forward_displf = compute_Psi(Length, Nc, deltaICNL)
matter_pos = forward_evolution(Length, Nc, forward_displf)
ww = np.ones(len(matter_pos))
with open('matter_file.dat', 'wb') as ff:
    matter_pos.tofile(ff); ww.tofile(ff); ff.seek(0)

matter_cat = BinaryCatalog(ff.name, [('Position', ('f8', 3)), ('Mass', ('f8', 1))], size=len(matter_pos))
matter_cat.attrs['BoxSize'] = np.array([Length, Length, Length])
matter_cat.attrs['Nmesh'] = np.array([Nc, Nc, Nc])
delta_dm = matter_cat.to_mesh(resampler='cic', interlaced=True, compensated=True)

r = FFTPower(delta_dm, mode='1d')
Pkdm = r.power['power'].real - r.attrs['shotnoise']
k = r.power['k']

alpha = 1.3
delta_th = 0.
gamma = 0.02
# n0 = 3.6e-4
# gamma = n0/np.mean((delta_dm.paint(mode='real'))**alpha)

galaxy_pos = make_catalog_g((delta_dm.paint(mode='real') - 1.), alpha, gamma, delta_th, Length, Nc)

Ng = len(galaxy_pos)
n = Ng/Length**3
print('Total of galaxies = ', Ng)
print('Galaxy number density = ', n)

ww = np.ones(len(galaxy_pos))
with open('galaxy_file.dat', 'wb') as ff:
    galaxy_pos.tofile(ff); ww.tofile(ff); ff.seek(0)

galaxy_cat = BinaryCatalog(ff.name, [('Position', ('f8', 3)), ('Mass', ('f8', 1))], size=len(galaxy_pos))
galaxy_cat.attrs['BoxSize'] = np.array([Length, Length, Length])
galaxy_cat.attrs['Nmesh'] = np.array([Nc, Nc, Nc])
delta_g = galaxy_cat.to_mesh(resampler='cic', interlaced=True, compensated=True)

r = FFTPower(delta_g, mode='1d')
Pkg = r.power['power'].real - r.attrs['shotnoise']

mask = (k>0.03)*(k<=0.09)
bg = np.mean(np.sqrt(Pkg[mask]/Pkdm[mask]))
print('Galaxy bias = {:.2f}'.format(bg))

peculiar_field = compute_Psi(Length, Nc, (delta_dm.paint(mode='real') - 1.))
observer = np.array([Length/2,Length/2,Length/2])
vr = compute_vr(field_interpolation(Length, Nc, peculiar_field, galaxy_pos), galaxy_pos, observer, zobs)
galaxy_posRSD = galaxy_pos + vr

galaxy_cat['PositionRSD'] = galaxy_posRSD
delta_gRSD = galaxy_cat.to_mesh(position='PositionRSD', resampler='cic', compensated=True, interlaced=True)

matter_cat.save('Matter_catalog.bigfile')
galaxy_cat.save('Galaxy_catalog.bigfile')