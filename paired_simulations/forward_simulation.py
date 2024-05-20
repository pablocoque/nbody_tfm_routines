from nbodykit.lab import *
import numpy as np
import sys

sys.path.append('../general_routines/')
from general_tools import *
from catalogue_generation import *

### MAIN

paired = sys.argv[1]

# Setup initial conditions
cosmo = cosmology.Planck15
L_mesh = BigFileMesh('paired'+str(paired)+'_Lmesh.bigfile', dataset='Field')
NL_mesh = BigFileMesh('paired'+str(paired)+'_NLmesh.bigfile', dataset='Field')

Nc = L_mesh.attrs['Nmesh'][0]
Length = L_mesh.attrs['BoxSize'][0] # Mpc  h-1
zobs = 0.3

Nlinear = NL_field(L_mesh, NL_mesh)
Nlinear = FieldMesh(Nlinear)

forward_displf = compute_Psi(Length, Nc, Nlinear.compute(mode='real') - 1.)
matter_pos = forward_evolution(Length, Nc, forward_displf)
ww = np.ones(len(matter_pos))
with open('matter_file.dat', 'wb') as ff:
    matter_pos.tofile(ff); ww.tofile(ff); ff.seek(0)

matter_cat = BinaryCatalog(ff.name, [('Position', ('f8', 3)), ('Mass', ('f8', 1))], size=len(matter_pos))
matter_cat.attrs['BoxSize'] = np.array([Length, Length, Length])
matter_cat.attrs['Nmesh'] = np.array([Nc, Nc, Nc])
delta_dm = matter_cat.to_mesh(resampler='cic', interlaced=True, compensated=True)

alpha = 1.3
delta_th = 0.
gamma = 0.02

galaxy_pos = make_catalog_g(delta_dm.compute(mode='real') - 1., alpha, gamma, delta_th, Length, Nc)

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

bg = evaluate_bias(galaxy_cat, matter_cat)
print('Galaxy bias = {:.2f}'.format(bg))

peculiar_field = compute_Psi(Length, Nc, (delta_dm.compute(mode='real') - 1.))
observer = np.array([Length/2,Length/2,Length/2])
vr = compute_vr(field_interpolation(Length, Nc, peculiar_field, galaxy_pos), galaxy_pos, observer, zobs)
galaxy_posRSD = boxfit_conditions(galaxy_pos + vr, Length)

galaxy_cat['PositionRSD'] = galaxy_posRSD


matter_cat.save('Matterpaired'+str(paired)+'_catalog.bigfile')
galaxy_cat.save('Galaxypaired'+str(paired)+'_catalog.bigfile')