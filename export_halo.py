from nbodykit.lab import *
import numpy as np

halos = BigFileCatalog('halo_catalog.bigfile')

halo_dat = np.zeros((halos.csize, 7))
halo_dat[:,:3] = halos['Position'].compute()
halo_dat[:,3:6] = halos['Velocity'].compute()
halo_dat[:,6] = halos['Mass'].compute()

mat = np.matrix(halo_dat)
with open('halo_cat_pre_z03.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.7e')