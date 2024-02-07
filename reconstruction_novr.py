import sys
from nbodykit.lab import *
import numpy as np
import dask.array as da
# import matplotlib.pyplot as plt

# FUNCTIONS
def divide_bias(x, v):
    return v/bg

def fft(inarr):
    finarr= np.fft.fftn(inarr)
    return(finarr)

def ifft(finarr):
    inarr= np.fft.ifftn(finarr)
    return(inarr)

def compute_zeld(L, nc, delta):
    '''
    Obtain displacement field under Zeldovich approximation
    '''
    deltaf = fft(delta)

    vfx = deltaf.copy()
    vfy = deltaf.copy()
    vfz = deltaf.copy()
    
    kfac = 2.0*np.pi/L
    
    for i in range(nc):
        for j in range(nc):
            for k in range(nc):

                if i <= nc/2:
                    kx = kfac*np.float64(i)
                else:
                    kx = -kfac*np.float64(nc-i)
                if j <= nc/2:
                    ky = kfac*np.float64(j)
                else:
                    ky = -kfac*np.float64(nc-j)
                if k <= nc/2:
                    kz = kfac*np.float64(k)
                else:
                    kz = -kfac*np.float64(nc-k)

                k2 = kx**2 + ky**2 + kz**2

                kernelx = 0.
                kernely = 0.
                kernelz = 0.


    # kernel: -nabla/nabla2 = i*kvec/k2 viene de: 
    # d(exp(i*kvec*r))/dr=i*kvec  , d(exp(i*kvec*r))2/dr2=(i*kvec)*(i*kvec)=-k2 

                epsk = 1e-14
                if k2>epsk:
                    kernelx = kx/k2
                    kernely = ky/k2
                    kernelz = kz/k2
                
                vfx.real[i,j,k] = -kernelx*deltaf.imag[i,j,k]
                vfx.imag[i,j,k] =  kernelx*deltaf.real[i,j,k]

                vfy.real[i,j,k] = -kernely*deltaf.imag[i,j,k]
                vfy.imag[i,j,k] =  kernely*deltaf.real[i,j,k]

                vfz.real[i,j,k] = -kernelz*deltaf.imag[i,j,k]
                vfz.imag[i,j,k] =  kernelz*deltaf.real[i,j,k]

    vxg = ifft(vfx)
    vx = vxg.real
    
    vyg = ifft(vfy)
    vy = vyg.real

    vzg = ifft(vfz)
    vz = vzg.real

    vel1D = np.zeros(nc**3 * 3)
    psi   = vel1D.reshape(nc,nc,nc,3)

    psi[:,:,:,0] = vx
    psi[:,:,:,1] = vy
    psi[:,:,:,2] = vz
    return psi

def D(z):
    '''
    Obtain growth factor
    '''
    return cosmology.background.MatterDominated(cosmology.Planck15.Omega0_m).D1(1/(1+z))

def f(z):
    '''
    Obtain growth rate
    '''
    return cosmology.background.MatterDominated(cosmology.Planck15.Omega0_m).f1(1/(1+z))

def field_interpolation(L, nc, psi, s):
    # tamano celda
    cellsize = L/nc
    # interpolamos el campo de velocidad en la malla a la posición de la galaxia
    psi_g = np.zeros_like(s)

    for i in range(len(psi_g)):
        j = int(np.floor(s[i, 0]/cellsize))
        k = int(np.floor(s[i, 1]/cellsize))
        l = int(np.floor(s[i, 2]/cellsize))

        j = int(np.fmod(j,nc))
        k = int(np.fmod(k,nc))
        l = int(np.fmod(l,nc))

        psi_g[i, 0] = psi[j, k, l, 0]
        psi_g[i, 1] = psi[j, k, l, 1]
        psi_g[i, 2] = psi[j, k, l, 2]

    return psi_g

def iteration_s(Niter, L, nc, zobs, zinit, cat):#, k, Pkcomp):
    s = cat['PositionRSD']
    cat['PositionSp'] = cat['PositionRSD']
    Sp = cat['PositionSp']
    delta = cat.to_mesh(resampler='cic', position='PositionSp', interlaced=True, compensated=True)
    r = FFTPower(delta, mode='1d')
    Pk1 = r.power['power'].real - r.attrs['shotnoise']
    # k1 = r.power['k']
    for i in range(Niter):
        # plt.loglog(k, Pkcomp, 'k', label=r'$P_{\mathrm{lin}}(z=0.3)$')

        psi = compute_zeld(L, nc, delta.apply(divide_bias, mode='real', kind='index')\
            .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber').paint(mode='real'))
        psi_g = field_interpolation(L, nc, psi, Sp.compute())
        
        # sp = s - psi(s)
        cat['PositionSp'] = s - (D(zobs) - D(zinit))*psi_g
        Sp = cat['PositionSp']
        delta = cat.to_mesh(resampler='cic', position='PositionSp', interlaced=True, compensated=True)
        
        r = FFTPower(delta, mode='1d')
        Pk2 = r.power['power'].real - r.attrs['shotnoise']
        # k2 = r.power['k']
        # plt.loglog(k1, Pk1, label='Iteration {:1n}'.format(i))
        # plt.loglog(k2, Pk2, label='Iteration {:1n}'.format(i+1))
        # plt.legend()
        # plt.xlabel(r'$k\ [\mathrm{h Mpc}^{-1}]$')
        # plt.ylabel(r'$P(k, z)$ $[h^{-3} \mathrm{Mpc}^3]$')
        # plt.show()
        
        print('Iteration {:2d}: Mean difference between Pks: {:.2f}'.format(i+1, np.mean(abs(Pk1 - Pk2))))
        
        Pk1 = Pk2
    return delta

######### MAIN

paired = sys.argv[1]

cosmo = cosmology.Planck15

### Import DM catalog
matter = BigFileCatalog('Matter_paired'+str(paired)+'.bigfile')

# Define global variables
Length = matter.attrs['BoxSize'][0]
Nc = matter.attrs['Nmesh'][0]
zobs = 0.3
zinit = 3.
r_s = 10. # smoothing radius

# Compute matter density field
delta_dm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k,z)
r = FFTPower(delta_dm, mode="1d", Nmesh=Nc)
# shot-noise from nbodykit
Pkdm = r.power['power'].real - r.power.attrs['shotnoise']
k = r.power['k']

### Import Halos catalog
halos = BigFileCatalog('Halos_paired'+str(paired)+'.bigfile')

# Compute galaxy density field
delta_h = halos.to_mesh(resampler='cic', interlaced=True, compensated=True)
delta_hRSD = halos.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
ngal = halos.csize

# Compute galaxy P(k,z)
r = FFTPower(delta_h, mode="1d", Nmesh=Nc)
# shot-noise from nbodykit
Pkh = r.power['power'].real - r.attrs['shotnoise']

# Obtain bias
mask = (k <= 0.04)*(k > 0.02)
bg = np.sqrt(np.mean(Pkh[mask]/Pkdm[mask]))

# Compute celestial coordinates pre-reconstruction
# halos['SkyCoordz'] = transform.CartesianToSky(pos=halos['Position'], cosmo=cosmo, \
#     observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
# halos['SkyCoordzpec'] = transform.CartesianToSky(pos=halos['Position'], cosmo=cosmo, velocity=halos['Velocity'], \
#     observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# # Save catalog
# coord_array = np.zeros((ngal, 4))
# coord_array[:, 0:3] = halos['SkyCoordz'].compute()
# coord_array[:, 3] = halos['SkyCoordzpec'][:,2].compute()
# mat = np.matrix(coord_array)
# with open('pairedsim'+str(paired)+'_cat_pre_z0.3.dat', 'wb') as ff:
#     for line in mat:
#         np.savetxt(ff, line, fmt='%.5f')

reconstructed = iteration_s(5, Length, Nc, zobs, zinit, halos)

r = FFTPower(reconstructed, mode="1d", Nmesh=Nc)
Pkhrecon = r.power['power'].real - r.attrs['shotnoise']

bgrecon = np.mean(np.sqrt(Pkhrecon[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))

bgrecon2 = np.mean(np.sqrt(Pkhrecon[mask]/Pkdm[mask]))

print('bias halos POST= {:.2f},{:.2f}'.format(bgrecon,bgrecon2))

# fig = plt.figure()
# plt.loglog(k, Pkh, label='Pre-reconstruction, z=0.3')
# plt.loglog(k, Pkhrecon, label='Post-reconstruction, z=3')
# plt.legend()
# plt.xlabel(r'$k$ $[h \mathrm{Mpc}^{-1}]$')
# plt.ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^{3}]$')
# plt.savefig('Pkhalosrecon_pairedsim'+str(paired)+'.pdf')

# fig, ax = plt.subplots(1, 2, figsize=(10,5))
# ax[0].imshow(delta_hRSD.paint(mode='real').preview(axes=[0,1]))
# ax[0].set_title('Pre-reconstruction')
# ax[1].imshow(reconstructed.paint(mode='real').preview(axes=[0,1]))
# ax[1].set_title('Post-reconstruction')
# plt.savefig('reconstructed_pairedsim'+str(paired)+'.pdf')

# Compute celestial coordinates post-reconstruction
halos['SkyCoordz'] = transform.CartesianToSky(pos=halos['PositionSp'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
halos['SkyCoordzpec'] = transform.CartesianToSky(pos=halos['PositionSp'], cosmo=cosmo, velocity=halos['Velocity'], \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = halos['SkyCoordz'].compute()
coord_array[:, 3] = halos['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('pairedsim'+str(paired)+'_cat_post_z3_novr.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.5f')

halos.save('Halos_paired'+str(paired)+'_novr.bigfile')