import sys
from nbodykit.lab import *
import numpy as np
import dask.array as da
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# FUNCTIONS
def divide_bias(x, v):
    return v/bg

def correct_delta(x, v):
    return v-1.

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

def field_interpolation(L, nc, psi, pos):
    
    psi_g = np.zeros_like(pos)

    x = np.linspace(0., L, nc)
    y = np.linspace(0., L, nc)
    z = np.linspace(0., L, nc)

    psix = psi[:,:,:,0]
    psiy = psi[:,:,:,1]
    psiz = psi[:,:,:,2]

    fnx = RegularGridInterpolator((x, y, z), psix, method='linear')
    fny = RegularGridInterpolator((x, y, z), psiy, method='linear')
    fnz = RegularGridInterpolator((x, y, z), psiz, method='linear')

    psi_g[:,0] = fnx(pos)
    psi_g[:,1] = fny(pos)
    psi_g[:,2] = fnz(pos)
    # # tamano celda
    # # interpolamos el campo de velocidad en la malla a la posición de la galaxia

    # for i in range(len(psi_g)):
    #     j = int(np.floor(s[i, 0]/cellsize))
    #     k = int(np.floor(s[i, 1]/cellsize))
    #     l = int(np.floor(s[i, 2]/cellsize))

    #     j = int(np.fmod(j,nc))
    #     k = int(np.fmod(k,nc))
    #     l = int(np.fmod(l,nc))

    #     psi_g[i, 0] = psi[j, k, l, 0]
    #     psi_g[i, 1] = psi[j, k, l, 1]
    #     psi_g[i, 2] = psi[j, k, l, 2]

    return psi_g

def compute_vr(vel, q, observer, z):
    # Omega_mz = cosmology.Planck15.Omega_m(zobs)
    # f = Omega_mz**0.6
    position_origin = q - observer
    projection_norm = np.linalg.norm(position_origin, axis=1)
    line_of_sight = np.zeros_like(position_origin)
    line_of_sight = position_origin/projection_norm[:, np.newaxis]
    dot_prod = np.sum(vel*line_of_sight, axis=1)
    return f(z)*dot_prod[:, np.newaxis]*line_of_sight

def periodic_conditions(coord):
    maskx1 = coord[:,0]<0.
    maskx2 = coord[:,0]>=Length
    coord[maskx1,0] += Length
    coord[maskx2,0] -= Length
    masky1 = coord[:,1]<0.
    masky2 = coord[:,1]>=Length
    coord[masky1,1] += Length
    coord[masky2,1] -= Length
    maskz1 = coord[:,2]<0.
    maskz2 = coord[:,2]>=Length
    coord[maskz1,2] += Length
    coord[maskz2,2] -= Length
    return coord

def iteration(Niter, L, nc, zobs, zinit, cat, observer, k, Pkcomp, deltadm):
    cat['PositionQ'] = cat['Position']
    s = cat['Position'].compute()
    delta = cat.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)
    r = FFTPower(delta, mode='1d')
    Pk1 = r.power['power'].real - r.attrs['shotnoise']
    # k1 = r.power['k']
    for i in range(Niter):
        # fig, ax = plt.subplots(1, 1, figsize=(8,6))
        # ax.loglog(k, Pkcomp, 'k', label='PRE reconstruction')

        q = cat['PositionQ'].compute()
        psi = compute_zeld(L, nc, deltadm.apply(correct_delta, mode='real', kind='index')\
            .apply(divide_bias, mode='real', kind='index')\
            .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber').compute(mode='real'))
        psi_g = field_interpolation(L, nc, psi, q)
        #vr = compute_vr(psi_g, q, observer, zobs)
        
        cat['PositionQ'] = periodic_conditions(s - (D(zobs)-D(zinit))*psi_g)# - vr) # Iteration 1st step: q' = s - Psi(q) - vr(q)

        delta = cat.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)

        r = FFTPower(delta, mode='1d')
        Pk2 = r.power['power'].real - r.attrs['shotnoise']
        # k2 = r.power['k']
        # ax.loglog(k1, Pk1, label='Iteration {:1n}'.format(i))
        # ax.loglog(k2, Pk2, ':', label='Iteration {:1n}'.format(i+1))
        # ax.legend()
        # ax.set_xlabel(r'$k [\mathrm{h Mpc}^{-1}]$', fontsize=18)
        # ax.set_ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$', fontsize=18)
        # ax.tick_params(axis='both', which='major', labelsize=14)
        # plt.tight_layout()
        # plt.savefig('Pkcomparison_iter{:1d}_paired'.format(i+1)+str(paired)+'.pdf')
        
        print('Iteration {:1d}, Mean difference between Pks: {:.2f}'.format(i+1, np.mean(abs(Pk1 - Pk2))))

        Pk1 = Pk2
    
    q = cat['PositionQ'].compute()
    psi = compute_zeld(L, nc, deltadm.apply(correct_delta, mode='real', kind='index')\
        .apply(divide_bias, mode='real', kind='index')\
        .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber').compute(mode='real'))
    psi_g = field_interpolation(L, nc, psi, q)
    cat['PositionQS'] = periodic_conditions(s-(D(zobs)-D(zinit))*psi_g)
    deltaQS = cat.to_mesh(resampler='cic', position='PositionQS', interlaced=True, compensated=True)
    return delta, deltaQS

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
r_s = 2*(Length/Nc) # smoothing radius
print('Smoothing radius=', r_s)

# Compute matter density field
delta_dm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k,z)
r = FFTPower(delta_dm, mode="1d")
# shot-noise from nbodykit
Pkdm = r.power['power'].real - r.power.attrs['shotnoise']
k = r.power['k']

### Import Halos catalog
halos = BigFileCatalog('Halos_paired'+str(paired)+'_rsdfixed.bigfile')

# Compute galaxy density field
delta_h = halos.to_mesh(resampler='cic', interlaced=True, compensated=True)
delta_hRSD = halos.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
nhalos = halos.csize

# Compute galaxy P(k,z)
r = FFTPower(delta_h, mode="1d")
Pkh = r.power['power'].real - r.attrs['shotnoise']

r = FFTPower(delta_hRSD, mode="1d")
PkhRSD = r.power['power'].real - r.attrs['shotnoise']

# Calculate bias pre reconstruction
mask = (k <= 0.09)*(k > 0.03)
bg = np.mean(np.sqrt(Pkh[mask]/Pkdm[mask]))
bgRSD = np.mean(np.sqrt(PkhRSD[mask]/Pkdm[mask]))

print('bias halos (real) PRE= {:.2f} +- {:.2f}'.format(bg, np.std(np.sqrt(Pkh[mask]/Pkdm[mask]))))
print('bias halos (redshift) PRE= {:.2f} +- {:.2f}'.format(bgRSD, np.std(np.sqrt(PkhRSD[mask]/Pkdm[mask]))))

# bg = bgRSD

# Compute celestial coordinates pre-reconstruction
# halos['SkyCoordz'] = transform.CartesianToSky(pos=halos['Position'], cosmo=cosmo, \
#     observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
# halos['SkyCoordzpec'] = transform.CartesianToSky(pos=halos['Position'], cosmo=cosmo, velocity=halos['Velocity'], \
#     observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# # Save catalog
# coord_array = np.zeros((nhalos, 4))
# coord_array[:, 0:3] = halos['SkyCoordz'].compute()
# coord_array[:, 3] = halos['SkyCoordzpec'][:,2].compute()
# mat = np.matrix(coord_array)
# with open('pairedsim'+str(paired)+'_cat_pre_z0.3.dat', 'wb') as ff:
#     for line in mat:
#         np.savetxt(ff, line, fmt='%.5f')

reconstructedQ, reconstructedQS = iteration(1, Length, Nc, zobs, zinit, halos, [Length/2, Length/2, Length/2], k, Pkh, delta_dm)

# Bias in real space
r = FFTPower(reconstructedQ, mode="1d")
Pkhreconr = r.power['power'].real - r.attrs['shotnoise']
bgreconr1 = np.mean(np.sqrt(Pkhreconr[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))

bgreconr2 = np.mean(np.sqrt(Pkhreconr[mask]/Pkdm[mask]))
print('bias (real) halos POST= {:.2f}(z={:.2f}),{:.2f}(z={:.2f})'.format(bgreconr1, zinit,bgreconr2, zobs))

# Bias in redshift space
r = FFTPower(reconstructedQS, mode="1d")
Pkhrecons = r.power['power'].real - r.attrs['shotnoise']

bgrecons1 = np.mean(np.sqrt(Pkhrecons[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))

bgrecons2 = np.mean(np.sqrt(Pkhrecons[mask]/Pkdm[mask]))

print('bias (redshift) halos POST= {:.2f}(z={:.2f}),{:.2f}(z={:.2f})'.format(bgrecons1, zinit,bgrecons2, zobs))

# Compute celestial coordinates post-reconstruction
halos['SkyCoordz'] = transform.CartesianToSky(pos=halos['PositionQ'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
halos['SkyCoordzpec'] = transform.CartesianToSky(pos=halos['PositionQS'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((nhalos, 6))
coord_array[:, 0:3] = halos['SkyCoordz'].compute()
coord_array[:, 3:6] = halos['SkyCoordzpec'].compute()
mat = np.matrix(coord_array)
print(mat[:2])
# with open('pairedsim'+str(paired)+'_cat_post_z3_compare.dat', 'wb') as ff:
#     for line in mat:
#         np.savetxt(ff, line, fmt='%.5f')

# halos.save('Halos_paired'+str(paired)+'_reconvr.bigfile')