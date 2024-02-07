import sys
from nbodykit.lab import *
import numpy as np
import dask.array as da

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

def compute_vr(vel, q, observer, zobs):
    # Omega_mz = cosmology.Planck15.Omega_m(zobs)
    # f = Omega_mz**0.6
    position_origin = q - observer
    projection_norm = np.linalg.norm(position_origin, axis=1)
    line_of_sight = np.zeros_like(position_origin)
    line_of_sight = position_origin/projection_norm[:, np.newaxis]
    dot_prod = np.sum(vel*line_of_sight, axis=1)
    return f(zobs)*dot_prod[:, np.newaxis]*line_of_sight

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
observer = [Length/2, Length/2, Length/2]
print('Smoothing radius=', r_s)

# Compute matter density field
delta_dm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k,z)
r = FFTPower(delta_dm, mode="1d")
# shot-noise from nbodykit
Pkdm = r.power['power'].real - r.power.attrs['shotnoise']
k = r.power['k']

### Import Halos catalog
halos = BigFileCatalog('Halos_paired'+str(paired)+'_vrN.bigfile')
nhalos = halos.csize
delta = halos.to_mesh(resampler='cic', position='Position', interlaced=True, compensated=True)
r = FFTPower(delta, mode="1d")
Pkh = r.power['power'].real - r.attrs['shotnoise']

# Obtain bias
mask = (k <= 0.09)*(k > 0.03)
bg = np.sqrt(np.mean(Pkh[mask]/Pkdm[mask]))

print('bias halos PRE= {:.2f}'.format(bg))

s = halos['PositionRSD'].compute()

psi = compute_zeld(Length, Nc, delta.apply(divide_bias, mode='real', kind='index')\
        .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber').compute(mode='real'))
psi_g = field_interpolation(Length, Nc, psi, s)
vr = compute_vr((D(zobs)-D(zinit))*psi_g, s, observer, zobs)

halos['PositionQS'] = halos['PositionQ'] + vr

deltaQ = halos.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)

r = FFTPower(deltaQ, mode="1d")
Pkhreconr = r.power['power'].real - r.attrs['shotnoise']
bgreconr1 = np.mean(np.sqrt(Pkhreconr[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))

bgreconr2 = np.mean(np.sqrt(Pkhreconr[mask]/Pkdm[mask]))
print('bias (real) halos POST= {:.2f}(z={:.2f}),{:.2f}(z={:.2f})'.format(bgreconr1, zinit,bgreconr2, zobs))

deltaQS = halos.to_mesh(resampler='cic', position='PositionQS', interlaced=True, compensated=True)

r = FFTPower(deltaQS, mode="1d")
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
coord_array = np.zeros((nhalos, 4))
coord_array[:, 0:3] = halos['SkyCoordz'].compute()
coord_array[:, 3] = halos['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('pairedsim'+str(paired)+'_cat_post_z3_vrs.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.5f')

halos.save('Halos_paired'+str(paired)+'_vrs.bigfile')