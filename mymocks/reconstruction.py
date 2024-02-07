import sys
from nbodykit.lab import *
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import dask.array as da

# FUNCTIONS
def divide_bias(x, v):
    return v/bg

def fft(inarr):
    finarr= np.fft.fftn(inarr)
    return(finarr)

def ifft(finarr):
    inarr= np.fft.ifftn(finarr)
    return(inarr)

def compute_Psi(L, nc, delta):
   
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
    return psi_g

def compute_vr(vel, q, observer, z):
    position_origin = q - observer
    projection_norm = np.linalg.norm(position_origin, axis=1)
    line_of_sight = np.zeros_like(position_origin)
    line_of_sight = position_origin/projection_norm[:, np.newaxis]
    dot_prod = np.sum(vel*line_of_sight, axis=1)
    return f(z)*dot_prod[:, np.newaxis]*line_of_sight

def f(z):
    Omega_mz = cosmology.Planck15.Omega_m(z)
    f = Omega_mz**0.66
    return f

def D(z):
    return cosmology.background.MatterDominated(cosmology.Planck15.Omega0_m).D1(1/(1+z))

def periodic_conditions(coord, L):
    mask1 = coord<0
    mask2 = coord>=L
    coord[mask1] += L
    coord[mask2] -= L
    return coord

def iteration(Niter, L, nc, zobs, zinit, cat, observer, k, Pkcomp):#, deltadm):
    cat['PositionQ'] = cat['PositionRSD']
    s = cat['PositionRSD'].compute()
    delta = cat.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)
    r = FFTPower(delta, mode='1d')
    Pk1 = r.power['power'].real - r.attrs['shotnoise']
    for i in range(Niter):
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        ax.loglog(k, Pkcomp, 'k', label='PRE reconstruction')

        q = cat['PositionQ'].compute()
        psi = compute_Psi(L, nc, (delta.apply(divide_bias, mode='real', kind='index')\
            .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber').paint(mode='real') - 1.))
        # psi = compute_Psi(L, nc, (deltadm.paint(mode='real') - 1.))
        psi_g = field_interpolation(L, nc, psi, q)
        vr = compute_vr(psi_g, q, observer, zobs)
        
        cat['PositionQ'] = periodic_conditions(s - (D(zobs)-D(zinit))*psi_g - vr, L) # Iteration 1st step: q' = s - Psi(q) - vr(q)

        delta = cat.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)

        r = FFTPower(delta, mode='1d')
        Pk2 = r.power['power'].real - r.attrs['shotnoise']
        ax.loglog(k, Pk1, label='Iteration {:1n}'.format(i))
        ax.loglog(k, Pk2, ':', label='Iteration {:1n}'.format(i+1))
        ax.legend()
        ax.set_xlabel(r'$k [\mathrm{h Mpc}^{-1}]$', fontsize=18)
        ax.set_ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        # plt.show()
        plt.tight_layout()
        plt.savefig('Pkcomparison_iter{:1d}_paired'.format(i+1)+'.pdf')
        
        print('Iteration {:1d}, Mean difference between Pks: {:.2f}'.format(i+1, np.mean(abs(Pk1 - Pk2))))

        Pk1 = Pk2
    
    cat['PositionQS'] = periodic_conditions(s - (D(zobs)-D(zinit))*psi_g, L)
    deltaQS = cat.to_mesh(resampler='cic', position='PositionQS', interlaced=True, compensated=True)
    return delta, deltaQS
    
### MAIN

paired = sys.argv[1]

cosmo = cosmology.Planck15

### Import matter catalog
matter = BigFileCatalog('Matterpaired'+str(paired)+'_catalog.bigfile')

# Define global variables
Length = matter.attrs['BoxSize'][0]
Nc = matter.attrs['Nmesh'][0]
zobs = 0.3
zinit = 3.
r_s = 2.*(Length/Nc) # smoothing radius
print('Smoothing radius=', r_s)

# Compute matter density field
delta_dm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k,z)
r = FFTPower(delta_dm, mode="1d")
Pkdm = r.power['power'].real - r.power.attrs['shotnoise']
k = r.power['k']

### Import galaxy catalog
galaxy = BigFileCatalog('Galaxypaired'+str(paired)+'_catalog.bigfile')

# Compute galaxy density field
delta_g = galaxy.to_mesh(resampler='cic', interlaced=True, compensated=True)
delta_gRSD = galaxy.to_mesh(position='PositionRSD', resampler='cic', interlaced=True, compensated=True)
ngal = galaxy.csize

# Compute galaxy P(k,z)
r = FFTPower(delta_g, mode='1d')
Pkg = r.power['power'].real - r.attrs['shotnoise']

r = FFTPower(delta_gRSD, mode='1d')
PkgRSD = r.power['power'].real - r.attrs['shotnoise']

# Calculate bias pre reconstruction
mask = (k <= 0.09)*(k > 0.03)
bg = np.mean(np.sqrt(Pkg[mask]/Pkdm[mask]))
bgRSD = np.mean(np.sqrt(PkgRSD[mask]/Pkdm[mask]))

print('bias (real) PRE= {:.2f} +- {:.2f}'.format(bg, np.std(np.sqrt(Pkg[mask]/Pkdm[mask]))))
print('bias (redshift) PRE= {:.2f} +- {:.2f}'.format(bgRSD, np.std(np.sqrt(PkgRSD[mask]/Pkdm[mask]))))

# Valores de formulas
breczinitt = (bg-1)*(D(zobs)/D(zinit)) + 1
breczobst = (bg-1) + (D(zinit)/D(zobs))

print('Expected bias brez(zinit) = {:.2f}'.format(breczinitt))
print('Expected bias brez(zobs) = {:.2f}'.format(breczobst))

# Compute celestial coordinates pre-reconstruction
galaxy['SkyCoordz'] = transform.CartesianToSky(pos=galaxy['Position'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
galaxy['SkyCoordzpec'] = transform.CartesianToSky(pos=galaxy['PositionRSD'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = galaxy['SkyCoordz'].compute()
coord_array[:, 3] = galaxy['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('paired'+str(paired)+'_cat_pre_z0.3.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.5f')

reconstructedQ, reconstructedQS = iteration(3, Length, Nc, zobs, zinit, galaxy, np.array([Length/2, Length/2, Length/2]), k, Pkg)#, delta_dm)

# Bias in real space
r = FFTPower(reconstructedQ, mode='1d')
Pkgreconr = r.power['power'].real - r.attrs['shotnoise']
bgreconr1 = np.mean(np.sqrt(Pkgreconr[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))

bgreconr2 = np.mean(np.sqrt(Pkgreconr[mask]/Pkdm[mask]))
print('bias (real) POST= {:.2f}(z={:.2f}),{:.2f}(z={:.2f})'.format(bgreconr1, zinit,bgreconr2, zobs))

# Bias in redshift space
r = FFTPower(reconstructedQS, mode='1d')
Pkgrecons = r.power['power'].real - r.attrs['shotnoise']

bgrecons1 = np.mean(np.sqrt(Pkgrecons[mask]*(D(zobs)**2)/(Pkdm[mask]*(D(zinit)**2))))

bgrecons2 = np.mean(np.sqrt(Pkgrecons[mask]/Pkdm[mask]))

print('bias (redshift) POST= {:.2f}(z={:.2f}),{:.2f}(z={:.2f})'.format(bgrecons1, zinit,bgrecons2, zobs))

# Compute celestial coordinates post-reconstruction
galaxy['SkyCoordz'] = transform.CartesianToSky(pos=galaxy['PositionQ'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
galaxy['SkyCoordzpec'] = transform.CartesianToSky(pos=galaxy['PositionQS'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = galaxy['SkyCoordz'].compute()
coord_array[:, 3] = galaxy['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('paired'+str(paired)+'_cat_post_z3.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.5f')

galaxy.save('Galaxypaired'+str(paired)+'_catalog_POST.bigfile')