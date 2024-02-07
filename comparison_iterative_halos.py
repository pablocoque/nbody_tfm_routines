from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt

# FUNCTIONS
def divide_bias(x, v):
    return v/bh

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

def galaxy_displacement(L, nc, psi, s):
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

# sprime = s - psi(s) iterative
# s2prime = s - psi(sprime)
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
        psi_g = galaxy_displacement(L, nc, psi, Sp.compute())
        
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

# qprime = r - psi(r) iterative
# q2prime = r - psi(qprime)
def iteration_r(Niter, L, nc, zobs, zinit, cat):
    R = cat['Position']
    cat['PositionQp'] = cat['Position']
    Qp = cat['PositionQp']
    delta = cat.to_mesh(resampler='cic', position='PositionQp', interlaced=True, compensated=True)
    r = FFTPower(delta, mode='1d')
    Pk1 = r.power['power'].real - r.attrs['shotnoise']
    # k1 = r.power['k']
    for i in range(Niter):
        # plt.loglog(k, Pkcomp, 'k', label=r'$P_{\mathrm{lin}}(z=0.3)$')

        psi = compute_zeld(L, nc, delta.apply(divide_bias, mode='real', kind='index')\
            .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber').paint(mode='real'))
        psi_g = galaxy_displacement(L, nc, psi, Qp.compute())
        
        # Qp = R - psi(R)
        cat['PositionQp'] = R - (D(zobs) - D(zinit))*psi_g
        Qp = cat['PositionQp']
        delta = cat.to_mesh(resampler='cic', position='PositionQp', interlaced=True, compensated=True)
        
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

cosmo = cosmology.Planck15

### Import DM catalog
matter = BigFileCatalog('fastpm_dmcatalog.bigfile')

# Define global variables
Length = matter.attrs['BoxSize'][0]
Nc = matter.attrs['Nmesh'][0]
zobs = matter.attrs['plin.redshift']
print('Initial redshift z=', zobs)
zinit = 3.
print('Final redshift z={:.1f}'.format(zinit))
r_s = 10.
print('Smoothing radius r={:.1f} Mpc/h'.format(r_s))

# Compute matter density field
delta_dm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)

# Compute matter P(k,z)
r = FFTPower(delta_dm, mode="1d", Nmesh=Nc)
# shot-noise from nbodykit
Pkdm = r.power['power'].real - r.power.attrs['shotnoise']
k = r.power['k']
mask = (k <= 0.08)*(k > 0.02)

### Import Galaxy catalog
halos = BigFileCatalog('halos_catalogue.bigfile')

# Compute galaxy density field
delta_h = halos.to_mesh(resampler='cic', interlaced=True, compensated=True)
ngal = halos.csize

# Compute galaxy P(k,z)
r = FFTPower(delta_h, mode="1d", Nmesh=Nc)
# shot-noise from nbodykit
Pkh = r.power['power'].real - r.attrs['shotnoise']

# move halos to redshift space
## RSD formula
# vr = 1/(Ha) * (vec(v).vec(r_unit))vec(r_unit)
## Box centered
position_origin = halos['Position'] - 0.5*Length

projection_norm = np.linalg.norm(position_origin, axis=1)

line_of_sight = np.zeros_like(position_origin)
line_of_sight = position_origin/projection_norm[:, np.newaxis]

rsd_factor = (1+zobs) / (100 * cosmo.efunc(zobs))

dot_prod = np.sum(halos['Velocity']*line_of_sight, axis=1)

halos['PositionRSD'] = position_origin + rsd_factor*dot_prod[:, np.newaxis]*line_of_sight + 0.5*Length

# Bias from halos PRE
bh = np.mean(np.sqrt(Pkh[mask]/Pkdm[mask]))
print('bias halos PRE= {:.2f}'.format(bh))

deltaS5p = iteration_s(5, Length, Nc, zobs, zinit, halos)
deltaQ1p = iteration_r(1, Length, Nc, zobs, zinit, halos)
deltaQ5p = iteration_r(5, Length, Nc, zobs, zinit, halos)

r = FFTPower(deltaS5p, mode='1d', Nmesh=Nc)
PkS5p = r.power['power'].real - r.attrs['shotnoise']

r = FFTPower(deltaQ1p, mode='1d', Nmesh=Nc)
PkQ1p = r.power['power'].real - r.attrs['shotnoise']

r = FFTPower(deltaQ5p, mode='1d', Nmesh=Nc)
PkQ5p = r.power['power'].real - r.attrs['shotnoise']

plt.loglog(k, PkS5p, label='Practical, 5 iter')
plt.loglog(k, PkQ1p, label='Ideal, 1 iter')
plt.loglog(k, PkQ5p, label='Ideal, 5 iter')
plt.legend()
plt.xlabel(r'$k\ [\mathrm{h Mpc}^{-1}]$')
plt.ylabel(r'$P(k, z)$ $[h^{-3} \mathrm{Mpc}^3]$')
plt.savefig('Comparison_iterative.pdf')
