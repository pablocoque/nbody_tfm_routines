from nbodykit.lab import *
import numpy as np
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

def displace_galaxies(psi, posg, nc, boxsize, zinit, zobs):
    '''
    Reconstruction to a given redshift given a displacement field
    '''

    # tamano celda
    cellsize = boxsize/nc
    # interpolamos el campo de velocidad en la malla a la posición de la galaxia
    psi_g = np.zeros_like(posg)

    for i in range(len(psi_g)):
        j = int(np.floor(posg[i, 0]/cellsize))
        k = int(np.floor(posg[i, 1]/cellsize))
        l = int(np.floor(posg[i, 2]/cellsize))

        j = int(np.fmod(j,nc))
        k = int(np.fmod(k,nc))
        l = int(np.fmod(l,nc))

        psi_g[i, 0] = psi[j, k, l, 0]
        psi_g[i, 1] = psi[j, k, l, 1]
        psi_g[i, 2] = psi[j, k, l, 2]
    
    # calculamos los growth factors
    # Psi(q) está normalizado a D(z=0), por tanto si me voy a z=0 desde zinfty:
    # r(z=0)=q+Psi(q)
    # r(zobs)=q+D(zobs)*Psi(q)
    # r(zinit)=q+D(zinit)*Psi(q)

    # Entonces, para ir de zobs a zinit:
    # r(zinit)=r(zobs)-(D(zobs)-D(zinit))*Psi(q)
    posp = posg - (D(zobs) - D(zinit))*psi_g
    
    # periodic BC
    posp[posp < 0] += boxsize
    posp[posp > boxsize] -= boxsize
    return posp

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

# Compute celestial coordinates pre-reconstruction
hod['SkyCoordz'] = transform.CartesianToSky(pos=hod['Position'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
hod['SkyCoordzpec'] = transform.CartesianToSky(pos=hod['Position'], cosmo=cosmo, velocity=hod['Velocity'], \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = hod['SkyCoordz'].compute()
coord_array[:, 3] = hod['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('cat_pre_z0.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.2f')

# Compute redshift space distortions and shift galaxies
## Box center
position_origin = hod['Position'] - 0.5*Length

projection_norm = np.linalg.norm(position_origin, axis=1)

line_of_sight = np.zeros_like(position_origin)
line_of_sight = position_origin/projection_norm[:, np.newaxis]

rsd_factor = (1+zobs) / (100 * cosmo.efunc(zobs))

dot_prod = np.sum(hod['Velocity']*line_of_sight, axis=1)

hod['PositionRSD'] = position_origin + rsd_factor*dot_prod[:, np.newaxis]*line_of_sight + 0.5*Length

# Compute density field at redshift space
delta_gRSD = hod.to_mesh(resampler='cic', position='PositionRSD', interlaced=True, compensated=True)

# Obtain bias
mask = (k <= 0.04)*(k > 0.02)
bg = np.sqrt(np.mean(Pkg[mask]/Pkdm[mask]))

# Compute DM density field proxy
r_s = 10. # smoothing radius
delta_g_b = delta_gRSD.apply(divide_bias, mode='real', kind='index')
delta_dm_tilde = delta_g_b.apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber')

# Compute displacement field
psi = compute_zeld(Length, Nc, delta_dm_tilde.paint(mode='real'))

# Apply reconstruction
hod['Position_zinit'] = displace_galaxies(psi, hod['PositionRSD'].compute(), Nc, Length, zinit=zinit, zobs=zobs)

# Compute celestial coordinates post-reconstruction
hod['SkyCoordz'] = transform.CartesianToSky(pos=hod['Position_zinit'], cosmo=cosmo, \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')
hod['SkyCoordzpec'] = transform.CartesianToSky(pos=hod['Position_zinit'], cosmo=cosmo, velocity=hod['Velocity'], \
    observer=da.from_array([Length/2,Length/2,Length/2]), zmax=100., frame='icrs')

# Save catalog
coord_array = np.zeros((ngal, 4))
coord_array[:, 0:3] = hod['SkyCoordz'].compute()
coord_array[:, 3] = hod['SkyCoordzpec'][:,2].compute()
mat = np.matrix(coord_array)
with open('cat_post_z0_z3.dat', 'wb') as ff:
    for line in mat:
        np.savetxt(ff, line, fmt='%.2f')