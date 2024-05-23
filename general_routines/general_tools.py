# DEPENDENCIES
from nbodykit.lab import *
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# FUNCTIONS
def fft(inarr):
    finarr= np.fft.fftn(inarr)
    return(finarr)

def ifft(finarr):
    inarr= np.fft.ifftn(finarr)
    return(inarr)


def compute_Psi(L, nc, delta):
    '''
    Obtain displacement field under Zeldovich approximation
    '''

    deltaf = fft(delta)

    vfx = np.zeros_like(deltaf)
    vfy = np.zeros_like(deltaf)
    vfz = np.zeros_like(deltaf)

    kfac = 2.0*np.pi/L

    kii = np.arange(nc)
    mask = kii > nc/2
    kii[mask] = kii[mask] - nc
    kii = kfac*np.float64(kii)

    kx, ky, kz = np.meshgrid(kii, kii, kii)
    k2 = kx**2 + ky**2 + kz**2

    for i in range(nc):
        if k2[i,i,i]==0:
            vfx.real[i,i,i] =  0
            vfx.imag[i,i,i] =  0
            vfy.real[i,i,i] =  0
            vfy.imag[i,i,i] =  0
            vfz.real[i,i,i] =  0
            vfz.imag[i,i,i] =  0
        else:
            vfx.real[i,:,:] = -kii[i]*deltaf.imag[i,:,:]/k2[i,:,:]
            vfx.imag[i,:,:] =  kii[i]*deltaf.real[i,:,:]/k2[i,:,:]
            vfy.real[:,i,:] = -kii[i]*deltaf.imag[:,i,:]/k2[:,i,:]
            vfy.imag[:,i,:] =  kii[i]*deltaf.real[:,i,:]/k2[:,i,:]
            vfz.real[:,:,i] = -kii[i]*deltaf.imag[:,:,i]/k2[:,:,i]
            vfz.imag[:,:,i] =  kii[i]*deltaf.real[:,:,i]/k2[:,:,i]

    vxg = ifft(vfx)
    vx = vxg.real

    vyg = ifft(vfy)
    vy = vyg.real

    vzg = ifft(vfz)
    vz = vzg.real

    psi = np.zeros((nc,nc,nc,3))

    psi[:,:,:,0] = vx
    psi[:,:,:,1] = vy
    psi[:,:,:,2] = vz

    return psi

def forward_evolution(L, nc, vel):

    dL = L/nc

    pos = np.mgrid[0:nc,0:nc,0:nc].astype(np.float64)
    pos = np.transpose(pos, (1,2,3,0))*dL + 0.5*dL

    for i in range(3):
        pos[:,:,:,i] += vel[:,:,:,i]

    posn3 = pos.reshape((nc**3, 3))
    periodic_conditions(posn3,L)

    return posn3

def field_interpolation(L, nc, psi, pos):

    psi_g = np.zeros_like(pos)

    x = np.linspace(0, L, nc)
    y = np.linspace(0, L, nc)
    z = np.linspace(0, L, nc)

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

def compute_vr(vel, q, observer, z):
    position_origin = q - observer
    projection_norm = np.linalg.norm(position_origin, axis=1)
    line_of_sight = np.zeros_like(position_origin)
    line_of_sight = position_origin/projection_norm[:, np.newaxis]
    dot_prod = np.sum(vel*line_of_sight, axis=1)
    return f(z)*dot_prod[:, np.newaxis]*line_of_sight

def periodic_conditions(coord, L):
    mask1 = coord<0.
    mask2 = coord>=L
    coord[mask1] += L
    coord[mask2] -= L
    return coord

def boxfit_conditions(coord, L):
    mask1 = coord<0.
    mask2 = coord>=L
    coord[mask1] = 0.
    coord[mask2] = L
    return coord

def evaluate_bias(tracer, matter, tracer_pos = 'Position', kmin=0.03, kmax=0.09, return_std=False):
    delta_matter = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)
    r = FFTPower(delta_matter, mode='1d')
    Pkm = r.power['power'].real - r.attrs['shotnoise']

    delta_tracer = tracer.to_mesh(position=tracer_pos, resampler='cic', interlaced=True, compensated=True)
    r = FFTPower(delta_tracer, mode='1d')
    Pkg = r.power['power'].real - r.attrs['shotnoise']

    k = r.power['k']
    mask = (k <= kmax)*(k > kmin)
    if return_std:
        return np.mean(np.sqrt(Pkg[mask]/Pkm[mask])), np.std(np.sqrt(Pkg[mask]/Pkm[mask]))
    else:
        return np.mean(np.sqrt(Pkg[mask]/Pkm[mask]))
