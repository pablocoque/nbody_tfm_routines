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

    psi   = np.zeros((nc,nc,nc,3))

    psi[:,:,:,0] = vx
    psi[:,:,:,1] = vy
    psi[:,:,:,2] = vz

    return psi

def forward_evolution(L, nc, vel):

    dL = L/nc

    vx = vel[:,:,:,0]
    vy = vel[:,:,:,1]
    vz = vel[:,:,:,2]

    pos = np.zeros((nc,nc,nc,3))

    for i in range(nc):
          for j in range(nc):
            for k in range(nc):
                xp = (float(i) + 0.5)*dL
                yp = (float(j) + 0.5)*dL
                zp = (float(k) + 0.5)*dL

                xp += vx[i,j,k]
                yp += vy[i,j,k]
                zp += vz[i,j,k]

                pos[i,j,k,0] = xp
                pos[i,j,k,1] = yp
                pos[i,j,k,2] = zp

    posn3 = pos.reshape(nc**3, 3)
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
