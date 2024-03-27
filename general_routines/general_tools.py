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

    vel1D = np.zeros(nc**3 * 3)
    psi   = vel1D.reshape(nc,nc,nc,3)

    psi[:,:,:,0] = vx
    psi[:,:,:,1] = vy
    psi[:,:,:,2] = vz

    return psi

def forward_evolution(L, nc, vel):

    dL = L/nc

    vx = vel[:,:,:,0]
    vy = vel[:,:,:,1]
    vz = vel[:,:,:,2]

    pos1d = np.zeros(nc**3*3)
    pos = pos1d.reshape(nc,nc,nc,3)
    posic=pos
    posn3 = pos1d.reshape(nc**3, 3)

    for i in range(nc):
          for j in range(nc):
            for k in range(nc):

                rx = 0.5*dL
                ry = 0.5*dL
                rz = 0.5*dL

                xp = float(i)*dL+rx
                yp = float(j)*dL+ry
                zp = float(k)*dL+rz

                posic[i,j,k,0]=xp
                posic[i,j,k,1]=yp
                posic[i,j,k,2]=zp

                xp += vx[i,j,k]
                yp += vy[i,j,k]
                zp += vz[i,j,k]

                if(xp<0):
                    xp += L
                if(xp>=L):
                    xp -= L

                if(yp<0):
                    yp += L
                if(yp>=L):
                    yp -= L

                if(zp<0):
                    zp += L
                if(zp>=L):
                    zp -= L

                pos[i,j,k,0] = xp
                pos[i,j,k,1] = yp
                pos[i,j,k,2] = zp

                l = k + j*(nc + i)
                posn3[l,0] = xp
                posn3[l,1] = yp
                posn3[l,2] = zp

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
    mask1 = coord<0
    mask2 = coord>=L
    coord[mask1] += L
    coord[mask2] -= L
    return coord