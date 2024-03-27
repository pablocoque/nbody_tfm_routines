# DEPENDENCIES
from nbodykit.lab import *
import numpy as np
from general_tools import *

# FUNCTIONS
def obtain_paired(k, v):
    kk = sum(ki ** 2 for ki in k) # k^2 on the mesh
    Pklinth = power.__call__(kk)
    Pdelta = v.real**2 + v.imag**2
    mask = Pdelta==0
    Pdelta[mask] = 1
    return v*np.sqrt(Pklinth)/np.sqrt(Pdelta)

def convolve_NL(L, nc, deltaIC, deltaNL):
    deltaf = fft(deltaIC)
    deltaNLf = fft(deltaNL)

    deltaICNLf = deltaf.copy()

    #kfac = 2.0*np.pi/L

    for i in range(nc):
        for j in range(nc):
            for k in range(nc):
                PL = (deltaf[i,j,k].real**2 + deltaf[i,j,k].imag**2)
                PNL = (deltaNLf[i,j,k].real**2 + deltaNLf[i,j,k].imag**2)
                kernel = np.sqrt(PNL/PL)

                deltaICNLf[i,j,k] = kernel*deltaf[i,j,k]


    deltaICNL = ifft(deltaICNLf)
    deltaICNL = deltaICNL.real
    return deltaICNL

def get_kernel(k, v):
    return v.real**2 + v.imag**2

def NL_field(deltaL, deltaNL):
    Lkernel_field = deltaL.apply(get_kernel, mode='complex', kind='wavenumber')
    NLkernel_field = deltaNL.apply(get_kernel, mode='complex', kind='wavenumber')
    Lkernel = Lkernel_field.compute(mode='complex')
    mask = Lkernel == 0.
    Lkernel[mask] = 1.
    NLkernel = NLkernel_field.compute(mode='complex')
    kernel = np.sqrt(NLkernel/Lkernel)
    return kernel*deltaL.to_field(mode='complex')

def step_theta(delta, delta_th):
    if delta<delta_th:
        theta = 0
    else:
        theta = 1
    return theta

def obtain_rhog(delta, gamma, alpha, delta_th):
    theta = step_theta(delta, delta_th)
    # print('theta ', theta)
    if theta > 0:
        rhog = theta*gamma*((1.+delta)**alpha)
    else:
        rhog = 0
    return rhog

def make_catalog_g(delta, alpha, gamma, delta_th, Length, Nc):
    pos_array = []
    dL = Length/Nc

    for ix in range(Nc):
        for iy in range(Nc):
            for iz in range(Nc):
                rhog = obtain_rhog(delta[ix,iy,iz], gamma, alpha, delta_th)
                Ng = int(np.random.poisson(rhog))

                for i in range(Ng):
                    xpos = ix*dL + dL*np.random.uniform()
                    ypos = iy*dL + dL*np.random.uniform()
                    zpos = iz*dL + dL*np.random.uniform()

                    if xpos<0:
                        xpos+=Length
                    if xpos>=Length:
                        xpos-=Length

                    if ypos<0:
                        ypos+=Length
                    if ypos>=Length:
                        ypos-=Length

                    if zpos<0:
                        zpos+=Length
                    if zpos>=Length:
                        zpos-=Length

                    pos_array.append(np.array([xpos, ypos,zpos]))

    return np.array(pos_array)