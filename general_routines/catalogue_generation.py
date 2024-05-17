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
    kernel = np.sqrt(Pklinth)/np.sqrt(Pdelta)
    kernel[mask] = 0.
    return kernel*v

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
    kernel[mask] = 0.
    return kernel*deltaL.to_field(mode='complex')

def step_theta(delta, delta_th):
    if delta<delta_th:
        theta = 0
    else:
        theta = 1
    return theta

def obtain_rhog(delta, gamma, alpha, delta_th):
    theta = step_theta(delta, delta_th)
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

                    pos_array.append(np.array([xpos, ypos,zpos]))

    return periodic_conditions(np.array(pos_array), Length)