# DEPENDENCIES
from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt
from general_tools import *

# FUNCTIONS
def iterative_reconstruction(Niter, L, nc, zobs, zinit, tracer, matter, observer, real_space=False, test=False):
    def divide_bias(x, v):
        return v/bg
    
    r_s = 2.*(L/nc)
    if real_space:
        tracer['PositionQ'] = tracer['Position']
        s = tracer['Position'].compute()
    else:
        tracer['PositionQ'] = tracer['PositionRSD']
        s = tracer['PositionRSD'].compute()
    
    for i in range(Niter):
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        
        delta = tracer.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)
        r = FFTPower(delta, mode='1d')
        Pk1 = r.power['power'].real - r.attrs['shotnoise']
        
        q = tracer['PositionQ'].compute()
        if test:
            deltadm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)
            psi = compute_Psi(L, nc, (deltadm.paint(mode='real') - 1.))
        else:
            bg = evaluate_bias(tracer, matter, tracer_pos='PositionQ')
            psi = compute_Psi(L, nc, (delta.apply(divide_bias, mode='real', kind='index')\
                .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber')\
                .paint(mode='real') - 1.))

        psi_g = field_interpolation(L, nc, psi, q)
        vr = compute_vr(psi_g, q, observer, zobs)

        # Iteration: q' = s - Psi(q) - vr(q)
        tracer['PositionQ'] = periodic_conditions(s - (D(zobs)-D(zinit))*psi_g - vr, L) 
        delta = tracer.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)

        r = FFTPower(delta, mode='1d')
        Pk2 = r.power['power'].real - r.attrs['shotnoise']
        
        k = r.power['k']
        ax.loglog(k, Pk1, label='Iteration {:1n}'.format(i))
        ax.loglog(k, Pk2, ':', label='Iteration {:1n}'.format(i+1))
        ax.legend()
        ax.set_xlabel(r'$k [\mathrm{h Mpc}^{-1}]$', fontsize=18)
        ax.set_ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.show()
        # plt.tight_layout()
        # plt.savefig('Pkcomparison_iter{:1d}_paired'.format(i+1)+'.pdf')
        
        print('Iteration {:1d}, Mean difference between Pks: {:.2f}'.format(i+1, np.mean(abs(Pk1 - Pk2))))
    
    tracer['PositionQS'] = periodic_conditions(s - (D(zobs)-D(zinit))*psi_g, L)
    return