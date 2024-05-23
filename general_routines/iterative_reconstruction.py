# DEPENDENCIES
from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt
from general_tools import *

# FUNCTIONS
def iterative_reconstruction(Niter, L, nc, zobs, zinit, tracer, matter, observer, real_space=False,
                             test=False, plot_iterations=False, savefig=False, plot_suffix=''):
    def divide_bias(x, v):
        return v/bg
    
    r_s = 2.*(L/nc)
    if real_space:
        tracer['PositionQ'] = tracer['Position']
        s = tracer['Position'].compute()
        bg = evaluate_bias(tracer, matter, tracer_pos='Position')
    else:
        tracer['PositionQ'] = tracer['PositionRSD']
        s = tracer['PositionRSD'].compute()
        bg = evaluate_bias(tracer, matter, tracer_pos='PositionRSD')
    
    for i in range(Niter):
        if plot_iterations:
            fig, ax = plt.subplots(1, 1, figsize=(8,6))
        
        delta = tracer.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)
        r = FFTPower(delta, mode='1d')
        Pk1 = r.power['power'].real - r.attrs['shotnoise']
        
        q = tracer['PositionQ'].compute()
        if test:
            deltadm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)
            psi = compute_Psi(L, nc, (deltadm.compute(mode='real') - 1.))
        else:
            psi = compute_Psi(L, nc, (delta.apply(divide_bias, mode='real', kind='index')\
                .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber')\
                .compute(mode='real') - 1.))

        psi_g = field_interpolation(L, nc, psi, q)
        vr = compute_vr(psi_g, q, observer, zobs)

        # Iteration: q' = s(zobs) - D(zobs)Psi(q) - vr(r(zobs))
        tracer['PositionQ'] = periodic_conditions(s - D(zobs)*psi_g - vr, L) 
        delta = tracer.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)

        r = FFTPower(delta, mode='1d')
        Pk2 = r.power['power'].real - r.attrs['shotnoise']
        
        if plot_iterations:
            k = r.power['k']
            ax.loglog(k, Pk1, label='Iteration {:1n}'.format(i))
            ax.loglog(k, Pk2, ':', label='Iteration {:1n}'.format(i+1))
            ax.legend()
            ax.set_xlabel(r'$k [\mathrm{h Mpc}^{-1}]$', fontsize=18)
            ax.set_ylabel(r'$P(k)$ $[h^{-3} \mathrm{Mpc}^3]$', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=14)
            if savefig:
                plt.tight_layout()
                plt.savefig('reconstruction_iter_{:1d}'.format(i+1)+plot_suffix+'.pdf')
            else:
                plt.show()
        
        print('Iteration {:1d}, Mean difference between Pks: {:.2f}'.format(i+1, np.mean(abs(Pk1 - Pk2))))
    
    # Final reconstruction estimate: s(zinit) = q' + D(zinit)Psi(q) + vr(r(zinit))
    q = tracer['PositionQ'].compute()
    vr = compute_vr(psi_g, q, observer, zinit)
    tracer['PositionQS'] = periodic_conditions(q + D(zinit)*psi_g + vr, L)
    return