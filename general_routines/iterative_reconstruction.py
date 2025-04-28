# DEPENDENCIES
from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt
from general_tools import *

# FUNCTIONS
def iterative_reconstruction(Niter, L, nc, zlow, zhigh, tracer, matter, observer, redshift_pos='PositionRSD',
                             test=False, plot_iterations=False, savefig=False, plot_suffix=''):
    def divide_bias(x, v):
        return v/bg
    
    position = redshift_pos
    r_s = 2.*(L/nc)
    s = tracer[position].compute()
        
    if not test:
        for i in range(Niter):
            if plot_iterations:
                fig, ax = plt.subplots(1, 1, figsize=(8,6))
            
            bg = evaluate_bias(tracer, matter, tracer_pos=position)
            tracer['PositionQ'] = tracer[position]

            delta = tracer.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)
            r = FFTPower(delta, mode='1d')
            Pk1 = r.power['power'].real - r.attrs['shotnoise']
            
            q = tracer['PositionQ'].compute()
            psi = compute_Psi(L, nc, (delta.apply(divide_bias, mode='real', kind='index')\
                .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber')\
                .compute(mode='real') - 1.))

            psi_g = field_interpolation(L, nc, psi, q)
            vrzlow = compute_vr(psi_g, s, observer, zlow)
            
            # Iteration: q' = s(zlow) - D(zlow)Psi(q) - vr(r(zlow))
            tracer['PositionQ'] = periodic_conditions(s - D(zlow)*psi_g - vrzlow, L) 
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
        
        # Final reconstruction estimate: s(zhigh) = q' - (D(zlow) - D(zhigh))Psi(q) - (vr(zlow) - vr(r(zhigh))
        q = tracer['PositionQ'].compute()
        delta = tracer.to_mesh(resampler='cic', position='PositionQ', interlaced=True, compensated=True)

        psi = compute_Psi(L, nc, (delta.apply(divide_bias, mode='real', kind='index')\
            .apply(filters.Gaussian(r=r_s).filter, mode='complex', kind='wavenumber')\
            .compute(mode='real') - 1.))

        psi_g = field_interpolation(L, nc, psi, q)
        vrzlow = compute_vr(psi_g, s, observer, zlow)
        vrzhigh = compute_vr(psi_g, q, observer, zhigh)
        tracer['PositionQS'] = periodic_conditions(s - (D(zlow) + D(zhigh))*psi_g - vrzlow + vrzhigh, L)
    else:
        deltadm = matter.to_mesh(resampler='cic', interlaced=True, compensated=True)
        psi = compute_Psi(L, nc, (deltadm.compute(mode='real') - 1.))
        psi_g = field_interpolation(L, nc, psi, s)
        vrzlow = compute_vr(psi_g, s, observer, zlow)
        tracer['PositionQ'] = periodic_conditions(s - D(zlow)*psi_g - vrzlow, L) 
        psi_g = field_interpolation(L, nc, psi, tracer['PositionQ'].compute())
        vrzhigh = compute_vr(psi_g, tracer['PositionQ'].compute(), observer, zhigh)
        tracer['PositionQS'] = periodic_conditions(s - (D(zlow) + D(zhigh))*psi_g - vrzlow + vrzhigh, L)
    return