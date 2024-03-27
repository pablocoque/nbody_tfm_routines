######################################################################################3
# Nbody mocks
# de Francisco Shu Kitaura Joyanes - jueves, 11 de octubre de 2018, 12:49
# Number of replies: 0
######################################################################################3

import fastpm
from fastpm.nbkit import FastPMCatalogSource
from nbodykit.lab import *
from nbodykit import setup_logging, style
import matplotlib.pyplot as plt

Nc=200
Length=300 # 1 Gpc

print("ATTENTION!!!! Number of DM particles = ", Nc**3, " in a volume box of side length = ", Length, "Mpc/h")
print("If you want to change these values, change Nc and Length hard coded")
print()

#Set up IC
print("Computing P(k) linear with Planck15 cosmology")
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, 0)
linear = LinearMesh(Plin, BoxSize=Length, Nmesh=Nc,seed=42)

#Plot linear density
one_plus_delta = linear.paint(mode='real')
plt.imshow(one_plus_delta.preview(axes=[0,1]))
print("-----> Saving density plot in Linear_dens.png")
plt.savefig('Linear_dens.png')
plt.show()


#Compute P(k) of the IC
print("-----> Saving linear P(k) in Lin_power_spectrum.txt")
r = FFTPower(linear, mode="1d", Nmesh=Nc)
k = r.power['k']
Pk = r.power['power'].real
outfile = open('Lin_power_spectrum.txt', 'w')
for line in range(len(k)):
    outfile.write('%f %f\n' % (k[line], Pk[line]))
outfile.close()


#Run simulation
print("Running FastPM simulation")
sim = FastPMCatalogSource(linear, Nsteps=10)
#sim = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=500., Nmesh=128, bias=1.0, seed=42)

#Plot DM density
mesh = sim.to_mesh(window='tsc')
one_plus_delta = mesh.paint(mode='real')
plt.imshow(one_plus_delta.preview(axes=[0,1]))
print("-----> Saving density plot in DM_dens.png")
plt.savefig('DM_dens.png')
plt.show()


#Compute P(k) of the DM
print("-----> Saving DM P(k) in DM_power_spectrum.txt")
r = FFTPower(sim, mode="1d", Nmesh=Nc)
k = r.power['k']
Pk = r.power['power'].real-1/(Nc**3./(Length**3.))
outfile = open('DM_power_spectrum.txt', 'w')
for line in range(len(k)):
    outfile.write('%f %f\n' % (k[line], Pk[line]))
outfile.close()

#Compute the mass of each DM particle
print("Computing the mass of DM particle")
h=cosmology.Planck15.h
Omega_M = cosmology.Planck15.Omega0_cdm + cosmology.Planck15.Omega0_b
H0Mpc=100.0*h*(3.24078e-20)
GMsunMpcS = 4.5182422e-48
pi = 3.14159265359
rhomean=3.0/(8.0*pi*GMsunMpcS)*Omega_M*(H0Mpc*H0Mpc)
Mdm=rhomean*Length*Length*Length/h/h
DM_part_mass = Mdm/(Nc*Nc*Nc)
print('Mass of each particle = ', DM_part_mass, "Msun/h")


# run FOF to identify halo groups
print("Running FoF to compute halos")
fof = FOF(sim, 0.2*(Length/Nc), nmin=20, absolute=True)
halos = fof.to_halos(DM_part_mass, cosmo, 0.)

#Plot Halos density
mesh = halos.to_mesh(window='tsc')
one_plus_delta_halos = mesh.paint(mode='real')
plt.imshow(one_plus_delta_halos.preview(axes=[0,1]))
print("-----> Saving density plot in Halos_dens.png")
plt.savefig('Halos_dens.png')
plt.show()


#Compute P(k) of the Halos
print("-----> Saving Halos P(k) in Halos_power_spectrum.txt")
r = FFTPower(halos, mode="1d", Nmesh=Nc)
k = r.power['k']
nhal=halos.csize
print("-----> Number of halos = ",nhal)
Pk = r.power['power'].real-1/(nhal/(Length**3.))
outfile = open('Halos_power_spectrum.txt', 'w')
for line in range(len(k)):
    outfile.write('%f %f\n' % (k[line], Pk[line]))
outfile.close()


# populate halos with galaxies
print("Computing HOD with Zheng07 model")
hod = halos.populate(Zheng07Model)

#Plot HOD density
mesh = hod.to_mesh(window='tsc')
one_plus_delta_hod = mesh.paint(mode='real')
plt.imshow(one_plus_delta_hod.preview(axes=[0,1]))
print("-----> Saving density plot in HOD_dens.png")
plt.savefig('HOD_dens.png')
plt.show()


# Compute and save Pk
print("-----> Saving HOD P(k) in Galaxies_power_spectrum.txt")
r = FFTPower(hod, mode="1d", Nmesh=Nc)
k = r.power['k']
ngal=hod.csize
print("-----> Number of galaxies = ",ngal)
Pk = r.power['power'].real-1/(ngal/(Length**3.))
outfile = open('Galaxies_power_spectrum.txt', 'w')
for line in range(len(k)):
    outfile.write('%f %f\n' % (k[line], Pk[line]))
outfile.close()

print("Finished")
