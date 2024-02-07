import sys
from nbodykit.lab import *

paired = sys.argv[1]

halos = BigFileCatalog('Halos_paired'+str(paired)+'.bigfile')
Length = halos.attrs['BoxSize'][0]

s = halos['PositionRSD'].compute()

print(s.min(), s.max())

maskx1 = s[:,0]<0.
maskx2 = s[:,0]>=Length
s[maskx1,0] += Length
s[maskx2,0] -= Length
masky1 = s[:,1]<0.
masky2 = s[:,1]>=Length
s[masky1,1] += Length
s[masky2,1] -= Length
maskz1 = s[:,2]<0.
maskz2 = s[:,2]>=Length
s[maskz1,2] += Length
s[maskz2,2] -= Length

print(s.min(), s.max())

halos['PositionRSD'] = s

halos.save('Halos_paired'+str(paired)+'_rsdfixed.bigfile')