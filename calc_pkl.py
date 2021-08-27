#!/bin/python
#
###################################################################################
# This code outpus the 1-loop P(k) of mono, quad and hexadecapoles
#  using the 1-loop coeffients computed with calc_1loopcomp.py
#
# input: 
#  args[1]: real or redshift space [r,z]
#  args[2]: reconstruction smoothing scale rs (if rs=0, no reconstruction)
#  args[3]: output redshift
#  args[4]: linear bias
#  args[5]: input file of 1-loop coefficients
#  args[6]: output filename
#  args[7]: whose set of cosmological parameters [GB or Takahashi]
#
# output:
#   1st col: k
#   2nd -- 4th cols: P_00 (mono,quad,hexa)
#   5th -- 7th cols: P_13 (mono,quad,hexa)
#   8th -- 10th cols: P_22 (mono,quad,hexa)
#
# Note:
#   cosmological parameters is changeable from the 58th line
#    but only used for computing growth factor at a given redshift
#
#                                                     Chiaki Hikage (4 May 2021)
###################################################################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.integrate import quad,dblquad
from scipy import integrate, interpolate

args=sys.argv

### space [r,z]
space = args[1]

### smoothing for reconstruction
rs = int(args[2])

### output redshift
zout = args[3]
print('output redshift:',zout)

### set linear bias
b1 = float(args[4])
print('linear bias:',b1)

### input 1-loop coefficients
inf_1loopcomp = args[5]

### output filename
outf = args[6]
print('output filename:',outf)

cosmo = args[7]
### set cosmology only used for computing the linear growth factor at z=zout
if cosmo == 'Takahashi':
    hubble=0.6727
    ob0=0.04917
    om0=0.3156
    od0=1-om0
    sig8=0.831
    ns=0.9645
elif cosmo == 'GB':
    obh2=0.02236
    hubble=0.6727
    ob0=obh2/hubble**2
    om0=0.315
    od0=1-om0
    sig8=0.8234
    ns=0.9649
params={'flat':True,'H0':hubble*100,'Om0':om0,'Ob0':ob0,'sigma8':sig8,'ns':ns}
cosmo_for_colossus=cosmology.setCosmology('myCosmo',params)
#cosmo_for_colossus=cosmology.setCosmology('planck18-only')
#pk_nowig = cosmo_for_colossus.matterPowerSpectrum(kin,model='eisenstein98',z=zout)

### linear growth factor
gz = cosmo_for_colossus.growthFactor(z=float(zout))
#print('growth factor: ',gz)

### linear growth rate
zp=float(zout)+0.01
zm=float(zout)-0.01
gz_u=cosmo_for_colossus.growthFactor(z=zp)
gz_l=cosmo_for_colossus.growthFactor(z=zm)
fz=-np.log(gz_u/gz_l)/np.log((1+zp)/(1+zm))

print('growth rate: ',fz)
omz=om0*(1+float(zout))**3/(om0*(1+float(zout))**3+od0)
print('cf. Omm(z)^0.55: ',omz**0.55)

if space == 'r':
    fz = 0.

kout = np.loadtxt(inf_1loopcomp,usecols=[0])
pklin = np.loadtxt(inf_1loopcomp,usecols=[1])

### number of P13/P22 components with different combinations of f, mu, and b1
### these numbers must be equal to those set in pkl1loop.py
nacomp=30
nbcomp=20

Bterm=np.zeros((nbcomp,len(kout)))
for i in range(nbcomp):
    Bterm[i] = np.loadtxt(inf_1loopcomp,usecols=[i+2])
        
Aterm=np.zeros((nacomp,len(kout)))
for i in range(nacomp):
    Aterm[i] = np.loadtxt(inf_1loopcomp,usecols=[i+2+nbcomp])

P_00 = np.zeros((3,len(kout)))
### 0:mono, 1:quad, 2:hexadeca pole of (b1+fz*mu**2)^2*pklin averaged over mu
P_00[0] = (b1*b1 + 2./3.*b1*fz + 1./5.*fz*fz)*pklin
P_00[1] = (4./3.*b1*fz + 4./7.*fz*fz)*pklin
P_00[2] = (8./35.*fz*fz)*pklin

### B_n: mu**2n
B_0 = Bterm[0]*b1+Bterm[1]*b1*b1+Bterm[2]*b1**3 + fz*(Bterm[3]*b1+Bterm[4]*b1*b1) + fz*fz*(Bterm[8]+Bterm[9]*b1)
B_1 = fz*(Bterm[5]+Bterm[6]*b1+Bterm[7]*b1*b1) + fz*fz*(Bterm[10]+Bterm[11]*b1) + fz**3*Bterm[14]
B_2 = fz*fz*(Bterm[12]+Bterm[13]*b1)+fz**3*Bterm[15]
B_3 = fz**3*Bterm[16]
    
P_13 = np.zeros((3,len(kout)))
### 0:mono, 1:quad, 2:hexadeca pole of (b1+fz*mu**2)*B_n*mu**2n averaged over mu
P_13[0] = B_0*(b1+fz/3.)+B_1*(b1/3.+fz/5.)+B_2*(b1/5.+fz/7.)+B_3*(b1/7.+fz/9.)
P_13[1] = B_0*(fz*2./3.)+B_1*(b1*2./3.+fz*4./7.)+B_2*(b1*4./7.+fz*10./21.)+B_3*(b1*10./21.+fz*40./99.)
P_13[2] = B_1*fz*8./35.+B_2*(b1*8./35.+fz*24./77.)+B_3*(b1*24./77.+fz*48./143.)

### A_n: mu**2n components
A_0 = Aterm[0]*b1**2+Aterm[1]*b1**3+Aterm[2]*b1**4 + fz*(Aterm[3]*b1**2+Aterm[4]*b1**3) + fz*fz*(Aterm[8]*b1+Aterm[9]*b1*b1) + fz**3*Aterm[15]*b1 + fz**4*Aterm[22]
A_1 = fz*(Aterm[5]*b1+Aterm[6]*b1**2+Aterm[7]*b1**3) + fz*fz*(Aterm[10]*b1+Aterm[11]*b1**2) + fz**3*(Aterm[16]+Aterm[17]*b1) + fz**4*Aterm[23]
A_2 = fz*fz*(Aterm[12]+Aterm[13]*b1+Aterm[14]*b1**2) + fz**3*(Aterm[18]+Aterm[19]*b1) + fz**4*Aterm[24]
A_3 = fz**3*(Aterm[20]+Aterm[21]*b1) + fz**4*Aterm[25]
A_4 = fz**4*Aterm[26]
        
P_22 = np.zeros((3,len(kout)))
### multipole expansion A_n*mu**2n averaged over mu
P_22[0] = A_0 + A_1/3. + A_2/5. + A_3/7. + A_4/9.
P_22[1] = A_1*2./3.+ A_2*4./7. + A_3*10./21. + A_4*80./198.
P_22[2] = A_2*8./35. + A_3*24./77. + A_4*48./143.


### output [k,P00(k,l=0),P00(k,l=2),P00(k,l=4),P13(k,l=0),P13(k,l=2),P13(k,l=4),P22(k,l=0),P22(k,l=2),P22(k,l=4)]
outval=np.zeros((10,len(kout)))

outval[0] = kout
for l in range(3):
    outval[l+1] = P_00[l]*gz*gz
    outval[l+4] = P_13[l]*gz**4
    outval[l+7] = P_22[l]*gz**4

np.savetxt(outf, outval.transpose())

