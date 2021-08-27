#!/bin/python
#
###################################################################################
# This code is computing the coefficients of 1-loop components of pre- and post-P(k)
#  with different combinations of mu, fz (growth rate), and linear bias.
#
# input: 
#  args[1]: real or redshift space [r,z]
#  args[2]: reconstruction smoothing scale rs (if rs=0, no reconstruction)
#  args[3]: input linear (real-space) P(k) at z=0
#  args[4]: output filename
#
# output:
#   The coefficients of 1-loop terms P_13 and P_22 
#       (see eqs A1&A2 in Appendix of arXiv:1911.06461) including linear bias b1 compoenents
#   1st col: k
#   2nd col: linear P(k) 
#   3rd -- 22th cols: P_13(k) coefficients B_nmp(k) which depends on mu^2n * f^m * b1^p
#     with the following order:
#       B_001,B_002,B_003,B_011,B_012,B_110,B_111,B_112,B_020,B_021,
#       B_120,B_121,B_220,B_221,B_130,B_230,B_330 (remaining 3 cols are zero)
#
#   23th -- 52th cols: P_22(k) coefficients A_nmp(k) which depends on mu^2n * f^m * b1^p
#     with the following order:
#       A_002,A_003,A_004,A_012,A_013,A_111,A_112,A_113,A_021,A_022,
#       A_121,A_122,A_220,A_221,A_222,A_031,A_130,A_131,A_230,A_231,
#       A_330,A_331,A_040,A_140,A_240,A_340,A_440 (remaining 3 cols are zero)
#
#  Note:
#   output range of k and its binning is assumed in the 71th line
#    but you can change as you wish
#
#                                                     Chiaki Hikage (4 May 2021)
###################################################################################
#
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad,dblquad
from scipy import integrate, interpolate

args=sys.argv

### space [r,z]
space = args[1]

### smoothing for reconstruction
rs = int(args[2])
if rs > 0:
    rec = 'y'
else:
    rec = 'n'

if rec == 'n':
    print('no recontruction')
else:
    print('recontruction with Rs[Mpc/h]=',rs)

### read input power spectrum at z=0
infpk = args[3]
kin = np.loadtxt(infpk,usecols=[0])
pklin = np.loadtxt(infpk,usecols=[1])
print('input linear power spectrum at z=0:',infpk)

### output directory
outf = args[4]
print('output filename:',outf)

### output range of k and binning
kout = np.linspace(0.01, 0.6, 60)

### number of P13/P22 components with different combinations of f, mu, and b1
nacomp=30
nbcomp=20

### set output number of rows
outval=np.zeros((len(kout),2+nacomp+nbcomp))

### range of k for computing 1-loop components
logkmin=-3.99
logkmax=2.
kmin=10.**logkmin
kmax=10.**logkmax

kmin=max(kmin,kin[0])
kmax=min(kmax,kin[len(kin)-1])

logkin = np.log(kin)
logpk = np.log(pklin)

interp1d_pkl = interpolate.interp1d(logkin,logpk,kind='cubic')

def win(k):
    return np.exp(-0.5*(k*rs)**2)

for index_k, k in enumerate(kout):

    rmax=kmax/k
    rmin=kmin/k
    logk=np.log(k)

    pklout=np.exp(interp1d_pkl(logk))

    def f13(r,i):

        xmin=np.maximum(-1.,(1+r*r-rmax*rmax)/2./r)
        xmax=np.minimum(1.,(1+r*r-rmin*rmin)/2./r)

        Borg=0.

        ### B00: 0:b1,1:b1**2,2:b1**3
        ### B01: 3:b1,4:b1**2
        ### B11: 5:nobias,6:b1,7:b1*b1
        ### B02: 8:nobias,9:b1
        ### B12: 10:nobias,11:b1
        ### B22: 12:nobias,13:b1
        ### B13: 14:nobias
        ### B23: 15:nobias
        ### B33: 16:nobias

        if space == 'r':
            if r < 5e-3:
                if i == 0:
                    Borg = -2./3. + 232./315.*r*r - 376./735.*r**4
                else:
                    Borg =0.

            elif r > 5e2:
                s = 1./r
                if i == 0:
                    Borg = -122./315. + 8./105.*s*s - 40./1323.*s**4
                else:
                    Borg = 0.
            
            elif r >= 0.995 and r<=1.005:
                r1 = r - 1.
                if i == 0:
                    Borg = (-22. + 2.*r1 - 29.*r1**2)/63.
                else:
                    Borg = 0.
            else:
                if i == 0:
                    Borg = (12./r/r - 158 + 100.*r*r - 42.*r**4 + 3./(r*r*r)*(r*r-1)**3*(7*r*r+2)*np.log(np.abs((1+r)/(1-r))))/252.
                else:
                    Borg=0.
                    
            if rec == 'y':
                def fp13_int_x(x,i):
                    ks = k*np.sqrt(1 + r*r - 2*r*x)
                    W=win(k*r)
                    Ws=win(ks)
                    Bcomp=0.
                    if i == 1:
                        Bcomp=(-10*r**2*Ws+r*(7+17*r**2)*Ws*x+(7*(1+r**2)**2*W-r**2*(11+7*r**2)*Ws)*x**2-2*r*(7*(1+r**2)*W-2*r**2*Ws)*x**3)/(21.*r**2*(1+r**2-2*r*x)) #B00:b1**2
                    elif i == 2:
                        Bcomp=-(W*x)**2/(6.*r*r) #B00:b1**3
                    else:
                        Bcomp=0.

                    return Bcomp * 6 * r *r

                B_rec, B_err = quad(lambda x: fp13_int_x(x,i),xmin,xmax)

                Borg += B_rec

        else:  #space == 'z'
            if r < 5e-3:
                if i == 0:
                    Borg = -2./3. + 223./315.*r*r - 376./735.*r**4 #B00:b1
                elif i == 5:
                    Borg = -2./3. -8./105.*r**2 - 40./245.*r**4 #B11:nobias
                elif i == 6:
                    Borg = -4./3. + 240./105.*r**2 - 336./245.*r**4 #B11:b1
                elif i == 10:
                    Borg =  -32./35.*r**2 + 96./245.*r**4 #B12:nobias
                elif i == 11:
                    Borg = -2./3. #B12:b1
                elif i == 12:
                    Borg = -4./3. + 48./35.*r*r - 48./49.*r**4 #B22
                elif i == 15:
                    Borg = -2./3. #B23
                else:
                    Borg=0.

            elif r > 5e2:
                s = 1./r
                if i == 0:
                    Borg = -122./315. + 8./105.*s**2 - 40./1323.*s**4 #B00:b1
                elif i == 5:
                    Borg = -6./5.+104./245.*s*s #B11:nobias
                elif i == 6:
                    Borg = 4./105.-48./245.*s*s #B11:b1
                elif i == 10:
                    Borg = -32./35. + (96*s**2)/245. - (32*s**4)/735. #B12:nobias
                elif i == 11:
                    Borg = -2./3. #B12:b1
                elif i == 12:
                    Borg = -92./105. + 48./245.*s*s - 16./245.*s**4 #B22
                elif i == 15:
                    Borg = -2./3. #B23
                else:
                    Borg=0.

            elif r >= 0.995 and r <= 1.005:
                r1 = r-1.
                if i == 0:
                    Borg = -22./63. + 2./63.*r1 - 29./63.*r1*r1 #B00:b1
                elif i == 5:
                    Borg = -6./7. - 10./21.*r1 + r1*r1/21. #B11:nobias
                elif i == 6:
                    Borg = -4./21. + 4./7.*r1 -10./7.*r1*r1 #B11:b1
                elif i == 10:
                    Borg = -12./21. - 4./7.*r1 + 4./7.*r1*r1 #B12: nobias
                elif i == 11:
                    Borg = -2./3. #B12: b1
                elif i == 12:
                    Borg = (-16. - 18.*r1**2)/21. #B22
                elif i == 15:
                    Borg=-2./3. #B23
                else:
                    Borg=0.
            else:
                if i == 0:
                    Borg=(2*r*(6-79*r**2+50*r**4-21*r**6)+3*(-1+r**2)**3*(2+7*r**2)*np.log(np.abs((1+r)/(1-r))))/(252.*r**3) #B00:b1
                elif i == 5:
                    Borg=(12*r-82*r**3+4*r**5-6*r**7+3*(-1+r**2)**3*(2+r**2)*np.log(np.abs((1+r)/(1-r))))/(84.*r**3) #B11:nobias
                elif i == 6:
                    Borg=(-2*r*(19-24*r**2+9*r**4)+9*(-1+r**2)**3*np.log(np.abs((1+r)/(1-r))))/(42.*r) #B11:b1
                elif i == 10:
                    Borg=(2*r*(9-33*r**2-33*r**4+9*r**6)-9*(-1+r**2)**4*np.log(np.abs((1+r)/(1-r))))/(168.*r**3) #B12:nobias
                elif i == 11:
                    Borg=-2./3. #B12:b1
                elif i == 12:
                    Borg=(18./r/r - 218 + 126.*r*r - 54.*r**4 + 9./(r*r*r)*(r*r-1)**3*(3*r*r+1)*np.log(np.abs((1+r)/(1-r))))/168. #B22
                elif i == 15:
                    Borg=-2./3. #B23
                else:
                    Borg=0

            if rec == 'y':
                def fp13_rec_x(x,i):
                    ks = k*np.sqrt(1 + r*r - 2*r*x)
                    W=win(k*r)
                    Ws=win(ks)
                    if i == 1:
                        Bcomp=(-10*r**2*Ws+r*(7+17*r**2)*Ws*x+(7*(1+r**2)**2*W-r**2*(11+7*r**2)*Ws)*x**2-2*r*(7*(1+r**2)*W-2*r**2*Ws)*x**3)/(21.*r**2*(1+r**2-2*r*x)) #B00:b1**2
                    elif i == 2:
                        Bcomp=-(W*x)**2/(6.*r*r) #B00:b1**3
                    elif i == 3:
                        Bcomp=((-1+x**2)*(2*r**2*(1+r**2)*(5+8*r**2)*Ws-r*(7+38*r**2+49*r**4+30*r**6)*Ws*x-((1+r**2)*(7+28*r**2+23*r**4+14*r**6)*W-r**2*(11-10*r**2+21*r**4+14*r**6)*Ws)*x**2+2*r*((7+28*r**2+23*r**4+14*r**6)*W+r**2*(12+41*r**2+r**4)*Ws)*x**3+4*r**2*(1+r**2)*((7+11*r**2)*W-11*r**2*Ws)*x**4-8*r**3*((7+11*r**2)*W-2*r**2*Ws)*x**5))/(42.*r**2*(1+r**2-2*r*x)**2*(1+r**2+2*r*x)) #B01:b1
                    elif i == 4:
                        Bcomp=-((-1+x**2)*(-(r**2*Ws)+r**3*Ws*x-(1+r**2)*W**2*x**2+2*r*W**2*x**3))/(6.*r**2*(1+r**2-2*r*x)) #B01:b1**2
                    elif i == 6:
                        Bcomp=(2*r**2*(1+r**2)*(-1+8*r**2)*Ws+r*(7+2*r**2+r**4-30*r**6)*Ws*x+((7+7*r**2+15*r**4+r**6-14*r**8)*W+r**2*(-49-112*r**2-93*r**4+14*r**6)*Ws)*x**2+r*(2*(-7-15*r**4+14*r**6)*W+(21+98*r**2+201*r**4+120*r**6)*Ws)*x**3+((1+r**2)*(21+56*r**2+89*r**4+42*r**6)*W-r**2*(33-106*r**2+83*r**4+42*r**6)*Ws)*x**4-2*r*((21+56*r**2+89*r**4+42*r**6)*W+3*r**2*(12+49*r**2+r**4)*Ws)*x**5-12*r**2*(1+r**2)*((7+11*r**2)*W-11*r**2*Ws)*x**6+24*r**3*((7+11*r**2)*W-2*r**2*Ws)*x**7)/(42.*r**2*(1+r**2-2*r*x)**2*(1+r**2+2*r*x)) #B11:b1
                    elif i == 7:
                        Bcomp=(-(r**2*Ws)+r*(2+3*r**2)*Ws*x+(2*(1+r**2)**2*W-r**2*(5+2*r**2)*Ws)*x**2+r*(-4*(1+r**2)*W+3*r**2*Ws)*x**3-3*(1+r**2)*W**2*x**4+6*r*W**2*x**5)/(6.*r**2*(1+r**2-2*r*x)) #B11:b1**2
                    elif i == 8:
                        Bcomp=((-1+x**2)**2*(-6*r**2*(1+r**2)*Ws+r*(7+8*r**2+13*r**4)*Ws*x+((7+9*r**2+9*r**4+7*r**6)*W+r**2*(-1+4*r**2-7*r**4)*Ws)*x**2-2*r*((7+2*r**2+7*r**4)*W+r**2*(11+3*r**2)*Ws)*x**3-16*r**2*(W+r**2*W-r**2*Ws)*x**4+32*r**3*W*x**5))/(56.*(1+r**2-2*r*x)**2*(1+r**2+2*r*x)) #B02:nobias
                    elif i == 9:
                        Bcomp=((-1+x**2)**2*(-2*r**2*Ws+2*r**3*Ws*x-(1+r**2)*W**2*x**2+2*r*W**2*x**3))/(16.*r**2*(1+r**2-2*r*x)) #B02:b1
                    elif i == 10:
                        Bcomp=-((-1+x**2)*(14*W*x**2+21*r**8*(W-Ws)*x**2*(-1+5*x**2)+14*r*x*(Ws-2*W*x**2)-3*r**7*(13*Ws*x-(14*W+99*Ws)*x**3+10*(7*W+3*Ws)*x**5)+r**4*(6*Ws+5*(27*W-35*Ws)*x**2+(7*W+281*Ws)*x**4-240*W*x**6)+2*r**5*x*(37*Ws+(-80*W+63*Ws)*x**2-3*(2*W+87*Ws)*x**4+240*W*x**6)+r**6*(18*Ws+(59*W-272*Ws)*x**2+3*(37*W+28*Ws)*x**4+240*(-W+Ws)*x**6)+r**2*(W*x**2*(69+x**2)-2*Ws*(6+43*x**2))+r**3*(-2*W*x**3*(55+x**2)+Ws*x*(67+73*x**2))))/(84.*r**2*(1+r**2-2*r*x)**2*(1+r**2+2*r*x)) #B12:nobias
                    elif i == 11:
                        Bcomp=((-1+x**2)*(6*r**2*Ws-6*r*(2+5*r**2)*Ws*x+((1+r**2)*W*(-12-24*r**2+W)+6*r**2*(7+4*r**2)*Ws)*x**2-2*r*(W*(-12-24*r**2+W)+15*r**2*Ws)*x**3+15*(1+r**2)*W**2*x**4-30*r*W**2*x**5))/(24.*r**2*(1+r**2-2*r*x)) #B12:b1
                    elif i == 12:
                        Bcomp=(4*(-1+3*x**2)*((W*x*(7*x+r*(-6+7*r*x-8*x**2)))/(1+r**2-2*r*x)-(2*r*Ws*(-1+r*x)*(7*x+r*(-6+7*r*x-8*x**2)))/(1+r**2-2*r*x)**2+(W*x*(7*x+r*(6+7*r*x+8*x**2)))/(1+r**2+2*r*x))+r**2*(3-30*x**2+35*x**4)*((W*x*(7*x+r*(-6+7*r*x-8*x**2)))/(1+r**2-2*r*x)-(2*r*Ws*(-1+r*x)*(7*x+r*(-6+7*r*x-8*x**2)))/(1+r**2-2*r*x)**2+(W*x*(7*x+r*(6+7*r*x+8*x**2)))/(1+r**2+2*r*x))+8*r*x*(-3+5*x**2)*((W*x*(6*r-7*(1+r**2)*x+8*r*x**2))/(1+r**2-2*r*x)+(2*r*Ws*(-1+r*x)*(7*x+r*(-6+7*r*x-8*x**2)))/(1+r**2-2*r*x)**2+(W*x*(7*x+r*(6+7*r*x+8*x**2)))/(1+r**2+2*r*x)))/(336.*r**2) #B22:nobias
                    elif i == 13:
                        Bcomp=(18*r**2*Ws-2*r*(4+33*r**2)*Ws*x+((1+r**2)*W*(-8-48*r**2+5*W)+4*r**2*(-1+12*r**2)*Ws)*x**2+2*r*((8+48*r**2-5*W)*W+20*Ws+46*r**2*Ws)*x**3+2*((1+r**2)*W*(20+40*r**2+3*W)-5*r**2*(11+8*r**2)*Ws)*x**4-2*r*(2*W*(20+40*r**2+3*W)-35*r**2*Ws)*x**5-35*(1+r**2)*W**2*x**6+70*r*W**2*x**7)/(48.*r**2*(1+r**2-2*r*x)) #B22:b1
                    elif i == 14:
                        Bcomp=-((-1+x**2)**2*(4*r**2*Ws-14*r**3*Ws*x+(r**2*(-10+W)*W+W**2+10*r**4*(-W+Ws))*x**2+2*r*(10*r**2-W)*W*x**3))/(16.*r**2*(1+r**2-2*r*x)) #B13
                    elif i == 15:
                        Bcomp=-(-3*(1-6*x**2+5*x**4)*(W**2*x**2*(1+r**2-2*r*x)-4*r**2*Ws*(-1+r*x))-12*x*(1-x**2)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))+10*r**2*x*(3-10*x**2+7*x**4)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x)))/(24.*r**2*(1+r**2-2*r*x)) #B23
                    elif i == 16:
                        Bcomp=-((3-30*x**2+35*x**4)*(W**2*x**2*(1+r**2-2*r*x)-4*r**2*Ws*(-1+r*x))+8*x*(3-5*x**2)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))-2*r**2*x*(15-70*x**2+63*x**4)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x)))/(48.*r**2*(1+r**2-2*r*x)) #B33
                    else:
                        Bcomp=0.

                    return Bcomp * 6 * r *r

                B_rec, B_err = quad(lambda x: fp13_rec_x(x,i),xmin,xmax)

                Borg += B_rec

        logkr = np.log(k*r)
        pkr = np.exp(interp1d_pkl(logkr))

        return pkr * Borg

    def f22(r,i):
    
        xmin=np.maximum(-1,(1+r*r-rmax*rmax)/2./r)
        xmax=np.minimum(1,(1+r*r-rmin*rmin)/2./r)
    
        if space == 'r':
            if rec == 'y':
                def fp22(x,i):
                    ks = k*np.sqrt(1 + r*r - 2*r*x)
                    fr = ((3*r + 7*x - 10*r*x*x - 7*win(ks)*r*(1-r*x))/(1+r*r-2*r*x)-7*win(k*r)*x)**2/98.
                    logks = np.log(ks)
                    pks = np.exp(interp1d_pkl(logks))
                    if i == 0:
                        return pks*fr
                    else:
                        return 0.
            else:
                def fp22(x,i):
                    ks = k*np.sqrt(1 + r*r - 2*r*x)
                    fr = ((3*r + 7*x - 10*r*x*x)/(1+r*r-2*r*x))**2/98.
                    logks = np.log(ks)
                    pks = np.exp(interp1d_pkl(logks))
                    if i == 0:
                        return pks*fr
                    else:
                        return 0.
        else: #space == 'z'
            ### A00: 0:b1**2,1:b1**3,2:b1**4
            ### A01: 3:b1**2,4:b1**3
            ### A11: 5:b1,6:b1**2,7:b1**3
            ### A02: 8:b1,9:b1**2
            ### A12: 10:b1,11:b1**2
            ### A22: 12:nobias,13:b1,14:b1**2
            ### A03: 15:b1
            ### A13: 16:nobias,17:b1
            ### A23: 18:nobias,19:b1
            ### A33: 20:nobias,21:b1
            ### A04: 22:nobias
            ### A14: 23:nobias
            ### A24: 24:nobias
            ### A34: 25:nobias
            ### A44: 26:nobias
            if rec == 'y':
                def fp22(x,i):
                    ks = k*np.sqrt(1 + r*r - 2*r*x)
                    W=win(k*r)
                    Ws=win(ks)

                    Acomp=0.
                    if i == 0:
                        Acomp=(7*x+r*(3-10*x**2))**2/(196.*r**2*(1+r**2-2*r*x)**2) #A00:b1**2
                    elif i == 1:
                        Acomp=((W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))*(-7*x+r*(-3+10*x**2)))/(14.*r**2*(1+r**2-2*r*x)**2) #A00:b1**3
                    elif i == 2:
                        Acomp=(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))**2/(4.*r**2*(1+r**2-2*r*x)**2) #A00:b1**4
                    elif i == 3:
                        Acomp=-((1+2*r*(r-x))*(-1+x**2)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))*(-7*x+r*(-3+10*x**2)))/(28.*r**2*(1+r**2-2*r*x)**3) #A01:b1**2
                    elif i == 4:
                        Acomp=-((1+2*r*(r-x))*(-1+x**2)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))**2)/(4.*r**2*(1+r**2-2*r*x)**3) #A01:b1**3
                    elif i == 5:
                        Acomp=((r-7*x+6*r*x**2)*(-7*x+r*(-3+10*x**2)))/(98.*r**2*(1+r**2-2*r*x)**2) #A11:b1
                    elif i == 6:
                        Acomp=(2*r**5*(W-Ws)*x*(4-13*x**2+30*x**4)+r*x*(20-W-21*Ws+(-76+97*W-21*Ws)*x**2+114*W*x**4)+r**2*(6-Ws+(-58-17*W+76*Ws)*x**2+(136-199*W+93*Ws)*x**4-204*W*x**6)+r**3*x*(8+7*W+3*Ws+(16+25*W-123*Ws)*x**2+4*(-20+67*W-33*Ws)*x**4+120*W*x**6)+2*r**4*(3+4*Ws-(16+7*W+14*Ws)*x**2+4*(5-2*W+16*Ws)*x**4+30*(-3*W+Ws)*x**6)-7*x**2*(-2+3*W*(1+x**2)))/(28.*r**2*(1+r**2-2*r*x)**3) #A11:b1**2
                    elif i == 7:
                        Acomp=((W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))*(2*r**4*(W-Ws)*x*(-1+3*x**2)+x*(-2+W+3*W*x**2)+r*(-2+Ws+(8-4*W+3*Ws)*x**2-12*W*x**4)+r**2*x*(2-W-3*Ws+(-8+13*W-9*Ws)*x**2+12*W*x**4)+2*r**3*(-1-Ws+(2+W+4*Ws)*x**2+3*(-3*W+Ws)*x**4)))/(4.*r**2*(1+r**2-2*r*x)**3) #A11:b1**3
                    elif i == 8:
                        Acomp=(3*(-1+x**2)**2*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))*(-7*x+r*(-3+10*x**2)))/(112.*(1+r**2-2*r*x)**3) #A02:b1
                    elif i == 9:
                        Acomp=(3*(-1+x**2)**2*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))**2*(1+2*r*(-2*x+r*(3+3*r**2-6*r*x+2*x**2))))/(32.*r**2*(1+r**2-2*r*x)**4) #A02:b1*b1
                    elif i == 10:
                        Acomp=((-1+x**2)*(28*W*x**2-r**5*(W-Ws)*x*(13-51*x**2+150*x**4)+r**3*x*(2-9*W-29*Ws+(40-43*W+393*Ws)*x**2-648*W*x**4)+r**2*(6+4*Ws-(48+5*W+172*Ws)*x**2+537*W*x**4)+r**4*(6-13*Ws+(-20+W+76*Ws)*x**2+21*(7*W-19*Ws)*x**4+300*W*x**6)+2*r*x*(7+14*Ws+W*(2-100*x**2))))/(56.*r**2*(1+r**2-2*r*x)**3) #A12:b1
                    elif i == 11:
                        Acomp=-((-1+x**2)*(2-12*W*x**2+18*r**8*(W-Ws)**2*x**2*(-1+5*x**2)+5*W**2*(x**2+3*x**4)+r**4*(2-3*(-16+36*W+3*W**2)*x**2+(32-640*W+813*W**2)*x**4+48*W*(-4+37*W)*x**6+240*W**2*x**8+Ws**2*(2+283*x**2+195*x**4)-2*Ws*(-2+(-122+99*W)*x**2+(-120+881*W)*x**4+300*W*x**6))+r**2*(5*Ws**2*(1+3*x**2)+Ws*(8+(84-134*W)*x**2-210*W*x**4)+4*(1+3*(4-8*W+W**2)*x**2+36*W*(-2+3*W)*x**4+90*W**2*x**6))-2*r*x*(8+6*Ws+12*W**2*x**2*(3+5*x**2)-W*(4+48*x**2+5*Ws*(1+3*x**2)))-4*r**7*x*(9*W**2*x**2*(-1+15*x**2)+Ws*(-1+12*x**2+9*Ws*(-1+6*x**2+5*x**4))+W*(1-12*x**2-9*Ws*(-1+5*x**2+20*x**4)))-2*r**3*x*(4*(3+8*x**2)+Ws**2*(31+45*x**2)+2*Ws*(19+54*x**2)+4*W**2*x**2*(21+158*x**2+60*x**4)-W*(6*(1+32*x**2+32*x**4)+Ws*(7+355*x**2+270*x**4)))+2*r**6*(Ws**2*(-9+82*x**2+267*x**4+30*x**6)+Ws*(-2+8*(5+2*W)*x**2+(72-696*W)*x**4-420*W*x**6)+x**2*(4-12*W*(1+10*x**2)+W**2*(-17+213*x**2+570*x**4)))+4*r**5*x*(-2+(-8+72*W-15*W**2)*x**2+3*(32-135*W)*W*x**4-240*W**2*x**6-5*Ws**2*(2+29*x**2+9*x**4)+Ws*(-9-80*x**2-24*x**4+W*(-8+214*x**2+504*x**4+60*x**6)))))/(16.*r**2*(1+r**2-2*r*x)**4) #A12:b1*b1
                    elif i == 12:
                        Acomp=(r-7*x+6*r*x**2)**2/(196.*r**2*(1+r**2-2*r*x)**2) #A22:nobias
                    elif i == 13:
                        Acomp=(56*x**2*(2-3*W*x**2)+r**5*(W-Ws)*x*(-17+96*x**2-261*x**4+350*x**6)+r**2*(4-7*(28+11*W)*x**2+(864+466*W)*x**4-2069*W*x**6+8*Ws*(2-19*x**2+101*x**4))+4*r*x*(25-(137+42*Ws)*x**2+W*(4-38*x**2+244*x**4))+r**4*(4-(108+11*W)*x**2+6*(36+23*W)*x**4-267*W*x**6-700*W*x**8+Ws*(-17+141*x**2-591*x**4+1139*x**6))+r**3*x*(92-108*x**2-432*x**4+Ws*(-61+482*x**2-1429*x**4)+W*(-1+34*x**2-281*x**4+1928*x**6)))/(112.*r**2*(1+r**2-2*r*x)**3) #A22:b1
                    elif i == 14:
                        Acomp=(-4+(12-8*W-5*W**2)*x**2+2*W*(-20+9*W)*x**4+35*W**2*x**6+6*r**8*(W-Ws)**2*x**2*(3-30*x**2+35*x**4)+2*r**6*(4+(-24+32*W+7*W**2)*x**2-8*(-3-25*W+60*W**2)*x**4-5*W*(80+37*W)*x**6+1330*W**2*x**8+Ws**2*(9-209*x**2-135*x**4+625*x**6+70*x**8)-4*Ws*(-1+(25-19*W)*x**2-279*W*x**4+5*(-12+61*W)*x**6+245*W*x**8))+r**4*(12+(-180+280*W-31*W**2)*x**2+(96+600*W-1498*W**2)*x**4+(192-1920*W+169*W**2)*x**6+320*W*(-2+13*W)*x**8+560*W**2*x**10+Ws**2*(-22-413*x**2+700*x**4+455*x**6)-2*Ws*(12+(224-263*W)*x**2-2*(158+427*W)*x**4+(-400+2097*W)*x**6+700*W*x**8))-2*r*x*(4*(-6+Ws+12*x**2+5*Ws*x**2)+4*W**2*x**2*(-13+26*x**2+35*x**4)-W*(8*(-2+3*x**2+20*x**4)+Ws*(-5+18*x**2+35*x**4)))-2*r**3*x*(8*(-3-11*x**2+24*x**4)+4*Ws*(-23+23*x**2+90*x**4)+Ws**2*(-47+86*x**2+105*x**4)+8*W**2*x**2*(-24-67*x**2+189*x**4+70*x**6)+W*(28+284*x**2-512*x**4-640*x**6+Ws*(27+484*x**2-889*x**4-630*x**6)))+r**2*(Ws**2*(-5+18*x**2+35*x**4)+Ws*(-32+2*(20+99*W)*x**2+(280-380*W)*x**4-490*W*x**6)+8*x**2*(-21+36*x**2-2*W*(-13+16*x**2+60*x**4)+W**2*(-4-70*x**2+137*x**4+105*x**6)))-4*r**7*x*(3*W**2*x**2*(-3-70*x**2+105*x**4)+Ws*(2-30*x**2+40*x**4+3*Ws*(3-39*x**2+25*x**4+35*x**6))+W*(-2+30*x**2-40*x**4-3*Ws*(3-42*x**2-45*x**4+140*x**6)))-4*r**5*x*(12*x**2*(-3+4*x**2)+2*Ws*(-13-57*x**2+120*x**4+40*x**6)+Ws**2*(-38-171*x**2+344*x**4+105*x**6)+W**2*x**2*(-79-554*x**2+745*x**4+560*x**6)-2*W*(-2-72*x**2+40*x**4+160*x**6+Ws*(-1-218*x**2-21*x**4+590*x**6+70*x**8))))/(32.*r**2*(1+r**2-2*r*x)**4) #A22:b1*b1
                    elif i == 15:
                        Acomp=(-5*(1+2*r*(r-x))*(-1+x**2)**3*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))**2)/(32.*(1+r**2-2*r*x)**4) #A03:b1
                    elif i == 16:
                        Acomp=(3*(-1+x**2)**2*(r-7*x+6*r*x**2)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x)))/(112.*(1+r**2-2*r*x)**3) #A13:nobias
                    elif i == 17:
                        Acomp=(3*(-1+x**2)**2*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))*(2*W*x+10*r**6*(W-Ws)*x*(-1+7*x**2)+2*r*(3+Ws-14*W*x**2)+r**3*(8+9*Ws+(64-92*W+99*Ws)*x**2-220*W*x**4)+r**4*x*(-26-W-59*Ws+5*(-8+49*W-29*Ws)*x**2+140*W*x**4)+2*r**5*(1-5*Ws+5*(2-3*W+12*Ws)*x**2+35*(-3*W+Ws)*x**4)+r**2*x*(-34-26*Ws+W*(11+123*x**2))))/(32.*r**2*(1+r**2-2*r*x)**4) #A13:b1
                    elif i == 18:
                        Acomp=((-1+x**2)*(r-7*x+6*r*x**2)*(-2*W*x-3*r**4*(W-Ws)*x*(-1+5*x**2)+r**2*x*(4+W+14*Ws-39*W*x**2)-2*r*(1+Ws-8*W*x**2)+r**3*(-2+Ws*(3-27*x**2)+6*W*(x**2+5*x**4))))/(56.*r**2*(1+r**2-2*r*x)**3) #A23:nobias
                    elif i == 19:
                        Acomp=((-1+x**2)*(-8+48*W*x**2+W**2*(4*x**2-60*x**4)-30*r**8*(W-Ws)**2*x**2*(1-14*x**2+21*x**4)+2*r**3*x*(40+176*x**2+4*Ws**2*(-7+65*x**2)+8*Ws*(-6+89*x**2)+W*Ws*(73+122*x**2-2195*x**4)-2*W*(-13+67*x**2+846*x**4)+W**2*x**2*(-359+574*x**2+2865*x**4))+r**2*(-8+(-240-68*W+77*W**2)*x**2+2*W*(898+59*W)*x**4-2715*W**2*x**6+Ws**2*(4-60*x**2)+20*Ws*(1-3*(7+2*W)*x**2+58*W*x**4))+r**6*(8+3*(-16-44*W+3*W**2)*x**2+10*W*(26+207*W)*x**4-35*W*(-32+61*W)*x**6-5040*W**2*x**8-Ws**2*(30-1149*x**2+1440*x**4+2135*x**6)+2*Ws*(6+(84-309*W)*x**2-30*(11+51*W)*x**4+35*(-8+121*W)*x**6+1260*W*x**8))+r**4*(Ws**2*(69+130*x**2-1735*x**4)+2*Ws*(16+(14-589*W)*x**2+2*(-583+347*W)*x**4+4055*W*x**6)-4*(-2+(56+62*W-28*W**2)*x**2+(48-382*W-552*W**2)*x**4+5*W*(-156+277*W)*x**6+1505*W**2*x**8))+4*r*x*(6*(3+2*Ws)+16*W**2*x**2*(-1+10*x**2)+W*(5-117*x**2+Ws*(2-30*x**2)))+2*r**5*x*(4*W**2*x**2*(-108-235*x**2+1120*x**4+315*x**6)+2*(2+48*x**2+Ws*(-35+107*x**2+460*x**4)+Ws**2*(-117+78*x**2+695*x**4))+W*(22+114*x**2-1160*x**4-560*x**6+Ws*(39+1626*x**2-3105*x**4-3640*x**6)))+2*r**7*x*(105*W**2*x**2*(-1-6*x**2+15*x**4)+Ws*(-6-30*x**2+140*x**4+15*Ws*(2-39*x**2+56*x**4+21*x**6))-2*W*(-3-15*x**2+70*x**4+15*Ws*(1-23*x**2+7*x**4+63*x**6)))))/(32.*r**2*(1+r**2-2*r*x)**4) #23:b1
                    elif i == 20:
                        Acomp=-((r-7*x+6*r*x**2)*(4*x*(2+W-3*W*x**2)-r**4*(W-Ws)*x*(3-30*x**2+35*x**4)+r**2*x*(W-28*Ws+24*x**2+66*W*x**2+52*Ws*x**2-115*W*x**4)+4*r*(1+Ws-(7+8*W)*x**2-3*Ws*x**2+16*W*x**4)-r**3*(-4+6*(2+3*W)*x**2+20*W*x**4-70*W*x**6+Ws*(3-54*x**2+75*x**4))))/(112.*r**2*(1+r**2-2*r*x)**3) #A33:nobias
                    elif i == 21:
                        Acomp=(-8+(24+32*W-2*W**2)*x**2-4*W*(20+9*W)*x**4+70*W**2*x**6+2*r**8*(W-Ws)**2*x**2*(-5+105*x**2-315*x**4+231*x**6)+r**2*(-8+(-280+38*W+33*W**2)*x**2+(528+1300*W-371*W**2)*x**4-W*(2346+1349*W)*x**6+2583*W**2*x**8+Ws**2*(-2-36*x**2+70*x**4)+Ws*(2+12*(-27+5*W)*x**2+(610+696*W)*x**4-1204*W*x**6))-2*r**7*x*(-3*W*(1-35*x**4+42*x**6)+3*Ws*(1-35*x**4+42*x**6)+W**2*x**2*(65+105*x**2-1197*x**4+1155*x**6)+W*Ws*(10-360*x**2+840*x**4+672*x**6-1386*x**8)+Ws**2*(-10+295*x**2-945*x**4+525*x**6+231*x**8))+r**4*(8+(-312-36*W+62*W**2)*x**2+4*(56+400*W+129*W**2)*x**4+(320-52*W-5590*W**2)*x**6+56*W*(-57+44*W)*x**8+4788*W**2*x**10+Ws**2*(37-245*x**2-901*x**4+1589*x**6)+Ws*(8-2*(101+255*W)*x**2+18*(-82+171*W)*x**4+(2630+2094*W)*x**6-6902*W*x**8))+r**6*(8+(-72-42*W+17*W**2)*x**2+5*(16+84*W+173*W**2)*x**4-7*W*(-42+475*W)*x**6-21*W*(48+17*W)*x**8+3696*W**2*x**10+Ws**2*(-10+587*x**2-2085*x**4+329*x**6+1659*x**8)+2*Ws*(3-3*(-5+59*W)*x**2-5*(63+29*W)*x**4+7*(27+487*W)*x**6-63*(-4+45*W)*x**8-924*W*x**10))+2*r*x*(44-92*x**2-8*Ws*(-2+5*x**2)+16*W**2*x**2*(1+12*x**2-21*x**4)+W*(1-178*x**2+345*x**4+Ws*(-2-36*x**2+70*x**4)))-2*r**3*x*(W**2*x**2*(131-1065*x**2-655*x**4+2485*x**6)+2*(4*(-7-15*x**2+42*x**4)+Ws**2*(-7-78*x**2+133*x**4)+Ws*(-2-272*x**2+454*x**4))+W*(-5+252*x**2+871*x**4-1958*x**6+Ws*(-35+307*x**2+1107*x**4-2051*x**6)))-2*r**5*x*(4*(-3-25*x**2+40*x**4)+2*Ws**2*(61-261*x**2-221*x**4+581*x**6)+Ws*(19-306*x**2-277*x**4+924*x**6)+4*W**2*x**2*(53-95*x**2-742*x**4+777*x**6+231*x**8)-W*(7-66*x**2-865*x**4+924*x**6+504*x**8+Ws*(27+575*x**2-3535*x**4+1197*x**6+2856*x**8))))/(32.*r**2*(1+r**2-2*r*x)**4) #A33:b1
                    elif i == 22:
                        Acomp=(35*r**2*(-1+x**2)**4*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))**2)/(512.*(1+r**2-2*r*x)**4) #A04
                    elif i == 23:
                        Acomp=(-5*(-1+x**2)**3*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))*(12*W*x+7*r**4*(W-Ws)*x*(-1+9*x**2)+4*r*(1+3*Ws-20*W*x**2)+r**2*x*(-8-68*Ws+5*W*(1+35*x**2))+r**3*(4+7*Ws*(-1+17*x**2)-42*W*(x**2+3*x**4))))/(128.*(1+r**2-2*r*x)**4) #A14
                    elif i == 24:
                        Acomp=(3*(-1+x**2)**2*(8*W**2*x**2+16*r*W*x*(3+Ws-12*W*x**2)+35*r**8*(W-Ws)**2*x**2*(1-18*x**2+33*x**4)+8*r**3*x*(-4+7*W-48*Ws-28*W*Ws-22*Ws**2+W*(179+86*W+334*Ws)*x**2-710*W**2*x**4)+8*r**2*(1+6*Ws+Ws**2-W*(54+13*W+46*Ws)*x**2+189*W**2*x**4)+10*r**7*(W-Ws)*x*(-4+7*Ws+14*(2+3*W-13*Ws)*x**2+21*(4*W+19*Ws)*x**4-462*W*x**6)+2*r**5*x*(-16-16*W-84*Ws-85*W*Ws+400*Ws**2+10*(65*W**2+W*(68-155*Ws)-22*Ws*(3+8*Ws))*x**2+35*W*(16-28*W+225*Ws)*x**4-5670*W**2*x**6)+r**4*(8*(2+Ws-15*Ws**2)+(32-197*W**2+32*W*(-16+47*Ws)+8*Ws*(137+146*Ws))*x**2-10*W*(208+103*W+904*Ws)*x**4+11235*W**2*x**6)+r**6*(8-10*W*(8+5*W)*x**2+5*(Ws*(-8+7*Ws)+2*(44+115*W-187*Ws)*Ws*x**2+7*(-8*W*(4+9*W)+4*(4+7*W)*Ws+153*Ws**2)*x**4+42*W*(31*W-65*Ws)*x**6+924*W**2*x**8))))/(256.*r**2*(1+r**2-2*r*x)**4) #A24
                    elif i == 25:
                        Acomp=(16*(1+r**2-2*r*x)**2-16*x**2*(1+r**2-2*r*x)**2+60*r**2*(1-15*x**2+35*x**4-21*x**6)*(r**3+3*W*x+r**2*(-2+3*W-3*Ws)*x+r*(1+3*Ws-6*W*x**2))*(W*x+r**2*(W-Ws)*x+r*(Ws-2*W*x**2))+80*r*x*(3-10*x**2+7*x**4)*(3*r**3+2*W*x+2*r**2*(-3+W-Ws)*x+r*(3+2*Ws-4*W*x**2))*(W*x+r**2*(W-Ws)*x+r*(Ws-2*W*x**2))+168*r**3*x*(-5+35*x**2-63*x**4+33*x**6)*(W*x+r**2*(W-Ws)*x+r*(Ws-2*W*x**2))**2-7*r**4*(5-140*x**2+630*x**4-924*x**6+429*x**8)*(W*x+r**2*(W-Ws)*x+r*(Ws-2*W*x**2))**2-32*x*(1+r**2-2*r*x)*(3-3*x**2)*(r**3+W*x+r**2*(-2+W-Ws)*x+r*(1+Ws-2*W*x**2))-24*(1-6*x**2+5*x**4)*(r**2*(1+r**2-2*r*x)**2+(W*x+r*W*(r-2*x)*x-r*Ws*(-1+r*x))**2-4*r*(1+r**2-2*r*x)*(-(W*x)-r*W*(r-2*x)*x+r*Ws*(-1+r*x))+2*r*(1+r**2-2*r*x)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))))/(128.*r**2*(1+r**2-2*r*x)**4) #A34
                    elif i == 26:
                        Acomp=(-64*(1+r**2-2*r*x)**2+192*x**2*(1+r**2-2*r*x)**2+r**4*(35-1260*x**2+6930*x**4-12012*x**6+6435*x**8)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))**2-16*r**2*(5-105*x**2+315*x**4-231*x**6)*(r**3+3*W*x+r**2*(-2+3*W-3*Ws)*x+r*(1+3*Ws-6*W*x**2))*(W*x+r**2*(W-Ws)*x+r*(Ws-2*W*x**2))-32*r*x*(15-70*x**2+63*x**4)*(3*r**3+2*W*x+2*r**2*(-3+W-Ws)*x+r*(3+2*Ws-4*W*x**2))*(W*x+r**2*(W-Ws)*x+r*(Ws-2*W*x**2))+32*r**3*x*(35-315*x**2+693*x**4-429*x**6)*(W*x+r**2*(W-Ws)*x+r*(Ws-2*W*x**2))**2+128*x*(1+r**2-2*r*x)*(3-5*x**2)*(r**3+W*x+r**2*(-2+W-Ws)*x+r*(1+Ws-2*W*x**2))+16*(3-30*x**2+35*x**4)*(r**2*(1+r**2-2*r*x)**2+(W*x+r*W*(r-2*x)*x-r*Ws*(-1+r*x))**2-4*r*(1+r**2-2*r*x)*(-(W*x)-r*W*(r-2*x)*x+r*Ws*(-1+r*x))+2*r*(1+r**2-2*r*x)*(W*x+r*(Ws-r*Ws*x+W*(r-2*x)*x))))/(512.*r**2*(1+r**2-2*r*x)**4) #A44
                    else:
                        Acomp = 0.

                    Acomp *= 2*r*r
                    logks = np.log(ks)
                    pks = np.exp(interp1d_pkl(logks))

                    return pks*Acomp

            else:
                def fp22(x,i):
                    Acomp=0.
                    if i == 0:
                        Acomp=(7*x+r*(3-10*x**2))**2/(196.*r**2) #A00:b1**2
                    elif i == 5:
                        Acomp=((r-7*x+6*r*x**2)*(-7*x+r*(-3+10*x**2)))/(98.*r**2) #A11:b1
                    elif i == 6:
                        Acomp=((-x+r*(-1+2*x*x))*(-7*x+r*(-3+10*x*x)))/(14.*r**2) #A11:b1**2
                    elif i == 10:
                        Acomp=((-1+x*x)*(7*x+r*(3-10*x*x)))/(28.*r) #A12:b1
                    elif i == 11:
                        Acomp=-((1-2*r*x)**2*(-1+x*x))/(8.*r**2) #A12:b1**2
                    elif i == 12:
                        Acomp=(r-7*x+6*r*x**2)**2/(196.*r**2) #A22:nobias
                    elif i == 13:
                        Acomp=(28*x*x+r*(25*x-81*x**3)+r*r*(1-27*x*x+54*x**4))/(28.*r*r) #A22:b1
                    elif i == 14:
                        Acomp=(-1+3*x*x+r*(8*x-12*x**3)+2*r*r*(1-6*x*x+6*x**4))/(8.*r*r) #A22:b1**2
                    elif i == 18:
                        Acomp=-((-1+x*x)*(r-7*x+6*r*x*x))/(28.*r)  #A23:nobias
                    elif i == 19:
                        Acomp=-((-1+x*x)*(1-5*r*x+r*r*(-1+6*x*x)))/(4.*r*r)  #A23:b1
                    elif i == 20:
                        Acomp=(14*x*x+r*(5*x-33*x**3)+r**2*(-1-3*x*x+18*x**4))/(28.*r*r) #A33:nobias
                    elif i == 21:
                        Acomp=(-1+3*x*x+r*(7*x-11*x**3)+r**2*(1-9*x*x+10*x**4))/(4.*r*r) #A33:b1
                    elif i == 24:
                        Acomp=(3*(-1+x*x)**2)/32. #A24
                    elif i == 25:
                        Acomp=-((-1+x*x)*(2-12*r*x+3*r**2*(-1+5*x*x)))/(16.*r*r) #A34
                    elif i == 26:
                        Acomp=(-4+12*x*x-8*r*x*(-3+5*x*x)+r*r*(3.-30*x*x+35*x**4))/(32.*r*r) #A44
                    else:
                        Acomp=0.

                    Acomp*=2*r*r/(1+r*r-2*r*x)**2

                    ks = k*np.sqrt(1 + r*r - 2*r*x)
                    logks = np.log(ks)
                    pks = np.exp(interp1d_pkl(logks))

                    return pks*Acomp

        g, gerr = quad(lambda x: fp22(x,i),xmin,xmax)

        logkr = np.log(k*r)

        pkr = np.exp(interp1d_pkl(logkr))

        return pkr*g

    if rec == 'y':
        np22=1000
    else:
        np22=4000

    B_int, B_int_err = quad(lambda r: f13(r,16),rmin,rmax)
    #print(B_int*pklout*k**3/(4*np.pi**2))
    #sys.exit()

    Bterm = np.zeros(nbcomp)
    for i in range(nbcomp):
        B_int, B_int_err = quad(lambda r: f13(r,i),rmin,rmax)
        Bterm[i] = B_int*pklout*k**3/(4*np.pi**2)
        print(i,Bterm[i])

    p22out=np.zeros(np22)
    Aterm=np.zeros(nacomp)

    #        for i in range(1):
    #            #A_int, A_int_err = quad(lambda r: r*f22(r,i),np.log(rmin),np.log(rmax))
    #            A_int, A_int_err = quad(lambda r: r*f22(r,i),np.log(rmin),-0.1)
    #            A_int2, A_int_err = quad(lambda r: r*f22(r,i),0.1,np.log(rmax))
    #            Aterm[i] = (A_int+A_int2)*k**3/(4*np.pi**2)
    #            print(i,Aterm[i])
    #        sys.exit()

    for i in range(nacomp):
        ra=np.logspace(np.log(rmin),np.log(rmax),np22,True,base=np.e)
        for ir, r in enumerate(ra):
            ### multiplying r is for integrating in logarithmic scale
            p22out[ir]=r*f22(r,i)*k**3/(4*np.pi**2)
        ra=np.log(ra)
        Aterm[i] = integrate.simps(p22out,ra)

    outval[index_k,0] = k
    outval[index_k,1] = pklout
    for i in range(nbcomp):
        outval[index_k,i+2] = Bterm[i]
    for i in range(nacomp):
        outval[index_k,i+2+nbcomp] = Aterm[i]

    #print(outval[index_k,:])

    np.savetxt(outf, outval[0:index_k+1,:])


