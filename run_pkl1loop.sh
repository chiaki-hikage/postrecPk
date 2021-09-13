#!/bin/bash

npro=1

### 1. output redshift
zout=0

### 2. r-space or z-space (r or z)
space='z'

### 3. reconstruction scale [unit:Mpc/h] (rs=0 means no reconstruction)
rs=10

### 4. linear bias
blin=1

### 5. input linear power spectrum at z=0
infpk='prepost2021_PLC18_matterpower-z0p0-noneutrino.dat'

### 6. output directory
outdir='output/'

if [ ! -d $outdir ]; then
mkdir $outdir
fi

### 7. output filename for 1-loop coefficients A_nml and B_nml
if [ $rs == '0' ]; then
    outf_1loopcomp=${outdir}/1loopcomp_${space}space_pre.dat
else
    outf_1loopcomp=${outdir}/1loopcomp_${space}space_postR${rs}.dat
fi

#python calc_1loopcomp.py $space $rs $infpk $outf_1loopcomp

### 8. output filename for P_l(k) upto 1-loop
if [ $rs == '0' ]; then
    outf_pkl1loop=${outdir}/pkl_${space}space_pre_z${zout}.dat
else
    outf_pkl1loop=${outdir}/pkl_${space}space_postR${rs}_z${zout}.dat
fi

inf_1loopcomp=$outf_1loopcomp

python calc_pkl.py $space $rs $zout $blin $inf_1loopcomp $outf_pkl1loop
