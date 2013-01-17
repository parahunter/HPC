#!/bin/sh
#PBS -N assignment2_convergence
#PBS -q hpc
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:10:00
#PBS -e out.err
#PBS -o out.out

date 

cd $PBS_O_WORKDIR
python ./ompScaling.py

date
exit 0
