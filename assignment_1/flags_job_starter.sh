#!/bin/sh
#PBS -N flagtest
#PBS -q hpc
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:10:00
date 

#cd $PBS_O_WORKDIR

python ./test-cpu-flags.py

date
exit 0
