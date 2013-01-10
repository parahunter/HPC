#!/bin/sh
#PBS -N s120929HPC
#PBS -q hpc
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:10:00

date 

python ./test_littleN.py

date
exit 0
