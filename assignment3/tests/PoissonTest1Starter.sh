#!/bin/sh
#
#PBS -N PoissonTest1
#PBS -l walltime=00:10:00
#PBS -q 02614
#PBS -l nodes=1:ppn=8:gpus=1

cd $PBS_O_WORKDIR

module load cuda

python ./PoissonTest1.py
